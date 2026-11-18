"""Compare five probabilistic learners, then tune the winner with grouped TPE."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from pathsnap.data import DEFAULT_DATA, read_raw, trip_ids
from pathsnap.features import CANDIDATE_COUNT, FEATURE_NAMES
from pathsnap.metrics import classification, predict_scores

SEED = 352
FOLDS = 3
TRAIN_CAP = 950
TRIALS = 14
MODEL_INFO = {
    "Logistic regression": (
        "Linear", {"C": 1., "max_iter": 300},
        "Regularised binary log loss",
    ),
    "Gaussian naive Bayes": (
        "Probabilistic", {"var_smoothing": 1e-9},
        "Gaussian class-conditional likelihood",
    ),
    "Decision tree": (
        "Tree", {"max_depth": 11, "min_samples_leaf": 24},
        "Gini impurity",
    ),
    "Random forest": (
        "Bagging", {"n_estimators": 60, "max_depth": 15, "min_samples_leaf": 6, "max_features": "sqrt"},
        "Gini impurity across bootstrapped trees",
    ),
    "Histogram gradient boosting": (
        "Boosting", {"max_iter": 110, "learning_rate": .09, "max_leaf_nodes": 25,
                     "min_samples_leaf": 32, "l2_regularization": 1.0},
        "Binary log loss",
    ),
}
SEARCH = {
    "Logistic regression": {"C": ["float", .01, 20., True]},
    "Gaussian naive Bayes": {"var_smoothing": ["float", 1e-11, .1, True]},
    "Decision tree": {"max_depth": ["int", 6, 23], "min_samples_leaf": ["int", 8, 80]},
    "Random forest": {"n_estimators": ["int", 40, 100], "max_depth": ["int", 10, 23],
                      "min_samples_leaf": ["int", 3, 28], "max_features": ["float", .25, .9, False]},
    "Histogram gradient boosting": {
        "max_iter": ["int", 80, 210], "learning_rate": ["float", .035, .18, True],
        "max_leaf_nodes": ["int", 12, 48], "min_samples_leaf": ["int", 16, 85],
        "l2_regularization": ["float", .02, 8.0, True],
    },
}


def make_model(name, overrides=None):
    params = {**MODEL_INFO[name][1], **(overrides or {})}
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **params))
    if name == "Gaussian naive Bayes":
        return make_pipeline(StandardScaler(), GaussianNB(**params))
    if name == "Decision tree":
        return DecisionTreeClassifier(random_state=SEED, **params)
    if name == "Random forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=5, **params)
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params)


def load_cases(root, cache, split):
    ids = trip_ids(root / split)
    if not ids:
        raise ValueError(f"No trips found in {split}")
    cases = {}
    for trip_id in ids:
        with np.load(cache / split / f"{trip_id}.npz") as cached:
            if list(cached["feature_names"]) != FEATURE_NAMES:
                raise ValueError("Feature cache schema changed. Run experiments.prepare.")
            cases[trip_id] = {name: cached[name] for name in ("arcs", "features", "truth", "aligned")}
        if split == "student_test":
            cases[trip_id]["road_arcs"] = read_raw(root / split, trip_id, "arcs", 2, np.int32)
    return cases


def training_arrays(cases, ids):
    features = []
    labels = []
    for trip_id in ids:
        case = cases[trip_id]
        available = np.flatnonzero(case["truth"].any(axis=1))
        positions = np.linspace(0, len(available) - 1, min(TRAIN_CAP, len(available)), dtype=int)
        rows = available[positions]
        features.append(case["features"][rows].reshape(-1, len(FEATURE_NAMES)))
        labels.append(case["truth"][rows].ravel())
    return np.concatenate(features), np.concatenate(labels)


def split_training(cases):
    ids = np.asarray(list(cases))
    splits = GroupKFold(n_splits=FOLDS).split(ids, groups=ids)
    return [(ids[train].tolist(), ids[val].tolist()) for train, val in splits]


def fold_work(cases, splits):
    return [(training_arrays(cases, train), [cases[i] for i in val]) for train, val in splits]


def scored(model, cases):
    started = perf_counter()
    predictions = [(case, predict_scores(model, case["features"])) for case in cases]
    inference = perf_counter() - started
    return classification(predictions), inference


def fit_validate(name, overrides, work):
    records = []
    for i, (xy, validation) in enumerate(work, 1):
        started = perf_counter()
        model = make_model(name, overrides).fit(*xy)
        elapsed = perf_counter() - started
        measures, inference = scored(model, validation)
        records.append({"fold": i, **measures, "fit_s": elapsed, "inference_s": inference})
    return records


def mean_cv(folds):
    f1s = [row["macro_f1"] for row in folds]
    return {
        "cv_macro_f1": float(np.mean(f1s)), "cv_macro_f1_sd": float(np.std(f1s)),
        "cv_top1": float(np.mean([row["top1"] for row in folds])),
        "cv_log_loss": float(np.mean([row["log_loss"] for row in folds])),
        "cv_fit_s": float(sum(row["fit_s"] for row in folds)),
    }


def measure_model(name, overrides, folds, training, test, xy):
    started = perf_counter()
    model = make_model(name, overrides).fit(*xy)
    fit_time = perf_counter() - started
    fitted, _ = scored(model, list(training.values()))
    held, held_time = scored(model, list(test.values()))
    row = {
        "model": name, "family": MODEL_INFO[name][0], "objective": MODEL_INFO[name][2],
        "configuration": json.dumps({**MODEL_INFO[name][1], **(overrides or {})}, sort_keys=True),
        **mean_cv(folds), "train_macro_f1": fitted["macro_f1"], "train_top1": fitted["top1"],
        "refit_rows": len(xy[1]), "refit_s": fit_time, **held, "student_inference_s": held_time,
        "inference_ms_per_obs": 1000 * held_time / held["observations"],
    }
    return row, model


def write_rows(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def json_file(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def compare(training, test, results):
    splits = split_training(training)
    json_file(results / "cv_splits.json", [
        {"fold": i, "train": train, "validation": val} for i, (train, val) in enumerate(splits, 1)
    ])
    work = fold_work(training, splits)
    xy = training_arrays(training, list(training))
    rows, all_folds = [], []
    for name in MODEL_INFO:
        folds = fit_validate(name, None, work)
        row, _ = measure_model(name, None, folds, training, test, xy)
        print(f"{name:28} CV F1={row['cv_macro_f1']:.4f}  test top-1={row['top1']:.4f}  "
              f"F1={row['macro_f1']:.4f}  loss={row['log_loss']:.4f}", flush=True)
        all_folds.extend({"model": name, **record} for record in folds)
        rows.append(row)
    write_rows(results / "model_comparison.csv", rows)
    write_rows(results / "cv_folds.csv", all_folds)


def tune(training, test, results, artifacts, trials):
    import optuna

    with (results / "model_comparison.csv").open(newline="", encoding="utf-8") as stream:
        baseline = max(csv.DictReader(stream), key=lambda row: float(row["cv_macro_f1"]))
    name = baseline["model"]
    search = SEARCH[name]
    work = fold_work(training, split_training(training))
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    trial_folds = {}

    def objective(trial):
        overrides = {}
        for key, spec in search.items():
            kind, lower, upper, *log = spec
            if kind == "int":
                overrides[key] = trial.suggest_int(key, lower, upper)
            else:
                overrides[key] = trial.suggest_float(key, lower, upper, log=log[0])
        trial_folds[trial.number] = fit_validate(name, overrides, work)
        return mean_cv(trial_folds[trial.number])["cv_macro_f1"]

    started = perf_counter()
    study.optimize(objective, n_trials=trials)
    elapsed = perf_counter() - started
    best = study.best_trial
    xy = training_arrays(training, list(training))
    tuned, model = measure_model(name, best.params, trial_folds[best.number], training, test, xy)
    write_rows(results / "tuning_comparison.csv", [
        {"variant": "Untuned", **baseline}, {"variant": "Tuned", **tuned}
    ])
    write_rows(results / "bayesian_trials.csv", [
        {"trial": trial.number, "params": json.dumps(trial.params, sort_keys=True),
         **mean_cv(trial_folds[trial.number]),
         "seconds": trial.duration.total_seconds(),
         "best": trial.number == best.number} for trial in study.trials
    ])
    write_rows(results / "bayesian_folds.csv", [
        {"trial": number, **row} for number, rows in trial_folds.items() for row in rows
    ])
    use_tuned = best.value >= float(baseline["cv_macro_f1"])
    selected = "Tuned" if use_tuned else "Untuned"
    config = {
        "model": name, "variant": selected, "seed": SEED, "folds": FOLDS,
        "parameter_config": json.loads((tuned if use_tuned else baseline)["configuration"]),
        "candidate_count": CANDIDATE_COUNT, "feature_names": FEATURE_NAMES,
        "training_cap_observations_per_trip": TRAIN_CAP,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    if not use_tuned:
        model = make_model(name).fit(*xy)
    joblib.dump(model, artifacts / "classifier.joblib", compress=3)
    json_file(artifacts / "model.json", config)
    json_file(results / "bayesian_search.json", {
        "model": name, "sampler": "Tree-structured Parzen estimator", "startup": 5,
        "objective": "Mean grouped CV candidate macro F1", "seed": SEED, "trials": trials,
        "search_space": search, "winning_trial": best.number, "best_params": best.params,
        "best_score": best.value, "selected_variant": selected, "search_s": elapsed,
    })
    active = tuned if use_tuned else baseline
    print(f"Selected {selected.lower()} {name}: test top-1={float(active['top1']):.4f} "
          f"CV F1={float(active['cv_macro_f1']):.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["compare", "tune"])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    train = load_cases(args.data, args.cache, "train")
    student = load_cases(args.data, args.cache, "student_test")
    with threadpool_limits(limits=4):
        if args.stage == "compare":
            compare(train, student, args.results)
        else:
            tune(train, student, args.results, args.artifacts, args.trials)
