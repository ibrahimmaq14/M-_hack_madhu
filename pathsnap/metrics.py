"""Candidate metrics, directed continuity and observation-level scoring."""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support, confusion_matrix


def predict_scores(model, features):
    return model.predict_proba(features.reshape(-1, features.shape[-1]))[:, 1].reshape(features.shape[:2])


def argmax_arcs(arcs, scores):
    return np.take_along_axis(arcs, scores.argmax(axis=1, keepdims=True), 1).ravel()


def compress(arcs):
    return arcs[np.r_[True, arcs[1:] != arcs[:-1]]] if len(arcs) else arcs


def continuity(arc_pairs, sequence):
    a = arc_pairs[compress(sequence)]
    return int(np.count_nonzero(a[:-1, 1] == a[1:, 0])), max(0, len(a) - 1)


def safe_ratio(a, b, *, default=0):
    return a / b if b else default


def classification(results):
    probabilities = np.concatenate([r[1].ravel() for r in results])
    truth = np.concatenate([r[0]["truth"].ravel() for r in results])
    binary = probabilities >= .5
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, binary, average="macro", zero_division=0
    )
    confusion = confusion_matrix(truth, binary, labels=[False, True])
    obs = hits = misses = matches = changes = 0
    for case, scores in results:
        winner = argmax_arcs(case["arcs"], scores)
        obs += len(winner)
        hits += np.count_nonzero(winner == case["aligned"])
        misses += np.count_nonzero(~case["truth"].any(axis=1))
        if "road_arcs" in case:
            connected, total = continuity(case["road_arcs"], winner)
            matches += connected
            changes += total
    return {
        "candidate_accuracy": float(accuracy_score(truth, binary)),
        "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
        "log_loss": float(log_loss(truth, probabilities, labels=[False, True])),
        "top1": safe_ratio(hits, obs), "candidate_miss": safe_ratio(misses, obs),
        "continuity": safe_ratio(matches, changes, default=1) if "road_arcs" in results[0][0] else None,
        "observations": obs, "candidate_rows": len(truth), "route_changes": changes,
        "tn": int(confusion[0, 0]), "fp": int(confusion[0, 1]),
        "fn": int(confusion[1, 0]), "tp": int(confusion[1, 1]),
    }
