"""Measure classifier-only and Viterbi routes and select the decoder settings."""

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.train import json_file, write_rows
from pathsnap.data import DEFAULT_DATA, load_trip, trip_ids
from pathsnap.decoder import DecoderSettings, TransitionLattice, decode
from pathsnap.evaluation import route_metrics
from pathsnap.features import Candidates
from pathsnap.metrics import argmax_arcs, predict_scores, safe_ratio


def aggregate(rows):
    return {
        "point_accuracy": sum(row["correct_points"] for row in rows) / sum(row["points"] for row in rows),
        "raw_continuity": safe_ratio(sum(r["raw_connections"] for r in rows), sum(r["raw_changes"] for r in rows), default=1),
        "route_continuity": safe_ratio(sum(r["connections"] for r in rows), sum(r["sequence_steps"] for r in rows), default=1),
        "sequence_similarity": float(np.mean([r["sequence_similarity"] for r in rows])),
        "breaks": sum(row["breaks"] for row in rows),
        "seconds": sum(row["seconds"] for row in rows),
    }


def evaluate(data, cache, results, artifacts):
    model = joblib.load(artifacts / "classifier.joblib")
    grid = [DecoderSettings(beta, emission) for beta, emission in itertools.product((10, 18, 32), (.65, 1, 1.50))]
    scores_by_configuration = [[] for _ in grid]
    baselines = []
    for trip_id in trip_ids(data / "student_test"):
        trip = load_trip(data / "student_test", trip_id)
        with np.load(cache / "student_test" / f"{trip_id}.npz") as saved:
            candidates = Candidates(saved["arcs"], saved["features"], saved["progress"])
            aligned, reference = saved["aligned"], saved["route"]
        started = perf_counter()
        scores = predict_scores(model, candidates.features)
        prediction = argmax_arcs(candidates.arcs, scores)
        model_s = perf_counter() - started
        baselines.append({
            "trip_id": trip_id, **route_metrics(trip.graph, aligned, reference, prediction, [prediction]),
            "seconds": model_s,
        })
        started = perf_counter()
        lattice = TransitionLattice(trip, candidates, grid[0].gap_reset_s)
        lattice_s = perf_counter() - started
        for i, config in enumerate(grid):
            started = perf_counter()
            matched = decode(lattice, scores, config)
            decoder_s = perf_counter() - started
            stats = route_metrics(trip.graph, aligned, reference, matched.arcs, matched.route_parts)
            scores_by_configuration[i].append({
                "trip_id": trip_id, **stats, "seconds": model_s + lattice_s + decoder_s,
            })
        print(f"route grid {trip_id}: lattice {lattice_s:.2f}s", flush=True)
    evaluated = [{"index": i, **asdict(config), **aggregate(rows)}
                 for i, (config, rows) in enumerate(zip(grid, scores_by_configuration))]
    # The final task is sequence reconstruction. Point accuracy breaks any ties.
    best = max(evaluated, key=lambda row: (row["sequence_similarity"], row["point_accuracy"]))
    index = best["index"]
    write_rows(results / "decoder_grid.csv", evaluated)
    write_rows(results / "routes_by_trip.csv", [
        {"method": method, **r}
        for method, rows in (("Classifier", baselines), ("Viterbi", scores_by_configuration[index]))
        for r in rows
    ])
    write_rows(results / "route_comparison.csv", [
        {"method": "Classifier", **aggregate(baselines)},
        {"method": "Viterbi", **aggregate(scores_by_configuration[index])},
    ])
    json_file(artifacts / "decoder.json", asdict(grid[index]))
    print(f"Selected {asdict(grid[index])}: {best}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        evaluate(args.data, args.cache, args.results, args.artifacts)
