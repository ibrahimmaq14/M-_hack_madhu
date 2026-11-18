"""Audit the supplied partitions and cache supervised candidate features."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from pathsnap.data import DEFAULT_DATA, load_reference, load_trip, trip_ids
from pathsnap.features import align_reference, build_candidates, FEATURE_NAMES


def prepare(root, cache, results):
    cache.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    records, extents = [], []
    for split in ("train", "student_test"):
        partition = root / split
        for trip_id in trip_ids(partition):
            started = perf_counter()
            trip = load_trip(partition, trip_id)
            route = load_reference(partition, trip)
            feature_start = perf_counter()
            candidates = build_candidates(trip)
            feature_s = perf_counter() - feature_start
            aligned = align_reference(trip, route)
            truth = candidates.arcs == aligned[:, None]
            route_arcs = trip.graph.arcs[route]
            record = {
                "partition": split, "trip": trip_id, **trip.audit, **trip.graph.connectivity(),
                "zero_length_arcs": int(np.count_nonzero(trip.graph.lengths < 1e-6)),
                "reference_arcs": len(route),
                "reference_bytes": (partition / trip_id / f"{trip_id}.route").stat().st_size,
                "reference_breaks": int(np.count_nonzero(route_arcs[:-1, 1] != route_arcs[1:, 0])),
                "candidates_missing_reference": int(np.count_nonzero(~truth.any(axis=1))),
                "median_nearest_m": round(float(np.median(candidates.features[:, 0, 0])), 3),
                "p95_nearest_m": round(float(np.quantile(candidates.features[:, 0, 0], .95)), 3),
            }
            geo = {name: record.pop(name) for name in ("lon_min", "lon_max", "lat_min", "lat_max")}
            extents.append({"partition": split, "trip": trip_id, **geo})
            records.append(record)
            target = cache / split
            target.mkdir(exist_ok=True)
            np.savez(
                target / f"{trip_id}.npz", arcs=candidates.arcs, features=candidates.features,
                progress=candidates.progress, truth=truth, aligned=aligned, route=route,
                feature_s=feature_s, feature_names=np.asarray(FEATURE_NAMES),
            )
            print(f"{split:12} {trip_id} {len(trip.xy):6} obs  "
                  f"candidate coverage={truth.any(1).mean():.2%}  {perf_counter()-started:.2f}s", flush=True)
    if not records:
        raise ValueError("No development trips found")
    with (results / "data_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (cache / "geometry_extents.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(extents[0]))
        writer.writeheader()
        writer.writerows(extents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    prepare(args.data, args.cache, args.results)
