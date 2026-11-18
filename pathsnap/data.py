"""Raw-data parsing, projected graph geometry, and permitted-split auditing."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

METRES_PER_DEGREE = 111_195.08
TRIP_ID = re.compile(r"\d{8}", re.ASCII)
DEFAULT_DATA = Path("../Student folder/Student folder")


def trip_ids(root):
    return sorted(
        path.name for path in Path(root).iterdir()
        if path.is_dir() and TRIP_ID.fullmatch(path.name)
        and all((path / f"{path.name}.{suffix}").is_file() for suffix in ("track", "nodes", "arcs"))
    )


def read_raw(root, trip_id, suffix, columns, dtype):
    if not TRIP_ID.fullmatch(trip_id):
        raise ValueError("Invalid trip ID")
    path = Path(root) / trip_id / f"{trip_id}.{suffix}"
    rows = np.loadtxt(path, dtype=dtype, ndmin=2)
    if not len(rows) or rows.shape[1] != columns:
        raise ValueError(f"{path.name} must contain {columns} columns and at least one row")
    return rows


def valid_coords(points):
    return np.isfinite(points).all(axis=1) & (np.abs(points) <= [180, 90]).all(axis=1)


class RoadNetwork:
    def __init__(self, lonlat, arcs, origin):
        if not valid_coords(lonlat).all():
            raise ValueError("Network contains invalid coordinates")
        if arcs.min() < 0 or arcs.max() >= len(lonlat):
            raise ValueError("Network contains an invalid node ID")
        self.lonlat = lonlat
        self.arcs = arcs
        self.origin = origin
        self.scale = METRES_PER_DEGREE * np.array([np.cos(np.deg2rad(origin[1])), 1])
        self.xy = self.project(lonlat)
        self.starts = self.xy[arcs[:, 0]]
        self.vectors = self.xy[arcs[:, 1]] - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        self.directions = self.vectors / np.maximum(self.lengths[:, None], 1e-6)
        self.order = np.argsort(arcs[:, 0], kind="stable")
        self.offsets = np.r_[0, np.cumsum(np.bincount(arcs[:, 0], minlength=len(lonlat)))]

    def project(self, lonlat):
        offsets = np.asarray(lonlat, dtype=float) - self.origin
        offsets[..., 0] = (offsets[..., 0] + 180) % 360 - 180
        return offsets * self.scale

    def outgoing(self, node):
        return self.order[self.offsets[node]:self.offsets[node + 1]]

    def connectivity(self):
        rows, cols = self.arcs.T
        edges = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(self.xy),) * 2)
        weak, component = connected_components(edges, directed=True, connection="weak")
        strong = connected_components(edges, directed=True, connection="strong", return_labels=False)
        return {
            "weak_components": int(weak),
            "strong_components": int(strong),
            "largest_weak_component": int(np.bincount(component).max()),
            "isolated_nodes": int(len(self.xy) - len(np.unique(self.arcs))),
        }


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray
    xy: np.ndarray
    graph: RoadNetwork
    audit: dict


def load_trip(root, trip_id):
    """Load inference inputs exclusively. A .route file is never opened here."""
    raw = read_raw(root, trip_id, "track", 3, float)
    nodes = read_raw(root, trip_id, "nodes", 2, float)
    arcs = read_raw(root, trip_id, "arcs", 2, np.int32)
    valid = valid_coords(raw[:, :2]) & np.isfinite(raw[:, 2])
    if not np.any(valid):
        raise ValueError("No valid GPS observations")
    track = raw[valid]
    reversals = int(np.count_nonzero(np.diff(track[:, 2]) < 0))
    track = track[np.argsort(track[:, 2], kind="stable")]
    keep = np.r_[True, np.diff(track[:, 2]) > 1e-6]
    track = track[keep]
    graph = RoadNetwork(nodes, arcs, np.median(track[:, :2], axis=0))
    gaps = np.diff(track[:, 2])
    sizes = {f"{ext}_bytes": (Path(root) / trip_id / f"{trip_id}.{ext}").stat().st_size
             for ext in ("track", "nodes", "arcs")}
    audit = {
        "raw_observations": len(raw), "observations": len(track), "nodes": len(nodes), "arcs": len(arcs),
        "nonfinite_track_rows": int(np.count_nonzero(~np.isfinite(raw).all(axis=1))),
        "invalid_track_rows": int(np.count_nonzero(~valid)),
        "reversed_timestamps": reversals,
        "duplicate_timestamps": int(np.count_nonzero(~keep)),
        "one_second_steps": int(np.count_nonzero(np.isclose(gaps, 1))),
        "gaps_over_10s": int(np.count_nonzero(gaps > 10)),
        "max_gap_s": round(float(gaps.max(initial=0)), 6),
        "lon_min": float(track[:, 0].min()), "lon_max": float(track[:, 0].max()),
        "lat_min": float(track[:, 1].min()), "lat_max": float(track[:, 1].max()),
        **sizes,
    }
    return Trip(trip_id, track, graph.project(track[:, :2]), graph, audit)


def load_reference(root, trip):
    """Used by offline evaluation and the server's scoring stage after prediction."""
    ref = read_raw(root, trip.trip_id, "route", 1, np.int32).ravel()
    if ref.min() < 0 or ref.max() >= len(trip.graph.arcs):
        raise ValueError("Reference contains an invalid arc ID")
    return ref
