"""Inference and API-facing geometry. Reference routes are outside this module."""

import json
from pathlib import Path

import joblib
import numpy as np

from .data import load_trip
from .decoder import DecoderSettings, TransitionLattice, decode
from .features import CANDIDATE_COUNT, FEATURE_NAMES, build_candidates
from .metrics import predict_scores


class Matcher:
    def __init__(self, artifact_dir):
        artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((artifact_dir / "model.json").read_text(encoding="utf-8"))
        if (self.metadata["feature_names"] != FEATURE_NAMES
                or self.metadata["candidate_count"] != CANDIDATE_COUNT):
            raise ValueError("Model artifact and candidate feature schema disagree")
        config = json.loads((artifact_dir / "decoder.json").read_text(encoding="utf-8"))
        self.settings = DecoderSettings(**config)
        self.model = joblib.load(artifact_dir / "classifier.joblib")

    def match(self, root, trip_id):
        trip = load_trip(root, trip_id)
        candidates = build_candidates(trip, self.metadata["candidate_count"])
        scores = predict_scores(self.model, candidates.features)
        lattice = TransitionLattice(trip, candidates, self.settings.gap_reset_s)
        decoded = decode(lattice, scores, self.settings)
        return trip, candidates, decoded


def visible_roads(trip, candidates=None, limit=8000):
    """Sample contextual roads across the viewport, always keeping candidate roads."""
    graph = trip.graph
    midpoint = graph.starts + graph.vectors / 2
    a = trip.xy.min(axis=0) - 150
    b = trip.xy.max(axis=0) + 150
    relevant = np.flatnonzero(((midpoint >= a) & (midpoint <= b)).all(axis=1))
    if len(relevant) > limit:
        relevant = relevant[np.linspace(0, len(relevant) - 1, limit, dtype=int)]
    if candidates is not None:
        relevant = np.union1d(relevant, candidates.arcs)
    arcs = graph.arcs[relevant]
    physical = np.unique(np.sort(arcs, axis=1), axis=0)
    return graph.lonlat[physical].round(6).tolist()


def public_prediction(trip, candidates, matched):
    graph = trip.graph
    chunks = []
    for part in matched.route_parts:
        nodes = graph.arcs[part]
        chunks.append(graph.lonlat[np.r_[nodes[0, 0], nodes[:, 1]]].round(6).tolist())
    return {
        "trip_id": trip.trip_id,
        "observations": len(trip.xy),
        "predicted_arcs": matched.arcs.tolist(),
        "route_parts": matched.route_parts,
        "geometry": {
            "gps": trip.track[:, :2].round(6).tolist(),
            "route": chunks,
            "roads": visible_roads(trip, candidates),
        },
        "decoder": {
            "parts": len(chunks),
            "matched_distance_km": round(float(sum(graph.lengths[part].sum()
                                                     for part in matched.route_parts)) / 1000, 3),
        },
    }
