"""Reference-based metrics, intentionally separate from prediction."""

from rapidfuzz.distance import Levenshtein
import numpy as np

from .metrics import compress, continuity, safe_ratio


def route_metrics(graph, aligned, reference, prediction, parts):
    # Preserve trip order and explicit segment breaks. Consecutive equal IDs are one traversal.
    sequence = np.concatenate([compress(np.asarray(part, dtype=np.int32)) for part in parts])
    reference = compress(reference)
    raw_connections, raw_changes = continuity(graph.arcs, prediction)
    connections = 0
    for part in parts:
        n, _ = continuity(graph.arcs, np.asarray(part, dtype=np.int32))
        connections += n
    steps = max(0, len(sequence) - 1)
    return {
        "points": len(aligned), "correct_points": int(np.count_nonzero(prediction == aligned)),
        "point_accuracy": float(np.mean(prediction == aligned)),
        "raw_connections": raw_connections, "raw_changes": raw_changes,
        "raw_continuity": safe_ratio(raw_connections, raw_changes, default=1),
        "connections": connections, "sequence_steps": steps,
        "route_continuity": safe_ratio(connections, steps, default=1),
        "sequence_arcs": len(sequence), "reference_arcs": len(reference),
        "sequence_similarity": Levenshtein.normalized_similarity(sequence.tolist(), reference.tolist()),
        "breaks": len(parts) - 1,
    }
