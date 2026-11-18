"""Geometry-based candidate search and label-free temporal features."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

CANDIDATE_COUNT = 10
SAMPLE_SPACING_M = 40
POOL_SIZE = 96
FEATURE_NAMES = [
    "distance_m", "distance_rank", "distance_above_best_m", "arc_length_m",
    "projection_fraction", "before_start_m", "after_end_m",
    "heading_before_cos", "heading_after_cos", "heading_window_cos",
    "previous_distance_m", "next_distance_m", "previous3_distance_m", "next3_distance_m",
    "previous_progress_m", "next_progress_m", "speed_before_mps", "speed_after_mps",
    "since_previous_s", "until_next_s", "turn_cos",
]


def project_segments(points, starts, directions, lengths):
    shift = points - starts
    along = np.sum(shift * directions, axis=-1)
    clipped = np.clip(along, 0, lengths)
    distance = np.linalg.norm(shift - clipped[..., None] * directions, axis=-1)
    return distance, along, clipped


def motion(xy, time, offset):
    indices = np.arange(len(xy))
    before = np.maximum(indices - offset, 0)
    after = np.minimum(indices + offset, len(xy) - 1)
    before = np.where(time - time[before] > 10, indices, before)
    after = np.where(time[after] - time > 10, indices, after)
    return before, after


def unit_motion(delta):
    length = np.linalg.norm(delta, axis=-1)
    # Treat movements smaller than GPS-level noise as weak heading evidence.
    return delta / np.maximum(length, 6)[..., None], length


def candidate_ids(trip, count):
    graph = trip.graph
    count = min(count, len(graph.arcs))
    # Sampling covers the interior of long edges and the vicinity of both endpoints.
    repeats = np.maximum(np.ceil(graph.lengths / SAMPLE_SPACING_M).astype(int) + 1, 2)
    owners = np.repeat(np.arange(len(repeats), dtype=np.int32), repeats)
    indices = np.arange(len(owners)) - np.repeat(np.cumsum(repeats) - repeats, repeats)
    fractions = indices / (repeats[owners] - 1)
    samples = graph.starts[owners] + graph.vectors[owners] * fractions[:, None]
    tree = cKDTree(samples)

    def rank(points, pool_size):
        sample_ids = tree.query(points, k=min(pool_size, len(samples)), workers=2)[1]
        pool = np.sort(owners[sample_ids], axis=1)
        dist = project_segments(
            points[:, None], graph.starts[pool], graph.directions[pool], graph.lengths[pool]
        )[0]
        dist[:, 1:] = np.where(pool[:, 1:] == pool[:, :-1], np.inf, dist[:, 1:])
        order = np.argsort(dist, axis=1, kind="stable")[:, :count]
        return np.take_along_axis(pool, order, 1), np.isfinite(np.take_along_axis(dist, order, 1)).all(1)

    output = np.empty((len(trip.xy), count), dtype=np.int32)
    for start in range(0, len(trip.xy), 512):
        points = trip.xy[start:start + 512]
        arcs, complete = rank(points, POOL_SIZE)
        for i in np.flatnonzero(~complete):
            # A very long or isolated arc can occupy the nearest index samples.
            # Exact segment distances guarantee a complete set in this uncommon case.
            distances = project_segments(points[i], graph.starts, graph.directions, graph.lengths)[0]
            arcs[i] = np.argsort(distances, kind="stable")[:count]
        output[start:start + len(points)] = arcs
    return output


@dataclass
class Candidates:
    arcs: np.ndarray
    features: np.ndarray
    progress: np.ndarray


def build_candidates(trip, count=CANDIDATE_COUNT):
    arcs = candidate_ids(trip, count)
    graph = trip.graph
    xy, times = trip.xy, trip.track[:, 2]
    starts, directions, lengths = graph.starts[arcs], graph.directions[arcs], graph.lengths[arcs]
    dist, along, progress = project_segments(xy[:, None], starts, directions, lengths)
    before, after = motion(xy, times, 1)
    before3, after3 = motion(xy, times, 3)
    prev_dist, _, prev_progress = project_segments(xy[before, None], starts, directions, lengths)
    next_dist, _, next_progress = project_segments(xy[after, None], starts, directions, lengths)
    prev3_dist = project_segments(xy[before3, None], starts, directions, lengths)[0]
    next3_dist = project_segments(xy[after3, None], starts, directions, lengths)[0]
    head_before, step_before = unit_motion(xy - xy[before])
    head_after, step_after = unit_motion(xy[after] - xy)
    head_window, _ = unit_motion(xy[after3] - xy[before3])
    elapsed_before, elapsed_after = times - times[before], times[after] - times
    features = [
        dist, np.broadcast_to(np.arange(arcs.shape[1]), arcs.shape), dist - dist[:, :1],
        lengths, progress / np.maximum(lengths, 1e-6),
        np.maximum(-along, 0), np.maximum(along - lengths, 0),
        np.sum(head_before[:, None] * directions, axis=-1),
        np.sum(head_after[:, None] * directions, axis=-1),
        np.sum(head_window[:, None] * directions, axis=-1),
        prev_dist, next_dist, prev3_dist, next3_dist,
        progress - prev_progress, next_progress - progress,
        step_before / np.maximum(elapsed_before, 1), step_after / np.maximum(elapsed_after, 1),
        elapsed_before, elapsed_after, np.sum(head_before * head_after, axis=-1),
    ]
    features = np.stack(
        [np.broadcast_to(f[:, None], arcs.shape) if f.ndim == 1 else f for f in features], axis=-1
    ).astype(np.float32)
    return Candidates(arcs, features, progress)


def align_reference(trip, route):
    """Assign weak point labels by minimum-cost monotonic alignment to the reference route."""
    graph = trip.graph
    starts, directions, lengths = graph.starts[route], graph.directions[route], graph.lengths[route]
    xy, times = trip.xy, trip.track[:, 2]
    before, after = motion(xy, times, 2)
    heading, _ = unit_motion(xy[after] - xy[before])
    route_index = np.arange(len(route), dtype=np.int32)
    # A small forward-skip penalty disambiguates repeated nearby route sections.
    advance_cost = 0.05 * route_index + 0.001 * (np.cumsum(lengths) - lengths)
    back = np.empty((len(xy), len(route)), dtype=np.int32)
    costs = np.zeros(len(route))
    for start in range(0, len(xy), 256):
        points = xy[start:start + 256]
        dist = project_segments(points[:, None], starts, directions, lengths)[0]
        observations = np.log1p(np.square(dist / 12)) - 0.55 * (heading[start:start+len(points)] @ directions.T)
        for step, obs in enumerate(observations, start):
            adjusted = costs - advance_cost
            minimum = np.minimum.accumulate(adjusted)
            best = np.maximum.accumulate(np.where(adjusted == minimum, route_index, 0))
            back[step] = best
            costs = minimum + advance_cost + obs
    chosen = np.empty(len(xy), dtype=np.int32)
    index = int(np.argmin(costs))
    for step in range(len(xy) - 1, -1, -1):
        chosen[step] = route[index]
        index = back[step, index]
    return chosen
