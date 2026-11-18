"""Log-space Viterbi with sparse, directed shortest-path transitions."""

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush

import numpy as np


@dataclass(frozen=True)
class DecoderSettings:
    beta_m: float = 18.0
    emission_weight: float = 1.0
    gap_reset_s: float = 60.0
    reversal_cost: float = 3.0


class RoadRouter:
    def __init__(self, graph):
        self.graph = graph
        self.neighbourhood = lru_cache(maxsize=1200)(self._neighbourhood)

    def _neighbourhood(self, source, limit):
        """Return directed distances and predecessor arcs up to the distance bound."""
        graph = self.graph
        distances, predecessors = {source: 0.0}, {}
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            for arc in graph.outgoing(node):
                other = int(graph.arcs[arc, 1])
                next_distance = distance + float(graph.lengths[arc])
                if next_distance <= limit and next_distance < distances.get(other, np.inf):
                    distances[other] = next_distance
                    predecessors[other] = int(arc)
                    heappush(queue, (next_distance, other))
        return distances, predecessors

    def bridge(self, source, destination, bound):
        distances, predecessors = self.neighbourhood(source, bound)
        if destination not in distances:
            return None
        route = []
        while destination != source:
            arc = predecessors[destination]
            route.append(arc)
            destination = int(self.graph.arcs[arc, 0])
        return route[::-1]


class TransitionLattice:
    def __init__(self, trip, candidates, gap_reset_s):
        self.graph = trip.graph
        self.ids = candidates.arcs
        self.router = RoadRouter(self.graph)
        self.elapsed = np.diff(trip.track[:, 2])
        self.steps = np.linalg.norm(np.diff(trip.xy, axis=0), axis=1)
        self.bounds = np.clip(self.steps * 3 + 45 * np.minimum(self.elapsed, 25), 120, 4000)
        buckets = np.array([160, 320, 640, 1280, 2560, 4096])
        self.buckets = buckets[np.searchsorted(buckets, self.bounds)]
        self.gap_reset_s = gap_reset_s
        count = self.ids.shape[1]
        shape = (len(self.steps), count, count)
        self.penalties = np.full(shape, np.inf, dtype=np.float32)
        self.reversals = np.zeros(shape, dtype=bool)
        self.retro_m = np.zeros(shape, dtype=np.float32)
        for i, (previous, current) in enumerate(zip(self.ids[:-1], self.ids[1:])):
            if self.elapsed[i] > gap_reset_s:
                continue
            starts = self.graph.arcs[current, 0]
            prior_length = self.graph.lengths[previous] - candidates.progress[i]
            current_progress = candidates.progress[i + 1]
            ends = self.graph.arcs[previous, 1]
            bounded = int(self.buckets[i])
            network = np.empty((count, count))
            known = {}
            for row, end in enumerate(ends):
                end = int(end)
                if end not in known:
                    known[end] = self.router.neighbourhood(end, bounded)[0]
                routes = known[end]
                network[row] = [routes.get(int(start), np.inf) for start in starts]
            length = prior_length[:, None] + network + current_progress[None]
            same = previous[:, None] == current
            difference = current_progress[None] - candidates.progress[i, :, None]
            length = np.where(same, np.abs(difference), length)
            self.penalties[i] = np.where(length <= self.bounds[i], abs(length - self.steps[i]), np.inf)
            self.retro_m[i] = np.where(same, np.maximum(-difference - 8, 0), 0)
            self.reversals[i] = (~same & (ends[:, None] == starts[None])
                                 & (self.graph.arcs[previous, 0][:, None] == self.graph.arcs[current, 1]))

    def route_parts(self, predicted, breaks):
        parts = [[int(predicted[0])]]
        for i, (previous, current) in enumerate(zip(predicted[:-1], predicted[1:])):
            previous, current = int(previous), int(current)
            if breaks[i + 1]:
                parts.append([current])
            elif previous != current:
                source = int(self.graph.arcs[previous, 1])
                destination = int(self.graph.arcs[current, 0])
                bridge = self.router.bridge(source, destination, int(self.buckets[i]))
                if bridge is None:
                    parts.append([current])
                else:
                    parts[-1].extend([*bridge, current])
        return parts


@dataclass
class Decoded:
    arcs: np.ndarray
    breaks: np.ndarray
    route_parts: list


def decode(lattice, probabilities, settings, *, reconstruct=True):
    if settings.gap_reset_s != lattice.gap_reset_s:
        raise ValueError("Decoder and transition lattice gap limits disagree")
    emission = np.log(np.clip(probabilities, 1e-7, 1.0)) * settings.emission_weight
    scale = settings.beta_m + 0.22 * lattice.steps + np.minimum(40, 3 * np.sqrt(lattice.elapsed))
    transition = -(lattice.penalties + lattice.retro_m) / scale[:, None, None]
    transition -= settings.reversal_cost * lattice.reversals
    steps, count = emission.shape
    previous = emission[0]
    back = np.empty((steps, count), dtype=np.int32)
    breaks = np.zeros(steps, dtype=bool)
    segment_ends = []
    cols = np.arange(count)
    for i in range(1, steps):
        candidate_paths = previous[:, None] + transition[i - 1]
        parents = np.argmax(candidate_paths, axis=0)
        current = emission[i] + candidate_paths[parents, cols]
        if np.isfinite(current).any():
            previous = current - current.max()
            back[i] = parents
        else:
            segment_ends.append((i - 1, int(previous.argmax())))
            previous = emission[i]
            breaks[i] = True
    segment_ends.append((steps - 1, int(previous.argmax())))
    result = np.empty(steps, dtype=np.int32)
    start = 0
    for end, state in segment_ends:
        for i in range(end, start - 1, -1):
            result[i] = lattice.ids[i, state]
            if i > start:
                state = back[i, state]
        start = end + 1
    parts = lattice.route_parts(result, breaks) if reconstruct else []
    return Decoded(result, breaks, parts)
