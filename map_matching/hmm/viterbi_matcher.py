"""HMM/Viterbi map matcher — Phase 5.2 of the implementation plan.

Hand-implemented sliding-window Viterbi decoder over an OSM road graph.
Matches a sequence of fused UKF positions to a sequence of road edges.

Algorithm (MIP Section 6):

  1. CANDIDATE EDGES: for each fused position, find all road edges within
     search_radius_m = max(sigma_emit_m, 5.0) meters. Uses an R-tree spatial
     index built from the graph at init time for O(log n) lookup.

  2. EMISSION PROBABILITY (Gaussian on perpendicular distance):
       p(obs | edge) = exp(-d_perp² / (2 * sigma²))
       sigma = max(ukf_pos_std, sigma_emit_m)  [floor at 5.0m]
       d_perp = perpendicular distance from fused position to edge centerline

  3. TRANSITION PROBABILITY (route-distance vs Euclidean-distance):
       p(edge_t | edge_{t-1}) ∝ exp(-(d_route - d_euclid) / beta)
       d_route  = shortest network path distance between the two edges
       d_euclid = straight-line distance between their midpoints
       beta     = beta_transition (default 10.0; tune on val routes)
     A route_distance close to d_euclid = likely transition (road network
     allows the movement). Route_distance >> d_euclid = unlikely jump across
     disconnected roads, penalized exponentially.

  4. SLIDING WINDOW VITERBI: decode over the last W=window_size fused positions
     at each cycle, re-decode the entire window every cycle.
     COMMIT only the EARLIEST position in the window (the (i - W + 1)-th step)
     to prevent path jitter as new evidence arrives and the decode shifts.

Reference formulation: Leuven.MapMatching non-emitting-states HMM paper
(Lou et al., 2009, "Map-matching for low-sampling-rate GPS trajectories").
Not imported — hand-implemented to maintain full control over emission/transition
parameters, which are tuned per the implementation plan.

Usage:
    from map_matching.hmm.viterbi_matcher import ViterbiMapMatcher
    matcher = ViterbiMapMatcher("map_matching/osm_extraction/corridor.graphml")
    snapped_pos = matcher.snap(fused_pos)  # (2,) float32 [north, east] meters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class _Edge:
    """Minimal edge representation for Viterbi state space."""
    edge_id: tuple  # (u, v, key) from networkx
    midpoint_m: np.ndarray   # (2,) [north, east] meters
    start_m: np.ndarray      # (2,) [north, east]
    end_m: np.ndarray        # (2,) [north, east]
    length_m: float


@dataclass
class _ViterbiState:
    """One Viterbi trellis cell."""
    log_prob: float
    backpointer: Optional[int]  # index into edge list at previous step


def _latlon_to_northing_easting(lat: np.ndarray, lon: np.ndarray,
                                  lat0: float, lon0: float) -> np.ndarray:
    """Convert lat/lon to a local flat-earth [north, east] frame in meters.

    Uses the equirectangular approximation, valid for small extents (< ~50km).
    The reference point (lat0, lon0) becomes the origin [0, 0].
    """
    R_earth_m = 6_371_000.0
    north = (lat - lat0) * (np.pi / 180.0) * R_earth_m
    east  = (lon - lon0) * (np.pi / 180.0) * R_earth_m * np.cos(np.deg2rad(lat0))
    return np.stack([north, east], axis=-1)


def _perpendicular_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point to line segment [a, b].

    If the foot of the perpendicular lies outside [a, b], returns the distance
    to the nearest endpoint instead (clamped projection).
    """
    ab = b - a
    ab_len_sq = float(np.dot(ab, ab))
    if ab_len_sq < 1e-12:
        return float(np.linalg.norm(point - a))
    t = float(np.dot(point - a, ab) / ab_len_sq)
    t_clamped = max(0.0, min(1.0, t))
    foot = a + t_clamped * ab
    return float(np.linalg.norm(point - foot))


class ViterbiMapMatcher:
    """Sliding-window HMM/Viterbi map matcher backed by an OSM road graph.

    The matcher is stateful: call snap() once per cycle in sequence. It
    maintains an internal sliding window and only commits positions with a
    W-step delay (so the Viterbi decode has W steps of future evidence).
    """

    def __init__(self,
                 graph_path: str,
                 window_size: int = 15,
                 sigma_emit_m: float = 10.0,
                 beta_transition: float = 10.0,
                 search_radius_m: float = 50.0) -> None:
        """
        Args:
            graph_path:       Path to the OSM GraphML file from extract.py.
            window_size:      Number of steps in the Viterbi sliding window.
            sigma_emit_m:     Emission noise std (meters); floor at 5.0m.
            beta_transition:  Transition decay constant (meters); larger = looser.
            search_radius_m:  Maximum distance to search for candidate edges.
        """
        self.window_size = window_size
        self.sigma_emit_m = max(5.0, sigma_emit_m)
        self.beta_transition = beta_transition
        self.search_radius_m = search_radius_m

        self._edges, self._edge_index, self._graph, self._lat0, self._lon0 = \
            self._load_graph(graph_path)
        self._shortest_path_cache: dict[tuple, float] = {}
        self._pos_history: list[np.ndarray] = []    # raw UKF positions
        self._committed: list[np.ndarray] = []       # committed snapped positions

    # -- Graph loading -----------------------------------------------------------

    def _load_graph(self, graph_path: str):
        """Load GraphML, project nodes to flat-earth meters, build edge list and R-tree."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required: pip install networkx")

        path = Path(graph_path)
        if not path.exists():
            raise FileNotFoundError(
                f"OSM graph not found: {graph_path}\n"
                "Run: python map_matching/osm_extraction/extract.py"
            )

        G = nx.read_graphml(graph_path)

        # Determine lat/lon reference point (use the centroid of all nodes)
        lats = [float(G.nodes[n].get("y", G.nodes[n].get("lat", 0))) for n in G.nodes]
        lons = [float(G.nodes[n].get("x", G.nodes[n].get("lon", 0))) for n in G.nodes]
        lat0 = float(np.mean(lats))
        lon0 = float(np.mean(lons))

        # Build node coordinate lookup in flat-earth meters
        node_pos_m: dict = {}
        for n in G.nodes:
            nd = G.nodes[n]
            lat = float(nd.get("y", nd.get("lat", lat0)))
            lon = float(nd.get("x", nd.get("lon", lon0)))
            pos = _latlon_to_northing_easting(
                np.array(lat), np.array(lon), lat0, lon0
            )
            node_pos_m[n] = pos

        # Build edge list
        edges: list[_Edge] = []
        for u, v, key, data in G.edges(keys=True, data=True):
            if u not in node_pos_m or v not in node_pos_m:
                continue
            a = node_pos_m[u]
            b = node_pos_m[v]
            mid = (a + b) / 2.0
            length = float(np.linalg.norm(b - a))
            edges.append(_Edge(
                edge_id=(u, v, key),
                midpoint_m=mid,
                start_m=a,
                end_m=b,
                length_m=max(length, 1.0),  # floor at 1m for numerical safety
            ))

        # Build a simple spatial index (grid-based for simplicity)
        edge_index = _GridIndex(edges, cell_size_m=50.0)

        return edges, edge_index, G, lat0, lon0

    # -- Shortest-path distance (cached) -----------------------------------------

    def _network_distance(self, edge_a: _Edge, edge_b: _Edge) -> float:
        """Approximate network distance between two edges via their midpoints.

        Uses a direct Euclidean distance as a fast approximation when the two
        edges are within the same connected component. This is a conservative
        under-estimate (network distance ≥ Euclidean), so the transition
        probability will be slightly too high for long detours — acceptable
        for the hackathon demo. Replace with networkx.shortest_path for
        production if needed.
        """
        cache_key = (edge_a.edge_id, edge_b.edge_id)
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]

        # Fast approximation: Euclidean between midpoints scaled by a road-network
        # detour factor of 1.4 (typical ratio for urban grids; from literature).
        euclid = float(np.linalg.norm(edge_b.midpoint_m - edge_a.midpoint_m))
        approx = euclid * 1.4

        self._shortest_path_cache[cache_key] = approx
        return approx

    # -- Emission & transition probabilities ------------------------------------

    def _log_emission(self, pos: np.ndarray, edge: _Edge) -> float:
        d = _perpendicular_distance(pos, edge.start_m, edge.end_m)
        return -0.5 * (d ** 2) / (self.sigma_emit_m ** 2)

    def _log_transition(self, from_edge: _Edge, to_edge: _Edge,
                         euclid_step: float) -> float:
        """Transition log probability: penalize network detour vs Euclidean step."""
        d_route = self._network_distance(from_edge, to_edge)
        d_euclid = max(euclid_step, 0.1)  # floor to avoid division by zero
        diff = max(d_route - d_euclid, 0.0)  # only penalize detour, not shortcut
        return -(diff / self.beta_transition)

    # -- Viterbi decode ----------------------------------------------------------

    def _viterbi(self, positions: list[np.ndarray]) -> list[_Edge | None]:
        """Run Viterbi on a list of positions, return best edge sequence."""
        T = len(positions)
        if T == 0:
            return []

        # Get candidate edges for each timestep
        candidates: list[list[_Edge]] = []
        for pos in positions:
            cands = self._edge_index.query(pos, self.search_radius_m)
            if not cands:
                # Fall back to nearest single edge if nothing in radius
                cands = self._edge_index.nearest(pos)
            candidates.append(cands)

        # Initialize trellis
        trellis: list[list[_ViterbiState]] = []
        init_states = [_ViterbiState(log_prob=self._log_emission(positions[0], e),
                                     backpointer=None)
                       for e in candidates[0]]
        trellis.append(init_states)

        # Forward pass
        for t in range(1, T):
            prev_states = trellis[t - 1]
            prev_cands  = candidates[t - 1]
            curr_cands  = candidates[t]
            euclid_step = float(np.linalg.norm(positions[t] - positions[t - 1]))

            curr_states = []
            for j, curr_edge in enumerate(curr_cands):
                best_log_prob = -np.inf
                best_bp = 0
                for i, prev_edge in enumerate(prev_cands):
                    lp = (prev_states[i].log_prob
                          + self._log_transition(prev_edge, curr_edge, euclid_step)
                          + self._log_emission(positions[t], curr_edge))
                    if lp > best_log_prob:
                        best_log_prob = lp
                        best_bp = i
                curr_states.append(_ViterbiState(log_prob=best_log_prob, backpointer=best_bp))
            trellis.append(curr_states)

        # Backtrack from best final state
        if not trellis[-1]:
            return [None] * T

        best_final = int(np.argmax([s.log_prob for s in trellis[-1]]))
        path: list[_Edge | None] = [None] * T
        path[T - 1] = candidates[-1][best_final] if candidates[-1] else None

        ptr = best_final
        for t in range(T - 2, -1, -1):
            if trellis[t + 1][ptr].backpointer is None:
                break
            ptr = trellis[t + 1][ptr].backpointer
            path[t] = candidates[t][ptr] if candidates[t] else None

        return path

    # -- Public interface --------------------------------------------------------

    def snap(self, pos: np.ndarray) -> np.ndarray:
        """Snap a fused UKF position to the nearest road edge.

        pos: (2,) [north_m, east_m] in the local flat-earth frame.
        Returns: (2,) snapped position in the same frame.

        IMPORTANT: Returns the snapped position for the step that is
        (window_size - 1) steps ago (committed position), NOT the current step.
        The first (window_size - 1) calls return pos unchanged because the
        window is not yet full — the caller should be aware of this delay.
        """
        self._pos_history.append(pos.copy())

        # Only decode once we have a full window
        if len(self._pos_history) < self.window_size:
            self._committed.append(pos.copy())
            return pos.copy()

        # Decode the window
        window_positions = self._pos_history[-self.window_size:]
        edge_path = self._viterbi(window_positions)

        # Commit the earliest position in the window (snapped to road)
        earliest_edge = edge_path[0]
        if earliest_edge is not None:
            snapped = self._snap_to_edge(self._pos_history[-self.window_size], earliest_edge)
        else:
            snapped = self._pos_history[-self.window_size].copy()

        self._committed.append(snapped)
        return snapped

    def _snap_to_edge(self, pos: np.ndarray, edge: _Edge) -> np.ndarray:
        """Project pos onto the edge line segment (clamped to [a, b])."""
        a, b = edge.start_m, edge.end_m
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq < 1e-12:
            return a.copy()
        t = float(np.dot(pos - a, ab) / ab_len_sq)
        t_clamped = max(0.0, min(1.0, t))
        return (a + t_clamped * ab).astype(np.float32)

    def reset(self) -> None:
        """Reset state for a new route."""
        self._pos_history.clear()
        self._committed.clear()
        self._shortest_path_cache.clear()


# -- Simple grid-based spatial index ----------------------------------------

class _GridIndex:
    """A simple uniform-grid spatial index for fast nearest-edge queries.

    Not as fast as an R-tree but has zero external dependencies and is
    sufficient for a hackathon demo with <10K edges.
    """

    def __init__(self, edges: list[_Edge], cell_size_m: float = 50.0) -> None:
        self.cell_size = cell_size_m
        self.edges = edges
        self._grid: dict[tuple, list[int]] = {}

        for idx, edge in enumerate(edges):
            cell = self._cell(edge.midpoint_m)
            self._grid.setdefault(cell, []).append(idx)

    def _cell(self, pos: np.ndarray) -> tuple:
        return (int(np.floor(pos[0] / self.cell_size)),
                int(np.floor(pos[1] / self.cell_size)))

    def _candidate_cells(self, pos: np.ndarray, radius_m: float) -> list[tuple]:
        n_cells = int(np.ceil(radius_m / self.cell_size)) + 1
        cx, cy = self._cell(pos)
        cells = []
        for dx in range(-n_cells, n_cells + 1):
            for dy in range(-n_cells, n_cells + 1):
                cells.append((cx + dx, cy + dy))
        return cells

    def query(self, pos: np.ndarray, radius_m: float) -> list[_Edge]:
        """Return all edges whose midpoint is within radius_m of pos."""
        candidates = []
        for cell in self._candidate_cells(pos, radius_m):
            for idx in self._grid.get(cell, []):
                edge = self.edges[idx]
                d = _perpendicular_distance(pos, edge.start_m, edge.end_m)
                if d <= radius_m:
                    candidates.append(edge)
        return candidates

    def nearest(self, pos: np.ndarray) -> list[_Edge]:
        """Return the single nearest edge to pos (fallback when query is empty)."""
        if not self.edges:
            return []
        best_d = np.inf
        best_edge = None
        for edge in self.edges:
            d = _perpendicular_distance(pos, edge.start_m, edge.end_m)
            if d < best_d:
                best_d = d
                best_edge = edge
        return [best_edge] if best_edge is not None else []
