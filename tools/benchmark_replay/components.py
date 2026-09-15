"""The five swappable pieces of the benchmark pipeline, per handoff
doc section 2: Channel A, Channel B, road-signature classifier, the
UKF fusion core, and map-matching. Each is dummy or real behind the
same interface, selected independently in config.yaml, so pipeline.py
never needs to know which is currently wired up.

Checkpoint 0 bar (per Section 13 / handoff doc section 9): all five
dummy. Flip each to `real` in config.yaml one at a time as Layer 1 /
Layer 2 Person A deliver actual implementations - none exist in this
repo yet as of this writing (`data/processed/` and
`fusion_core/python_prototype/` are both still empty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# --- Channel A / Channel B velocity estimators -----------------------------


class VelocityEstimator(Protocol):
    def estimate(self, route, i: int) -> float:
        """Forward speed estimate (m/s) at route index i."""
        ...


@dataclass
class DummyVelocityEstimator:
    """Per handoff doc section 2: return a slightly-noised version of
    ground-truth velocity, not unrelated random noise - a UKF fed pure
    noise won't produce a meaningful drift number even for
    skeleton-testing purposes.
    """

    noise_std_mps: float = 0.3
    seed: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def estimate(self, route, i: int) -> float:
        true_speed = float(np.linalg.norm(route.vel[i]))
        return true_speed + self._rng.normal(0.0, self.noise_std_mps)


class RealVelocityEstimator:
    """Trained Channel A or Channel B ONNX velocity model (MIP Section 4.2 / 4.3).

    Loads the exported INT8 (or FP32 fallback) ONNX model plus its
    norm_stats.json and runs inference on the raw IMU window ending at
    each route step. Norm stats are applied at inference time, matching
    the Dataset.__getitem__ convention in models/*/dataset.py.

    delta_v_mode (Phase 2.1 deviation from MIP Section 4.2):
        If the model was trained on Δv labels (--delta-v flag in 03_window.py),
        set delta_v_mode=True. The estimator then accumulates Δv predictions
        into a running velocity estimate that it returns. The UKF receives this
        running estimate, NOT the Δv directly.
        If False (default, absolute velocity mode), model output is returned as-is.
    """

    def __init__(self,
                 onnx_path: str,
                 norm_stats_path: str,
                 window_len: int = 200,
                 n_imu_channels: int = 6,
                 delta_v_mode: bool = False) -> None:
        import json
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path,
                                            providers=["CUDAExecutionProvider",
                                                       "CPUExecutionProvider"])
        stats = json.loads(open(norm_stats_path).read())
        self.mean = np.array(stats["mean"], dtype=np.float32)  # (n_imu_channels,)
        self.std  = np.array(stats["std"],  dtype=np.float32)  # (n_imu_channels,)
        self.input_name = self.session.get_inputs()[0].name
        self.window_len = window_len
        self.n_imu_channels = n_imu_channels
        self.delta_v_mode = delta_v_mode
        self._v_running = 0.0  # accumulated velocity (delta_v_mode only)

    def estimate(self, route, i: int) -> float:
        """Run inference and return forward speed estimate (m/s) at step i."""
        window = route.get_imu_window(i, self.window_len, self.n_imu_channels)
        # Normalize: (T, C) raw -> apply mean/std -> (C, T) channels-first
        norm = (window - self.mean) / (self.std + 1e-8)  # (T, C)
        x = norm.T[np.newaxis].astype(np.float32)         # (1, C, T)
        out = self.session.run(None, {self.input_name: x})[0]
        pred = float(out[0, 0])
        if self.delta_v_mode:
            self._v_running += pred
            return max(0.0, self._v_running)  # velocity can't be negative
        return pred

    def reset(self) -> None:
        """Reset the running velocity accumulator (call at route start)."""
        self._v_running = 0.0


# --- Road-signature drift-anchor classifier ---------------------------------


@dataclass
class RoadSignatureResult:
    segment_id: str | None
    confidence: float


class RoadSignatureEstimator(Protocol):
    def classify(self, route, i: int) -> RoadSignatureResult:
        ...


@dataclass
class DummyRoadSignatureEstimator:
    """Per handoff doc section 2: always return "no confident match" -
    a safe no-op default that never triggers a drift-reset."""

    confidence_threshold: float = 0.85

    def classify(self, route, i: int) -> RoadSignatureResult:
        return RoadSignatureResult(segment_id=None, confidence=0.0)


class RealRoadSignatureEstimator:
    """Placeholder for the trained road-signature classifier (MIP
    Section 4.4), which also needs at least one road-signature pack
    trained on real collected corridor data (Section 3.1's "secondary"
    dataset) before it can run for real - see Section 12, Layer 1
    deliverables."""

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "Real road-signature classifier not implemented yet - no "
            "trained model or road-signature pack exists under "
            "models/road_signature/. Use components.road_signature: dummy "
            "in config.yaml until Layer 1 delivers one."
        )


# --- UKF fusion core ---------------------------------------------------------


@dataclass
class FusionState:
    pos: np.ndarray  # (2,) [north, east], meters
    vel: np.ndarray  # (2,) [v_north, v_east], m/s
    heading: float  # rad


class FusionCore(Protocol):
    def step(
        self,
        state: FusionState,
        dt: float,
        accel_body: np.ndarray,
        gyro_yaw: float,
        channel_a_speed: float,
        channel_b_speed: float,
        gnss_pos: np.ndarray | None,
        road_signature: RoadSignatureResult,
    ) -> FusionState:
        ...


@dataclass
class DummyConstantVelocityFusion:
    """Per handoff doc section 2: "a bare constant-velocity integrator
    is enough" before fusion_core/python_prototype/ has anything real.

    This is deliberately NOT a Kalman filter - no covariance, no
    innovation weighting, no GNSS/Channel-A/B disagreement handling
    (MIP Section 5.4). It exists purely to let the rest of the
    benchmark tool (blackout simulation, drift metric, plotting,
    regression logging) be built and tested end to end before the real
    UKF exists. Swap to the real fusion_core/python_prototype the
    moment Layer 2 Person A has even a rough version - do not build a
    second, throwaway filter here beyond this one, per the handoff doc.

    Uses whichever of GNSS / Channel A / Channel B is available at
    each step: GNSS position directly when present, otherwise
    dead-reckons off the average of Channel A/B speed and the
    gyro-integrated heading. A confident road-signature match resets
    position to the segment reference point if one is ever supplied by
    a non-dummy classifier.
    """

    def step(
        self,
        state: FusionState,
        dt: float,
        accel_body: np.ndarray,
        gyro_yaw: float,
        channel_a_speed: float,
        channel_b_speed: float,
        gnss_pos: np.ndarray | None,
        road_signature: RoadSignatureResult,
        road_signature_threshold: float = 0.85,
    ) -> FusionState:
        heading = state.heading + gyro_yaw * dt
        speed = (channel_a_speed + channel_b_speed) / 2.0
        vel = np.array([speed * np.cos(heading), speed * np.sin(heading)])
        pos = state.pos + (state.vel + vel) / 2.0 * dt

        if gnss_pos is not None:
            # GNSS available this step: trust it directly (no
            # covariance-weighted blend - that's real UKF territory).
            pos = gnss_pos.copy()
        elif (
            road_signature.segment_id is not None
            and road_signature.confidence >= road_signature_threshold
        ):
            # Placeholder for Section 4's drift-anchor reset - not
            # reachable with DummyRoadSignatureEstimator, kept here so
            # the real classifier can be dropped in without touching
            # this method.
            pass

        return FusionState(pos=pos, vel=vel, heading=heading)


class RealUkfFusion:
    """Wraps fusion_core/python_prototype/ukf.py's DualChannelUkf
    (MIP Section 5) behind this module's FusionCore protocol. Bridges
    the benchmark tool's per-step call shape (single FusionState in,
    single FusionState out, no persistent object of its own) onto the
    UKF wrapper's stateful `step()`, which needs to be constructed once
    with an initial state and then carries its own x/P across calls -
    so the actual UKF instance is lazily created on first `step()` and
    reused after that, ignoring the `state` argument on every call
    after the first (the UKF's internal state is the source of truth
    once it exists, per the protocol's own state-threading contract
    being just a convenience for the dummy fusion cores that don't
    keep internal state).
    """

    def __init__(self, road_signature_confidence_threshold: float = 0.85) -> None:
        from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig

        self._DualChannelUkf = DualChannelUkf
        self._config = FusionConfig(
            road_signature_confidence_threshold=road_signature_confidence_threshold
        )
        self._ukf = None

    def step(
        self,
        state: FusionState,
        dt: float,
        accel_body: np.ndarray,
        gyro_yaw: float,
        channel_a_speed: float,
        channel_b_speed: float,
        gnss_pos: np.ndarray | None,
        road_signature: RoadSignatureResult,
        road_signature_threshold: float = 0.85,
    ) -> FusionState:
        from fusion_core.python_prototype.ukf import UkfState

        if self._ukf is None:
            self._ukf = self._DualChannelUkf(
                UkfState(pos=state.pos.copy(), vel=state.vel.copy(), heading=state.heading),
                self._config,
            )

        road_signature_pos = None
        if road_signature.segment_id is not None:
            # tools/benchmark_replay's RoadSignatureResult (this
            # module) carries only a segment_id/confidence pair, not a
            # position - the real road-signature classifier's segment
            # midpoint lookup (MIP Section 4.4) isn't wired up here
            # yet. Nothing in this repo currently produces a non-None
            # segment_id (DummyRoadSignatureEstimator never does, and
            # RealRoadSignatureEstimator isn't implemented), so this
            # branch is unreachable today - kept explicit rather than
            # silently dropped so the gap is visible when a real
            # classifier lands.
            raise NotImplementedError(
                "RealUkfFusion received a road-signature match with no "
                "position lookup wired up - the real road-signature "
                "classifier (MIP Section 4.4) needs to supply the "
                "segment's midpoint position, not just a segment_id, "
                "before this path can be exercised."
            )

        result = self._ukf.step(
            dt=dt,
            gyro_yaw=gyro_yaw,
            channel_a_speed=channel_a_speed,
            channel_b_speed=channel_b_speed,
            gnss_pos=gnss_pos,
            gnss_vel=None,  # benchmark tool's Route has no GNSS-velocity channel (route_loader.py)
            road_signature_pos=road_signature_pos,
            road_signature_confidence=road_signature.confidence,
        )
        return FusionState(pos=result.pos, vel=result.vel, heading=result.heading)


# --- Map matching -------------------------------------------------------------


class MapMatcher(Protocol):
    def snap(self, pos: np.ndarray) -> np.ndarray:
        ...


class DummyMapMatcher:
    """Per handoff doc section 2: identity function - snapped position
    = fused position, unsnapped."""

    def snap(self, pos: np.ndarray) -> np.ndarray:
        return pos


class RealHmmMapMatcher:
    """HMM/Viterbi map-matcher (MIP Section 6) backed by an OSM road graph.

    Implemented in map_matching/hmm/viterbi_matcher.py (Phase 5 of the
    implementation plan). This wrapper loads that module and delegates
    to it. If the OSM graph file doesn't exist yet, raises a clear error
    pointing to the extraction script.
    """

    def __init__(self, graph_path: str,
                 window_size: int = 15,
                 sigma_emit_m: float = 10.0,
                 beta_transition: float = 10.0) -> None:
        try:
            from map_matching.hmm.viterbi_matcher import ViterbiMapMatcher
            self._matcher = ViterbiMapMatcher(
                graph_path=graph_path,
                window_size=window_size,
                sigma_emit_m=sigma_emit_m,
                beta_transition=beta_transition,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"OSM graph file not found: {graph_path}\n"
                "Run: python map_matching/osm_extraction/extract.py\n"
                "to generate it at build time (not at runtime)."
            )

    def snap(self, pos: np.ndarray) -> np.ndarray:
        """pos: (2,) [north_m, east_m] in local flat-earth frame.
        Returns snapped (2,) position in the same frame.
        """
        return self._matcher.snap(pos)


# --- Component factory --------------------------------------------------------


@dataclass
class ComponentSet:
    channel_a: VelocityEstimator
    channel_b: VelocityEstimator
    road_signature: RoadSignatureEstimator
    fusion: FusionCore
    map_matching: MapMatcher
    road_signature_threshold: float


def build_components(config: dict) -> ComponentSet:
    """Instantiate the five components per config.yaml's
    `components.*: dummy|real` switches.

    config.yaml conventions for 'real' components (Phase 4):
      components:
        channel_a: real
        channel_b: real
        road_signature: dummy
        fusion: real
        map_matching: dummy

      # Required when channel_a/channel_b: real
      channel_a:
        onnx_path: models/channel_a_velocity/exported/channel_a.onnx
        norm_stats_path: data/processed/channel_a_velocity/norm_stats.json
        window_len: 200          # 2s @ 100Hz
        n_imu_channels: 6        # accel xyz + gyro xyz
        delta_v_mode: false      # true if model was trained with --delta-v
      channel_b:
        onnx_path: models/channel_b_velocity/exported/channel_b.onnx
        norm_stats_path: data/processed/channel_b_velocity/norm_stats.json
        window_len: 400          # 4s @ 100Hz
        n_imu_channels: 3        # accel xyz only
        delta_v_mode: false

      # Required when map_matching: real
      map_matching:
        graph_path: map_matching/osm_extraction/corridor.graphml
        window_size: 15
        sigma_emit_m: 10.0
        beta_transition: 10.0
    """
    which = config["components"]
    threshold = config.get("road_signature_confidence_threshold", 0.85)

    if which["channel_a"] == "dummy":
        channel_a = DummyVelocityEstimator(seed=1)
    else:
        ca_cfg = config.get("channel_a", {})
        channel_a = RealVelocityEstimator(
            onnx_path=ca_cfg.get("onnx_path", "models/channel_a_velocity/exported/channel_a.onnx"),
            norm_stats_path=ca_cfg.get("norm_stats_path", "data/processed/channel_a_velocity/norm_stats.json"),
            window_len=ca_cfg.get("window_len", 200),
            n_imu_channels=ca_cfg.get("n_imu_channels", 6),
            delta_v_mode=ca_cfg.get("delta_v_mode", False),
        )

    if which["channel_b"] == "dummy":
        channel_b = DummyVelocityEstimator(seed=2)
    else:
        cb_cfg = config.get("channel_b", {})
        channel_b = RealVelocityEstimator(
            onnx_path=cb_cfg.get("onnx_path", "models/channel_b_velocity/exported/channel_b.onnx"),
            norm_stats_path=cb_cfg.get("norm_stats_path", "data/processed/channel_b_velocity/norm_stats.json"),
            window_len=cb_cfg.get("window_len", 400),
            n_imu_channels=cb_cfg.get("n_imu_channels", 3),
            delta_v_mode=cb_cfg.get("delta_v_mode", False),
        )

    road_signature = (
        DummyRoadSignatureEstimator(confidence_threshold=threshold)
        if which["road_signature"] == "dummy"
        else RealRoadSignatureEstimator()
    )
    fusion = (
        DummyConstantVelocityFusion()
        if which["fusion"] == "dummy"
        else RealUkfFusion(road_signature_confidence_threshold=threshold)
    )
    if which["map_matching"] == "dummy":
        map_matching = DummyMapMatcher()
    else:
        mm_cfg = config.get("map_matching", {})
        map_matching = RealHmmMapMatcher(
            graph_path=mm_cfg.get("graph_path", "map_matching/osm_extraction/corridor.graphml"),
            window_size=mm_cfg.get("window_size", 15),
            sigma_emit_m=mm_cfg.get("sigma_emit_m", 10.0),
            beta_transition=mm_cfg.get("beta_transition", 10.0),
        )

    return ComponentSet(
        channel_a=channel_a,
        channel_b=channel_b,
        road_signature=road_signature,
        fusion=fusion,
        map_matching=map_matching,
        road_signature_threshold=threshold,
    )
