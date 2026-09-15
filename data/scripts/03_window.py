"""Window per-model and derive labels (Section 3.2 step 3 + Section 3.3).

IMPORTANT correction vs. a literal reading of Section 4's "input: accel
xyz, gyro xyz [, mag xyz]" for alignment_net/channel_a/channel_b: the
IO-VNBD *vehicle* (V-) stream does NOT contain raw accelerometer/
gyroscope/magnetometer data - it's CAN-bus data (Table 3), whose only
motion-sensor-like fields are 2-axis "Indicated Longitudinal/Lateral g"
and no gyro at all. Only the *smartphone* (S-) stream has a real 9-axis
IMU (Table 4). Since the deployed system only ever has the phone's IMU
at inference time (Section 0: "phone-based dead reckoning" / Section
7.1: raw SensorManager), the S- stream is unambiguously the correct
input source for every model in Section 4, and V- is ground truth
(labels) only. Get this backwards and Channel A trains on 6 channels
that don't exist on-device. Flagging this loudly since Section 4 itself
doesn't name the source stream.

Outputs, per model, per split, into data/processed/<model>/<split>/:
  windows.npy  (N, window_len, n_channels) float32, RAW (not normalized -
               normalization happens at Dataset.__getitem__ time per
               models/*/dataset.py, using norm_stats.json from
               04_normalize.py).
  labels.npy   (N, label_dim) float32
  meta.json    session_id per window (index-aligned with the two arrays
               above) - for traceability back to a source route, e.g.
               when the benchmark replay tool needs to know which
               session a bad prediction came from.

BUILDERS implemented (Phase 1 additions noted):
  alignment_net        - (200, 9), pitch/roll/sin(yaw)/cos(yaw)
  channel_a_velocity   - (200, 6), forward velocity m/s
  channel_b_velocity   - (400, 3), forward velocity m/s [Phase 1.2]
  road_signature       - (200, 6), segment_id int [Phase 1.4]

QUALITY GATE: reads alignment_report.json from --aligned-dir and skips
sessions with undefined/low-confidence alignment. Pass --include-flagged
to override (prints a loud warning when used).

ROTATION AUGMENTATION (--augment-rotation): Adds random small rotations
of the 3D accel and gyro vectors to simulate different phone mount angles.
Motivated by DVSE (arXiv:2505.18490, 2025) ablation showing this was the
single largest gain. Off by default - always off for val/test splits.
Deviation from MIP Section 3.4 (which specifies time-jitter, noise,
gain/bias perturbation). Measured effect reported in final report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iovnbd_common import make_windows


WINDOW_CONFIGS = {
    "alignment_net": {"window_len": 200, "stride": 100},       # 2s / 50% overlap, Section 4.1
    "channel_a_velocity": {"window_len": 500, "stride": 250},   # 5s / 50% overlap, Section 4.2
    "channel_b_velocity": {"window_len": 400, "stride": 200},   # 4s / 50% overlap, Section 4.3
    "road_signature": {"window_len": 200, "stride": 100},        # Section 4.4
    "calibration_adapter": {"window_len": 200, "stride": 100},   # Section 4.5 - TODO, needs real calibration-drive
}


# ---------------------------------------------------------------------------
# Rotation augmentation (Phase 1.3 / DVSE paper deviation)
# ---------------------------------------------------------------------------

def _random_rotation_matrix(rng: np.random.Generator,
                             max_yaw_deg: float = 30.0,
                             max_pitch_deg: float = 15.0,
                             max_roll_deg: float = 15.0) -> np.ndarray:
    """Sample a random small SO(3) rotation matrix for phone-mount simulation.

    Generates a rotation with:
      yaw   (rotation about z-axis): ±max_yaw_deg
      pitch (rotation about y-axis): ±max_pitch_deg
      roll  (rotation about x-axis): ±max_roll_deg

    Returns: (3, 3) float32 rotation matrix R such that v_rotated = R @ v.
    """
    yaw   = np.deg2rad(rng.uniform(-max_yaw_deg,   max_yaw_deg))
    pitch = np.deg2rad(rng.uniform(-max_pitch_deg, max_pitch_deg))
    roll  = np.deg2rad(rng.uniform(-max_roll_deg,  max_roll_deg))

    # Rz (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1]], dtype=np.float32)
    # Ry (pitch)
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
    # Rx (roll)
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(roll), -np.sin(roll)],
                   [0,  np.sin(roll),  np.cos(roll)]], dtype=np.float32)
    return Rz @ Ry @ Rx  # combined rotation


def _apply_rotation_augmentation(windows: np.ndarray,
                                  accel_slice: slice,
                                  gyro_slice: slice | None,
                                  rng: np.random.Generator) -> np.ndarray:
    """Apply independent random rotations to accel (and optionally gyro)
    channels of each window.

    Args:
        windows: (N, T, C) float32 raw IMU windows.
        accel_slice: slice into C selecting the 3 accel channels.
        gyro_slice:  slice into C selecting the 3 gyro channels, or None.
        rng: seeded Generator for reproducibility.

    Returns:
        (N, T, C) augmented windows (new array, original not modified).
    """
    aug = windows.copy()
    n = len(aug)
    for i in range(n):
        R = _random_rotation_matrix(rng)
        # accel: (T, 3) -> apply R to each timestep
        aug[i, :, accel_slice] = (R @ aug[i, :, accel_slice].T).T
        if gyro_slice is not None:
            R_gyro = _random_rotation_matrix(rng)  # independent rotation for gyro
            aug[i, :, gyro_slice] = (R_gyro @ aug[i, :, gyro_slice].T).T
    return aug


# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------

def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def build_alignment_net(df: pd.DataFrame, cfg: dict, min_speed_kmh: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Section 4.1 / 3.3: pitch/roll from the gravity vector, yaw offset
    from device heading vs. GPS course-over-ground - only where the GNSS
    course is well-defined, i.e. the vehicle is actually moving. Windows
    below `min_speed_kmh` mean speed produce no yaw label and are
    dropped (see module docstring for why we can't just leave yaw at 0).
    """
    g_cols = ["s_gravity_x", "s_gravity_y", "s_gravity_z"]
    imu_cols = ["s_accel_x", "s_accel_y", "s_accel_z",
                "s_gyro_yaw", "s_gyro_pitch", "s_gyro_roll",
                "s_mag_x", "s_mag_y", "s_mag_z"]
    missing = [c for c in imu_cols + g_cols + ["v_heading_deg", "v_velocity_kmh"] if c not in df.columns]
    if missing:
        raise ValueError(f"alignment_net needs columns not present in aligned frame: {missing}")

    x_windows = make_windows(df[imu_cols].to_numpy(dtype=np.float32), **cfg)
    g_windows = make_windows(df[g_cols].to_numpy(dtype=np.float32), **cfg)
    heading_windows = make_windows(df[["v_heading_deg"]].to_numpy(dtype=np.float32), **cfg)
    speed_windows = make_windows(df[["v_velocity_kmh"]].to_numpy(dtype=np.float32), **cfg)

    g_mean = g_windows.mean(axis=1)  # (N, 3)
    gx, gy, gz = g_mean[:, 0], g_mean[:, 1], g_mean[:, 2]
    g_norm = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-8
    pitch = np.arcsin(np.clip(-gx / g_norm, -1, 1))
    roll = np.arcsin(np.clip(gy / g_norm, -1, 1))

    mean_speed = speed_windows.mean(axis=(1, 2))
    valid = mean_speed >= min_speed_kmh

    orientation_yaw = None  # placeholder if S orientation_yaw exists; else fall back to a zero prior
    if "s_orientation_yaw" in df.columns:
        yaw_windows = make_windows(df[["s_orientation_yaw"]].to_numpy(dtype=np.float32), **cfg)
        orientation_yaw = yaw_windows[:, -1, 0]  # value at window's end, matches Section 4.1's "yaw offset" being a single number per window
    course_over_ground = heading_windows[:, -1, 0]

    if orientation_yaw is not None:
        yaw_offset_deg = _wrap_deg(orientation_yaw - course_over_ground)
    else:
        yaw_offset_deg = np.zeros_like(course_over_ground)

    yaw_rad = np.deg2rad(yaw_offset_deg)
    labels = np.stack([pitch, roll, np.sin(yaw_rad), np.cos(yaw_rad)], axis=1).astype(np.float32)

    return x_windows[valid], labels[valid]


def build_channel_a(df: pd.DataFrame, cfg: dict,
                    delta_v: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Section 4.2 / 3.3: 6ch phone IMU (accel+gyro) in, forward velocity
    (m/s, from the V/CAN ground truth) at the window's end timestamp out.

    delta_v=True (Phase 2.1 / MQN paper deviation): instead of predicting
    absolute velocity, predict the velocity change from the previous window.
    At inference time, the running velocity estimate must be maintained by
    the caller (RealVelocityEstimator in components.py).

    Deviation note (if delta_v=True): MIP Section 4.2 specifies absolute
    velocity prediction. The Δv reframing is motivated by MQN/DLAIDR
    (arXiv:2407.16387, 2024) showing ~20% RMSE improvement. The UKF
    receives accumulated velocity, not Δv directly.
    """
    imu_cols = ["s_accel_x", "s_accel_y", "s_accel_z",
                "s_gyro_yaw", "s_gyro_pitch", "s_gyro_roll"]
    missing = [c for c in imu_cols + ["v_velocity_kmh"] if c not in df.columns]
    if missing:
        raise ValueError(f"channel_a_velocity needs columns not present in aligned frame: {missing}")

    x_windows = make_windows(df[imu_cols].to_numpy(dtype=np.float32), **cfg)
    v_windows = make_windows(df[["v_velocity_kmh"]].to_numpy(dtype=np.float32), **cfg)
    velocity_mps = v_windows[:, -1, 0] / 3.6  # (N,)

    if delta_v:
        # Δv: difference from previous window. First window has no previous,
        # so Δv[0] = 0 (the velocity is unknown at the start of a route).
        labels = np.diff(velocity_mps, prepend=velocity_mps[0]).astype(np.float32)
    else:
        labels = velocity_mps.astype(np.float32)

    return x_windows, labels.reshape(-1, 1)


def build_channel_b(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Section 4.3 / 3.3: 3ch phone accel only in, forward velocity (m/s,
    from the V/CAN ground truth) at the window's end timestamp out.

    4s window (400 samples at 100 Hz), 50% overlap (stride 200).
    CarSpeedNet-style architecture (MIP Section 4.3). Input channels are
    accel xyz ONLY - no gyro. This is the independence contract: Channel B
    must not use gyro so it cannot share the same single-point-of-failure
    as Channel A. Both sharing a gyro failure would defeat the purpose of
    having two independent channels.

    Label is identical to Channel A (velocity at window end) but the
    4s window gives more temporal context - the paper this architecture
    reproduces (CarSpeedNet, arXiv:2401.07468) reports 1.8 m/s RMSE
    at this exact window size and input spec, which is the Phase 0
    prediction target.
    """
    imu_cols = ["s_accel_x", "s_accel_y", "s_accel_z"]
    missing = [c for c in imu_cols + ["v_velocity_kmh"] if c not in df.columns]
    if missing:
        raise ValueError(f"channel_b_velocity needs columns not present in aligned frame: {missing}")

    x_windows = make_windows(df[imu_cols].to_numpy(dtype=np.float32), **cfg)
    v_windows = make_windows(df[["v_velocity_kmh"]].to_numpy(dtype=np.float32), **cfg)
    velocity_mps = v_windows[:, -1, 0] / 3.6
    return x_windows, velocity_mps.reshape(-1, 1).astype(np.float32)


def build_road_signature(df: pd.DataFrame, cfg: dict,
                          segment_len_m: float = 75.0) -> tuple[np.ndarray, np.ndarray]:
    """Section 4.4 / 3.3: road vibration segment classifier labels.

    Labels each window with an integer segment_id derived from cumulative
    GPS arc-length along the route. Segment boundaries are at every
    `segment_len_m` meters of cumulative distance (midpoint of MIP's
    50-100m range).

    Note: this generates route-relative segment IDs (0, 1, 2... per session),
    NOT globally unique corridor IDs. For the hackathon demo, each IO-VNBD
    session is treated as its own corridor. Global cross-session IDs require
    the secondary Indian drive dataset (MIP Section 3.1's "secondary dataset")
    and an OSM-based edge assignment (Phase 5 of the implementation plan).

    The classifier's deployment rule (MIP Section 4.4): only trigger a
    drift-reset when softmax max-probability > 0.85. That threshold is
    not enforced here - it's enforced in the UKF's measurement update
    (fusion_core/python_prototype/ukf.py FusionConfig.road_signature_confidence_threshold).
    """
    imu_cols = ["s_accel_x", "s_accel_y", "s_accel_z",
                "s_gyro_yaw", "s_gyro_pitch", "s_gyro_roll"]
    required = imu_cols + ["v_lat", "v_lon", "v_velocity_kmh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"road_signature needs columns not present in aligned frame: {missing}")

    # --- Compute cumulative arc-length from GPS lat/lon ---
    lat_rad = np.deg2rad(df["v_lat"].to_numpy(dtype=np.float64))
    lon_rad = np.deg2rad(df["v_lon"].to_numpy(dtype=np.float64))
    R_earth_m = 6_371_000.0

    # Haversine step distances between consecutive rows
    dlat = np.diff(lat_rad)
    dlon = np.diff(lon_rad)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon / 2) ** 2
    step_dist_m = 2 * R_earth_m * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    cum_dist_m = np.concatenate([[0.0], np.cumsum(step_dist_m)])

    # Assign segment ID per row: floor(cum_dist / segment_len_m)
    # Clip to int32 max just in case (very long routes won't overflow in practice)
    segment_ids_per_row = (cum_dist_m / segment_len_m).astype(np.int32)

    # --- Window IMU features ---
    x_windows = make_windows(df[imu_cols].to_numpy(dtype=np.float32), **cfg)

    # Label: segment_id at the window's last timestep.
    # We need to align the segment_id array with the windowing stride.
    n_rows = len(df)
    window_len = cfg["window_len"]
    stride = cfg["stride"]
    if n_rows < window_len:
        return np.empty((0, window_len, len(imu_cols)), dtype=np.float32), np.empty((0, 1), dtype=np.int32)

    end_indices = range(window_len - 1, n_rows, stride)
    label_ids = segment_ids_per_row[list(end_indices)[:len(x_windows)]]
    labels = label_ids.reshape(-1, 1).astype(np.int32)

    # Filter windows that are below a minimum speed threshold (stationary
    # segments have near-zero vibration and would trivially separate from
    # all moving segments, skewing the classifier).
    v_windows = make_windows(df[["v_velocity_kmh"]].to_numpy(dtype=np.float32), **cfg)
    mean_speed = v_windows.mean(axis=(1, 2))
    moving = mean_speed >= 5.0  # km/h threshold - drop near-stationary windows

    return x_windows[moving], labels[moving]


# ---------------------------------------------------------------------------
# Builder registry
# ---------------------------------------------------------------------------

BUILDERS = {
    "alignment_net": build_alignment_net,
    "channel_a_velocity": build_channel_a,
    "channel_b_velocity": build_channel_b,
    "road_signature": build_road_signature,
    # calibration_adapter: TODO - needs a real vehicle calibration-drive session
    # (MIP Section 3.1 "secondary dataset"), not IO-VNBD.
}


# ---------------------------------------------------------------------------
# Alignment quality gate
# ---------------------------------------------------------------------------

def _load_alignment_quality(aligned_dir: Path) -> dict[str, dict] | None:
    """Read alignment_report.json written by 02_align.py, if present.
    Returns None (not {}) when the file is missing, so the caller can
    tell "no report, gate disabled" apart from "report exists, empty".
    """
    report_path = aligned_dir / "alignment_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(WINDOW_CONFIGS.keys()))
    ap.add_argument("--manifest", default="data/processed/split_manifest.json")
    ap.add_argument("--aligned-dir", default="data/raw/_aligned")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--min-corr", type=float, default=0.5,
                     help="Sessions with recorded alignment correlation below this are skipped.")
    ap.add_argument("--include-flagged", action="store_true",
                     help="Window every aligned session regardless of alignment quality. "
                          "Never use this for a production dataset.")
    ap.add_argument("--delta-v", action="store_true",
                     help="[channel_a_velocity only] Use Δv (velocity increment) labels instead "
                          "of absolute velocity. Deviation from MIP Section 4.2 - see Phase 2.1 "
                          "of the implementation plan and the MQN paper (arXiv:2407.16387).")
    ap.add_argument("--augment-rotation", action="store_true",
                     help="Add rotation augmentation to TRAIN split windows (accel and gyro axes "
                          "are independently rotated by a random small angle to simulate different "
                          "phone mount orientations). Off by default. Val/test splits are never "
                          "augmented. Deviation from MIP Section 3.4 - see DVSE (arXiv:2505.18490).")
    ap.add_argument("--rotation-seed", type=int, default=42,
                     help="RNG seed for rotation augmentation reproducibility.")
    ap.add_argument("--segment-len-m", type=float, default=75.0,
                     help="[road_signature only] Arc-length per road segment in meters. "
                          "Midpoint of MIP Section 4.4's 50-100m range.")
    args = ap.parse_args()

    if args.model not in BUILDERS:
        raise SystemExit(
            f"{args.model} isn't implemented yet - see this script's module "
            "docstring for the priority order."
        )

    if args.delta_v and args.model != "channel_a_velocity":
        raise SystemExit("--delta-v only applies to --model channel_a_velocity")

    if args.augment_rotation and args.model not in ("channel_a_velocity", "channel_b_velocity", "alignment_net", "road_signature"):
        raise SystemExit("--augment-rotation only applies to IMU-input models")

    manifest = json.loads(Path(args.manifest).read_text())
    cfg = WINDOW_CONFIGS[args.model]
    builder = BUILDERS[args.model]
    aligned_dir = Path(args.aligned_dir)
    out_root = Path(args.out_dir) / args.model

    quality = _load_alignment_quality(aligned_dir)
    if quality is None:
        print(
            f"WARNING: no alignment_report.json found in {aligned_dir} - the alignment quality "
            "gate is DISABLED and every *_aligned.parquet file will be windowed with no check."
        )
    if args.include_flagged:
        print(
            "WARNING: --include-flagged is set - low-confidence sessions will be windowed. "
            "Do not use this for a dataset that trains a kept model."
        )

    per_split: dict[str, list] = {"train": [], "val": [], "test": []}
    per_split_meta: dict[str, list] = {"train": [], "val": [], "test": []}
    skipped: list[tuple[str, str]] = []

    for sid, info in manifest["sessions"].items():
        split = info.get("split")
        if split not in per_split:
            continue
        aligned_path = aligned_dir / f"{sid}_aligned.parquet"
        if not aligned_path.exists():
            continue

        if quality is not None and not args.include_flagged:
            entry = quality.get(sid)
            if entry is None:
                skipped.append((sid, "no entry in alignment_report.json"))
                continue
            corr = entry.get("corr")
            if corr is None:
                skipped.append((sid, "undefined (NaN) alignment correlation"))
                continue
            if corr < args.min_corr:
                skipped.append((sid, f"alignment corr={corr:.2f} below --min-corr={args.min_corr}"))
                continue
            if entry.get("note") == "boundary":
                skipped.append((
                    sid,
                    f"alignment lag landed on the search boundary (corr={corr:.2f}) - "
                    "re-run 02_align.py with larger --max-lag for this session.",
                ))
                continue

        df = pd.read_parquet(aligned_path)
        try:
            # Pass extra kwargs only to the builders that accept them
            if args.model == "channel_a_velocity":
                windows, labels = builder(df, cfg, delta_v=args.delta_v)
            elif args.model == "road_signature":
                windows, labels = builder(df, cfg, segment_len_m=args.segment_len_m)
            else:
                windows, labels = builder(df, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED windowing {sid} for {args.model}: {e}")
            continue
        if len(windows) == 0:
            continue
        per_split[split].append((windows, labels))
        per_split_meta[split].extend([sid] * len(windows))

    if skipped:
        print(f"\n{len(skipped)} session(s) skipped due to low-confidence alignment:")
        for sid, reason in skipped:
            print(f"  {sid}: {reason}")
        print()

    # --- Rotation augmentation (train split only) ---
    if args.augment_rotation and per_split["train"]:
        rng = np.random.default_rng(args.rotation_seed)
        print(f"\nApplying rotation augmentation to TRAIN split (seed={args.rotation_seed})...")

        # Determine accel/gyro slices by model
        if args.model == "channel_a_velocity":
            accel_sl = slice(0, 3)   # s_accel_x/y/z
            gyro_sl  = slice(3, 6)   # s_gyro_yaw/pitch/roll
        elif args.model == "channel_b_velocity":
            accel_sl = slice(0, 3)   # s_accel_x/y/z only (no gyro for Channel B)
            gyro_sl  = None
        elif args.model in ("alignment_net", "road_signature"):
            accel_sl = slice(0, 3)
            gyro_sl  = slice(3, 6)
        else:
            accel_sl = slice(0, 3)
            gyro_sl  = None

        augmented_chunks = []
        for windows, labels in per_split["train"]:
            aug_windows = _apply_rotation_augmentation(windows, accel_sl, gyro_sl, rng)
            augmented_chunks.append((aug_windows, labels))

        # Append augmented windows to the training set (not replacing originals)
        per_split["train"].extend(augmented_chunks)
        orig_n = sum(len(c[0]) for c in per_split["train"]) // 2
        aug_n  = sum(len(c[0]) for c in augmented_chunks)
        print(f"  Added {aug_n} augmented windows alongside {orig_n} originals → {orig_n + aug_n} total train windows.")
        # Extend meta with duplicated session IDs for the augmented windows
        per_split_meta["train"].extend(per_split_meta["train"][:aug_n])

    # --- Save outputs ---
    for split, chunks in per_split.items():
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if not chunks:
            print(f"[{args.model}/{split}] no windows produced.")
            continue
        windows = np.concatenate([c[0] for c in chunks], axis=0)
        labels = np.concatenate([c[1] for c in chunks], axis=0)
        np.save(split_dir / "windows.npy", windows)
        np.save(split_dir / "labels.npy", labels)
        (split_dir / "meta.json").write_text(json.dumps({"session_ids": per_split_meta[split]}))
        print(f"[{args.model}/{split}] {windows.shape[0]} windows, shape {windows.shape[1:]}, labels {labels.shape[1:]}")

    # --- Print augmentation note for the record ---
    if args.augment_rotation:
        print("\nRotation augmentation ON: deviation from MIP Section 3.4.")
        print("  Motivated by DVSE (arXiv:2505.18490) -- single largest ablation gain.")
        print("  Record before/after RMSE when evaluating to quantify actual effect.")
    if args.delta_v:
        print("\n--delta-v ON: deviation from MIP Section 4.2 (absolute velocity -> delta-v residual).")
        print("  Motivated by MQN/DLAIDR (arXiv:2407.16387) -- ~20% RMSE improvement.")
        print("  Inference requires accumulating delta-v into a running estimate (see components.py).")


if __name__ == "__main__":
    main()
