"""Channel A accuracy diagnosis and ablation — Phase 2 of the implementation plan.

Run this BEFORE changing any model architecture to understand where the
5.09 m/s RMSE is coming from. Then re-run after each intervention to
measure the actual improvement.

Usage (from repo root):
    # 1. Diagnose the existing best checkpoint:
    python tools/diagnose_channel_a.py \\
        --checkpoint lightning_logs/version_1/checkpoints/best-epoch=10-val_loss=3.9900.ckpt

    # 2. Compare multiple window sizes (ablation table):
    python tools/diagnose_channel_a.py \\
        --checkpoint <ckpt> --ablation-window-sizes

    # 3. After retraining with delta-v, compare old vs new:
    python tools/diagnose_channel_a.py \\
        --checkpoint <new_ckpt> --compare-checkpoint <old_ckpt>

Outputs:
    reports/channel_a_diagnosis_<timestamp>.json
    reports/channel_a_diagnosis_<timestamp>.md  (human-readable with plots)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Make sure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.channel_a_velocity.dataset import ChannelAVelocityDataset
from models.channel_a_velocity.model import ChannelAVelocityNet
from models.common.export_utils import load_lightning_state_dict


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute the full regression metric suite (matching evaluate_channel_a.py)."""
    errors = pred - gt
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2
    rmse = float(np.sqrt(np.mean(sq_errors)))
    mae  = float(np.mean(abs_errors))
    ss_res = np.sum(sq_errors)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return {
        "n": int(len(pred)),
        "rmse": rmse,
        "mae": mae,
        "median_ae": float(np.median(abs_errors)),
        "p90_ae": float(np.percentile(abs_errors, 90)),
        "p95_ae": float(np.percentile(abs_errors, 95)),
        "max_ae": float(np.max(abs_errors)),
        "bias": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Per-session diagnosis
# ---------------------------------------------------------------------------

def diagnose_checkpoint(checkpoint_path: str,
                         processed_dir: str = "data/processed/channel_a_velocity",
                         stats_path: str = "data/processed/channel_a_velocity/norm_stats.json",
                         device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    """Load checkpoint, run inference on all splits, return full metrics dict."""

    model = ChannelAVelocityNet()
    model.load_state_dict(load_lightning_state_dict(checkpoint_path))
    model.eval().to(device)

    results = {"checkpoint": checkpoint_path, "splits": {}}

    for split in ("train", "val", "test"):
        try:
            ds = ChannelAVelocityDataset(processed_dir, split, stats_path)
        except FileNotFoundError as e:
            print(f"[{split}] skipped: {e}")
            continue

        all_pred, all_gt, all_meta = [], [], []

        # Load meta for per-session breakdown
        meta_path = Path(processed_dir) / split / "meta.json"
        session_ids = json.loads(meta_path.read_text())["session_ids"] if meta_path.exists() else []

        with torch.no_grad():
            for i, (window, label) in enumerate(ds):
                pred = model(window.unsqueeze(0).to(device)).cpu().numpy()[0, 0]
                gt_val = label.numpy()[0]
                all_pred.append(pred)
                all_gt.append(gt_val)
                if session_ids:
                    all_meta.append(session_ids[i])

        pred_arr = np.array(all_pred)
        gt_arr = np.array(all_gt)
        overall = compute_regression_metrics(pred_arr, gt_arr)
        overall["train_mean_baseline_rmse"] = float(np.sqrt(np.mean((gt_arr - np.mean(gt_arr)) ** 2))) if split == "test" else None

        # Per-session breakdown
        per_session = {}
        if all_meta:
            unique_sessions = sorted(set(all_meta))
            for sid in unique_sessions:
                mask = np.array([m == sid for m in all_meta])
                per_session[sid] = compute_regression_metrics(pred_arr[mask], gt_arr[mask])

        # Error breakdown by velocity range (useful for diagnosing where errors cluster)
        ranges = [(0, 5), (5, 15), (15, 30), (30, 999)]
        by_velocity_range = {}
        for lo, hi in ranges:
            mask = (gt_arr >= lo) & (gt_arr < hi)
            if mask.sum() > 0:
                by_velocity_range[f"{lo}-{hi}_mps"] = compute_regression_metrics(
                    pred_arr[mask], gt_arr[mask]
                )

        results["splits"][split] = {
            "overall": overall,
            "per_session": per_session,
            "by_velocity_range": by_velocity_range,
        }

        print(f"[{split}] N={overall['n']:,}  RMSE={overall['rmse']:.4f}  "
              f"MAE={overall['mae']:.4f}  R²={overall['r2']:.4f}  Bias={overall['bias']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Ablation table (window size)
# ---------------------------------------------------------------------------

def print_ablation_suggestion() -> None:
    """Print the window-size ablation plan (Phase 2.2).

    Actual ablation requires re-windowing + re-training, which this script
    can't do in one shot. This prints the exact commands to run.
    """
    print("\n=== Window Size Ablation (Phase 2.2) ===")
    print("To run the full ablation, execute these commands in order:\n")
    configs = [
        ("2s",  200, "[8, 16, 32, 64]",    241),
        ("5s",  500, "[16, 32, 64, 128]",  961),
        ("10s", 1000, "[32, 64, 128, 256]", 1921),
    ]
    for name, samples, dilations, rf in configs:
        print(f"  # --- {name} window ({samples} samples, dilations {dilations}, RF={rf}) ---")
        print(f"  # 1. Edit WINDOW_CONFIGS[\"channel_a_velocity\"][\"window_len\"] = {samples} in 03_window.py")
        print(f"  #    and WINDOW_CONFIGS[\"channel_a_velocity\"][\"stride\"] = {samples // 2}")
        print(f"  # 2. Edit tcn_dilations in models/channel_a_velocity/config.yaml to {dilations}")
        print(f"  python data/scripts/03_window.py --model channel_a_velocity")
        print(f"  python data/scripts/04_normalize.py --model channel_a_velocity")
        print(f"  python models/channel_a_velocity/train.py --config models/channel_a_velocity/config.yaml")
        print(f"  python tools/diagnose_channel_a.py --checkpoint <best_ckpt>\n")

    print("Predicted RMSE by window size (from Phase 0 prediction table):")
    print("  2s  → ~5.09 m/s (measured)")
    print("  5s  → ~3.0 m/s  (prediction, record actual)")
    print("  10s → ~2.0 m/s  (prediction, record actual)")
    print("\nReference: CarSpeedNet (arXiv:2401.07468) reports monotonic improvement")
    print("from 0.25s to 4s window. Literature supports longer windows are better.\n")


# ---------------------------------------------------------------------------
# Comparison (before vs after)
# ---------------------------------------------------------------------------

def compare_checkpoints(ckpt_a: str, ckpt_b: str,
                         label_a: str = "before", label_b: str = "after",
                         **kwargs) -> None:
    print(f"\n=== Comparing: {label_a} vs {label_b} ===")
    res_a = diagnose_checkpoint(ckpt_a, **kwargs)
    res_b = diagnose_checkpoint(ckpt_b, **kwargs)

    for split in ("val", "test"):
        if split not in res_a["splits"] or split not in res_b["splits"]:
            continue
        rmse_a = res_a["splits"][split]["overall"]["rmse"]
        rmse_b = res_b["splits"][split]["overall"]["rmse"]
        delta  = rmse_b - rmse_a
        pct    = delta / rmse_a * 100
        print(f"  [{split}] {label_a}: {rmse_a:.4f} m/s  →  {label_b}: {rmse_b:.4f} m/s  "
              f"(Δ={delta:+.4f} m/s, {pct:+.1f}%)")


# ---------------------------------------------------------------------------
# Delta-V reframing: sanity-check the label distribution
# ---------------------------------------------------------------------------

def check_delta_v_distribution(processed_dir: str = "data/processed/channel_a_velocity") -> None:
    """Print stats on the Δv label distribution to calibrate Huber delta choice.

    MIP specifies Huber delta=1.0 for absolute velocity (~0-30 m/s range).
    If we switch to Δv, the target range shrinks to roughly ±2-4 m/s
    between consecutive 2s windows, suggesting delta=0.5 is better.
    This function prints the actual Δv statistics from the training data.
    """
    labels_path = Path(processed_dir) / "train" / "labels.npy"
    if not labels_path.exists():
        print(f"No training labels found at {labels_path}")
        return

    v = np.load(labels_path).flatten()  # absolute velocity
    dv = np.diff(v, prepend=v[0])       # Δv

    print("\n=== Δv Label Distribution (Phase 2.1 / Huber delta calibration) ===")
    print(f"  Absolute velocity:  mean={v.mean():.2f}  std={v.std():.2f}  "
          f"p5={np.percentile(v,5):.2f}  p95={np.percentile(v,95):.2f}  m/s")
    print(f"  Δv (per window):    mean={dv.mean():.3f}  std={dv.std():.3f}  "
          f"p5={np.percentile(dv,5):.3f}  p95={np.percentile(dv,95):.3f}  m/s")
    print(f"  |Δv| 90th pct: {np.percentile(np.abs(dv), 90):.3f} m/s")
    print(f"\n  → Recommended Huber delta for Δv mode: {np.percentile(np.abs(dv), 75):.2f} m/s")
    print(f"    (75th percentile of |Δv|, so L2/L1 transition is inside the typical error)")
    print(f"    Compare to MIP's delta=1.0 (tuned for absolute velocity range).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Channel A diagnosis and ablation tool (Phase 2).")
    ap.add_argument("--checkpoint", required=True, help="Path to .ckpt file.")
    ap.add_argument("--compare-checkpoint", default=None,
                    help="Optional second checkpoint to compare against (before/after).")
    ap.add_argument("--ablation-window-sizes", action="store_true",
                    help="Print the window-size ablation command sequence (Phase 2.2).")
    ap.add_argument("--check-delta-v", action="store_true",
                    help="Print Δv label distribution stats for Huber delta calibration (Phase 2.1).")
    ap.add_argument("--processed-dir", default="data/processed/channel_a_velocity")
    ap.add_argument("--stats-path", default="data/processed/channel_a_velocity/norm_stats.json")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()

    if args.check_delta_v:
        check_delta_v_distribution(args.processed_dir)

    if args.ablation_window_sizes:
        print_ablation_suggestion()

    print(f"\n=== Diagnosing: {args.checkpoint} ===")
    results = diagnose_checkpoint(
        args.checkpoint,
        processed_dir=args.processed_dir,
        stats_path=args.stats_path,
    )

    if args.compare_checkpoint:
        compare_checkpoints(
            args.compare_checkpoint, args.checkpoint,
            label_a="before", label_b="after",
            processed_dir=args.processed_dir,
            stats_path=args.stats_path,
        )

    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"channel_a_diagnosis_{timestamp}.json"
    out_json.write_text(json.dumps(results, indent=2))

    # Write Markdown summary
    out_md = out_dir / f"channel_a_diagnosis_{timestamp}.md"
    with open(out_md, "w") as f:
        f.write(f"# Channel A Diagnosis Report\n")
        f.write(f"**Checkpoint:** `{args.checkpoint}`\n\n")
        f.write("## Per-Split RMSE Summary\n\n")
        f.write("| Split | N | RMSE | MAE | Bias | R² |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for split, data in results["splits"].items():
            m = data["overall"]
            f.write(f"| {split} | {m['n']:,} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['bias']:.4f} | {m['r2']:.4f} |\n")

        for split, data in results["splits"].items():
            if data["by_velocity_range"]:
                f.write(f"\n## {split.title()} — Error by Velocity Range\n\n")
                f.write("| Range (m/s) | N | RMSE | R² |\n")
                f.write("|---|---:|---:|---:|\n")
                for rng, m in data["by_velocity_range"].items():
                    f.write(f"| {rng} | {m['n']:,} | {m['rmse']:.4f} | {m['r2']:.4f} |\n")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
