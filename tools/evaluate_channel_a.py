"""Evaluate a Channel A checkpoint and write a reproducible metrics report.

The report is deliberately independent of Lightning's logged validation
batch averages: every metric is recomputed over the complete split and also
broken down by held-out session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.channel_a_velocity.dataset import ChannelAVelocityDataset
from models.channel_a_velocity.model import ChannelAVelocityNet


def load_lightning_state_dict(checkpoint_path: str | Path, prefix: str = "model.") -> dict:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    raw_state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in raw_state_dict.items()}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot else float("nan")
    return {
        "n": int(y_true.size),
        "rmse_mps": float(np.sqrt(np.mean(err**2))),
        "mae_mps": float(np.mean(abs_err)),
        "median_abs_error_mps": float(np.median(abs_err)),
        "p90_abs_error_mps": float(np.percentile(abs_err, 90)),
        "p95_abs_error_mps": float(np.percentile(abs_err, 95)),
        "max_abs_error_mps": float(np.max(abs_err)),
        "mean_signed_error_mps": float(np.mean(err)),
        "std_error_mps": float(np.std(err)),
        "r2": r2,
    }


def evaluate_split(model, dataset, device: torch.device):
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    ys, preds = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).cpu().numpy().reshape(-1)
            ys.append(y.numpy().reshape(-1))
            preds.append(pred)
    return np.concatenate(ys), np.concatenate(preds)


def evaluate_onnx(path: str, dataset):
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ys, preds = [], []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        pred = session.run(None, {input_name: x.numpy()[None, ...]})[0].reshape(-1)[0]
        ys.append(float(y.reshape(-1)[0]))
        preds.append(float(pred))
    return np.asarray(ys), np.asarray(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--processed-dir", default="data/processed/channel_a_velocity")
    ap.add_argument("--output", default="reports/channel_a_seed0_report.json")
    ap.add_argument("--onnx", nargs="*", default=[])
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    stats = processed / "norm_stats.json"
    model = ChannelAVelocityNet()
    model.load_state_dict(load_lightning_state_dict(args.checkpoint))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "splits": {},
        "comparison": {
            "teammate": {
                "author": "Vishmayraj",
                "implementation_reference": "upstream/main (merged at 61bfdd8)",
                "known_state": "data pipeline and Channel A export wiring are implemented; upstream documentation states the model had not yet been trained",
                "performance": "not comparable: no teammate checkpoint or metrics report is present in the checkout",
                "required_for_apples_to_apples": "same split_manifest.json, norm_stats.json, labels, held-out sessions, and this metric set",
            },
        },
    }

    train_ds = ChannelAVelocityDataset(processed, "train", stats)
    train_y, _ = evaluate_split(model, train_ds, device)
    train_mean = float(train_y.mean())
    report["train_label_mean_mps"] = train_mean

    for split in ("train", "val", "test"):
        ds = train_ds if split == "train" else ChannelAVelocityDataset(processed, split, stats)
        y_true, y_pred = evaluate_split(model, ds, device)
        split_report = {
            "model": metrics(y_true, y_pred),
            "train_mean_baseline": metrics(y_true, np.full_like(y_true, train_mean)),
        }
        meta = json.loads((processed / split / "meta.json").read_text())
        session_ids = np.asarray(meta["session_ids"])
        per_session = {}
        for sid in sorted(set(session_ids.tolist())):
            mask = session_ids == sid
            per_session[str(sid)] = metrics(y_true[mask], y_pred[mask])
        split_report["per_session"] = per_session
        report["splits"][split] = split_report

    if args.onnx:
        report["deployment"] = {}
        for onnx_path in args.onnx:
            name = Path(onnx_path).stem
            report["deployment"][name] = {}
            for split in ("val", "test"):
                ds = ChannelAVelocityDataset(processed, split, stats)
                y_true, y_pred = evaluate_onnx(onnx_path, ds)
                report["deployment"][name][split] = metrics(y_true, y_pred)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    md = output.with_suffix(".md")
    lines = [
        "# Channel A evaluation report",
        "",
        f"Checkpoint: `{report['checkpoint']}`  ",
        f"Device: `{report['device']}`  ",
        f"Parameters: `{report['model_parameters']}`  ",
        "",
        "| Split | RMSE (m/s) | MAE (m/s) | Median AE | P90 AE | P95 AE | Max AE | R2 | Baseline RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, result in report["splits"].items():
        m, b = result["model"], result["train_mean_baseline"]
        lines.append(
            f"| {split} | {m['rmse_mps']:.4f} | {m['mae_mps']:.4f} | "
            f"{m['median_abs_error_mps']:.4f} | {m['p90_abs_error_mps']:.4f} | "
            f"{m['p95_abs_error_mps']:.4f} | {m['max_abs_error_mps']:.4f} | "
            f"{m['r2']:.4f} | {b['rmse_mps']:.4f} |"
        )
    lines += [
        "",
        "Target: Channel A held-out RMSE < 0.5 m/s.",
        "",
        "Teammate comparison: upstream implementation is present, but no teammate checkpoint/metrics report is available; an apples-to-apples performance comparison is therefore pending.",
    ]
    if report.get("deployment"):
        lines += ["", "## ONNX deployment metrics", "", "| Export | Val RMSE | Test RMSE | Test MAE | Test R² |", "|---|---:|---:|---:|---:|"]
        for name, result in report["deployment"].items():
            vm, tm = result["val"], result["test"]
            lines.append(f"| {name} | {vm['rmse_mps']:.4f} | {tm['rmse_mps']:.4f} | {tm['mae_mps']:.4f} | {tm['r2']:.4f} |")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
