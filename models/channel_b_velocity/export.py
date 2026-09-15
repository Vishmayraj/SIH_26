"""Export the trained Channel B model to ONNX + quantized int8 - Section 4.6.

Usage:
    python models/channel_b_velocity/export.py --checkpoint <path to .ckpt>

Mirrors Channel A's export.py exactly (load Lightning ckpt via
load_lightning_state_dict, verify quantized accuracy within 5% relative
degradation per Section 4.6 step 3, fall back to fp16 if int8 fails).
"""

import argparse

import numpy as np
import onnxruntime as ort
import torch

from models.channel_b_velocity.dataset import ChannelBVelocityDataset
from models.channel_b_velocity.model import ChannelBVelocityNet
from models.common.export_utils import (
    export_to_onnx,
    load_lightning_state_dict,
    quantize_dynamic_int8,
    quantize_float16,
    verify_quantized_accuracy,
)


def _neg_rmse_mps(onnx_path: str, val_ds: ChannelBVelocityDataset) -> float:
    """Higher-is-better score for verify_quantized_accuracy: negative
    velocity RMSE (m/s) over the val set - the real Section 4.3 target
    metric, same one train.py logs as val_rmse_mps.
    """
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    sq_errors = []
    for window, label in val_ds:
        pred = session.run(None, {input_name: window.unsqueeze(0).numpy()})[0]
        sq_errors.append(float((pred[0, 0] - label.item()) ** 2))

    return -float(np.sqrt(np.mean(sq_errors)))


def main(checkpoint_path: str, output_dir: str = "models/channel_b_velocity/exported"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    model = ChannelBVelocityNet()
    model.load_state_dict(load_lightning_state_dict(checkpoint_path))

    dummy_input = torch.randn(1, 3, 400)  # batch=1, channels-first, per Section 4.6 step 1

    float32_path = f"{output_dir}/channel_b_fp32.onnx"
    int8_path    = f"{output_dir}/channel_b.onnx"   # naming per Section 4.6 step 4
    float16_path = f"{output_dir}/channel_b_fp16.onnx"

    export_to_onnx(model, dummy_input, float32_path)
    quantize_dynamic_int8(float32_path, int8_path)

    stats_path = "data/processed/channel_b_velocity/norm_stats.json"
    val_ds = ChannelBVelocityDataset("data/processed/channel_b_velocity", "val", stats_path)
    metric_fn = lambda path: _neg_rmse_mps(path, val_ds)  # noqa: E731

    if verify_quantized_accuracy(metric_fn, float32_path, int8_path):
        print(f"int8 export within 5% of float32 val RMSE - using {int8_path}")
    else:
        print(
            "int8 export failed the 5% accuracy-drop check "
            "(Section 4.6 step 3) - falling back to float16."
        )
        quantize_float16(float32_path, float16_path)
        print(f"float16 export written to {float16_path}")

    # Report final val RMSE from both exports for the final report table
    fp32_rmse = -metric_fn(float32_path)
    int8_rmse = -metric_fn(int8_path)
    rel_deg = abs(int8_rmse - fp32_rmse) / fp32_rmse * 100
    print(f"\nExport summary:")
    print(f"  FP32 val RMSE: {fp32_rmse:.4f} m/s")
    print(f"  INT8 val RMSE: {int8_rmse:.4f} m/s")
    print(f"  Relative degradation: {rel_deg:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="models/channel_b_velocity/exported")
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir)
