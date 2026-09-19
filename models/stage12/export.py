"""Export the trained Stage 12 MotionSpeedNet checkpoint to ONNX for
on-device (Android/ONNX Runtime Mobile) inference.

Usage:
    python -m models.stage12.export \
        --checkpoint models/stage12/checkpoints/best_motionspeednet.pth

Follows the same mechanical export path as models/alignment_net/export.py
(models/common/export_utils.export_to_onnx, opset 17, fixed batch=1,
dynamo=False), with one deliberate deviation:

No int8/float16 quantization here. Section 4.6's quantize-then-verify
path exists to shrink models whose size or CPU cost matters on-device.
MotionSpeedNet is 57,603 parameters (~225 KB fp32) - already far below
where quantization buys anything meaningful in load time or inference
latency on a phone CPU, and its two regression heads (speed_mean via
softplus, speed_log_var feeding straight into the UKF's per-cycle R)
are exactly the kind of small-magnitude scalar output int8's coarser
quantization grid is most likely to distort. There is also no
`quantize_dynamic_int8` support for `nn.GRU` in the ONNX Runtime
version this repo pins (the op is quantized as a black box or not at
all, depending on version, and either way is untested here). Shipping
fp32 and revisiting this only if a real device profile shows the model
itself is a bottleneck (unlikely - it runs once per 10 Hz tick on a
50x4 + 50x4 input) is the smaller-risk choice.
"""

import argparse

import torch

from models.common.export_utils import export_to_onnx
from models.stage12.motion_speed_net import MotionSpeedNet, get_parameter_count


def main(checkpoint_path: str, output_dir: str = "models/stage12/exported") -> None:
    model = MotionSpeedNet()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, (
        f"checkpoint does not match MotionSpeedNet's current architecture: "
        f"missing={missing} unexpected={unexpected}"
    )
    model.eval()
    print(f"Loaded checkpoint: {get_parameter_count(model):,} parameters.")

    # (batch=1, seq_len=50, channels=4) for both branches, matching the
    # causal 50-sample / 5.0 s window every caller in tools/stage12 uses.
    dummy_acc = torch.randn(1, 50, 4)
    dummy_gy = torch.randn(1, 50, 4)

    onnx_path = f"{output_dir}/motionspeednet.onnx"
    export_to_onnx(
        model,
        (dummy_acc, dummy_gy),
        onnx_path,
        input_names=["x_acc", "x_gy"],
        output_names=["speed_mean", "speed_log_var", "yaw_rate_correction"],
    )
    print(f"Exported float32 ONNX model to {onnx_path}")

    # Round-trip check: the ONNX graph must reproduce the PyTorch model's
    # outputs bit-close on the same input before this is trusted on-device.
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        None, {"x_acc": dummy_acc.numpy(), "x_gy": dummy_gy.numpy()}
    )
    with torch.no_grad():
        torch_out = model(dummy_acc, dummy_gy)
    for name, onnx_val in zip(
        ["speed_mean", "speed_log_var", "yaw_rate_correction"], onnx_out
    ):
        torch_val = torch_out[name].numpy()
        max_abs_diff = float(np.max(np.abs(onnx_val - torch_val)))
        print(f"  {name}: max |onnx - torch| = {max_abs_diff:.3e}")
        assert max_abs_diff < 1e-4, f"{name} diverged between torch and onnx export"
    print("ONNX export verified against the PyTorch checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="models/stage12/checkpoints/best_motionspeednet.pth",
    )
    parser.add_argument("--output-dir", default="models/stage12/exported")
    args = parser.parse_args()
    main(args.checkpoint, args.output_dir)
