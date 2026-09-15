"""Dataset for Channel B - MIP Section 4.3 / 3.3.

Label (Section 3.3): forward velocity scalar per window, from the V
(vehicle ground truth) stream - same label source as Channel A, but
windowed at 4s / 400 samples instead of 2s / 200 samples (Section 3.2
step 3, Channel B uses a 4s window per Section 4.3).

Input channels: s_accel_x, s_accel_y, s_accel_z only (no gyro).
The gyro-free constraint is the independence guarantee: Channel B
cannot fail at the same time as Channel A due to a gyro hardware fault.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from models.common.normalization import apply, load_stats


class ChannelBVelocityDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str, norm_stats_path: str | Path):
        """
        Args:
            processed_dir: data/processed/channel_b_velocity/ (or wherever
                Section 3's pipeline writes windows for this model).
            split: "train" | "val" | "test".
            norm_stats_path: this model's norm_stats.json (computed on
                train split only by 04_normalize.py).
        """
        self.processed_dir = Path(processed_dir)
        self.split = split
        self.norm_stats = load_stats(norm_stats_path)

        split_dir = self.processed_dir / split
        windows_path = split_dir / "windows.npy"
        labels_path = split_dir / "labels.npy"
        if not windows_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"{windows_path} / {labels_path} not found.\n"
                "Run the following commands from the repo root first:\n"
                "  python data/scripts/03_window.py --model channel_b_velocity\n"
                "  python data/scripts/04_normalize.py --model channel_b_velocity\n"
                "See data/processed/README.md for the full pipeline order."
            )

        # (N, 400, 3) raw; normalized and transposed to (3, 400) per-item
        # below, not here. Keeping windows.npy raw matches the Channel A
        # convention and means norm_stats.json is the only normalization
        # artifact (important: the same stats must be baked into the on-device
        # preprocessing code, not derived from a pre-normalized file).
        self.windows = np.load(windows_path)  # (N, 400, 3), float32
        self.labels = np.load(labels_path)    # (N, 1), float32
        if len(self.windows) != len(self.labels):
            raise ValueError(
                f"windows/labels length mismatch in {split_dir}: "
                f"{len(self.windows)} vs {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (window, label):
            window: (3, 400) float32, channels-first, normalized.
                    Channels: [accel_x, accel_y, accel_z].
            label:  (1,) float32 - forward velocity, m/s.
        """
        window = apply(self.windows[idx : idx + 1], self.norm_stats)[0]  # (400, 3), normalized
        window = torch.from_numpy(window).float().transpose(0, 1)  # -> (3, 400)
        label = torch.from_numpy(self.labels[idx]).float()
        return window, label
