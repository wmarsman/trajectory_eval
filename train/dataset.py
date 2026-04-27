"""Load trajectory JSONL: one (T, 3) tensor and row label per sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

__all__ = [
    "TrajectoryJsonlDataset",
    "collate_one_trajectory",
    "N_CLASSES",
    "N_FEATURES",
]

# Row labels in JSONL (integers 0, 1, 2). MIL merges 1 and 2 vs 0.
N_CLASSES = 3
N_FEATURES = 3


def _line_to_item(line: str) -> tuple[torch.Tensor, int]:
    rec: dict[str, Any] = json.loads(line)
    label = int(rec["label"])
    traj: list[list[float]] = rec["trajectory"]
    t = torch.tensor(traj, dtype=torch.float32)
    if t.size(0) == 0:
        raise ValueError("empty trajectory")
    if t.dim() != 2 or t.size(1) != N_FEATURES:
        raise ValueError(f"Expected trajectory shape (T, 3), got {tuple(t.shape)}")
    if not (0 <= label < N_CLASSES):
        raise ValueError(f"label out of range [0, {N_CLASSES}): {label}")
    return t, label


class TrajectoryJsonlDataset(Dataset[tuple[torch.Tensor, int]]):
    """One sample = one line: ``(T, 3)`` float tensor and integer label."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self._jsonl_path = Path(jsonl_path)
        if not self._jsonl_path.is_file():
            raise FileNotFoundError(f"Not a file: {self._jsonl_path.resolve()}")
        self._items: list[tuple[torch.Tensor, int]] = []
        with self._jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                s = line.strip()
                if not s:
                    continue
                try:
                    self._items.append(_line_to_item(s))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    raise ValueError(
                        f"Bad record at {self._jsonl_path!s} line {i + 1}"
                    ) from e

    @property
    def path(self) -> Path:
        return self._jsonl_path

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._items[index]


def collate_one_trajectory(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, int]:
    """DataLoader with ``batch_size=1`` only: pass through the single trajectory."""
    if len(batch) != 1:
        raise ValueError(
            "Use batch_size=1 so each step is one trajectory (no multi-bag stacking)."
        )
    return batch[0]
