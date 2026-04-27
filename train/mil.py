"""MIL hybrid loss and accuracy for a **single** trajectory: ``logits`` shape ``(T,)``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["mil_hybrid_loss", "trajectory_mil_target", "prediction_matches_target"]


def trajectory_mil_target(row_label: int) -> float:
    pass


def mil_hybrid_loss(logits: torch.Tensor, y_mil: float) -> torch.Tensor:
    pass


@torch.no_grad()
def prediction_matches_target(p: torch.Tensor, y_mil: float) -> bool:
    pass
