"""MIL training: one trajectory per step; instance MLP; hybrid loss on that trajectory.

- Merged row labels **1** or **2** (``y_mil==0``): BCE of **each** instance toward **1** (instance mean).
- Row label **0** (``y_mil==1``): BCE of **min** instance score toward **0** (min only).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Parent of ``train/`` on path so ``from train.…`` works when run as ``python train/train.py``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hydra
import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from train.dataset import (
    N_FEATURES,
    TrajectoryJsonlDataset,
    collate_one_trajectory,
)
from train.mil import (
    mil_hybrid_loss,
    prediction_matches_target,
    trajectory_mil_target,
)


def resolve_data_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (_PROJECT_ROOT / path).resolve()


class InstanceMlp(nn.Module):
    """MLP on each 3-D instance; one logit per time step, sigmoid gives instance score.

    ``num_hidden_layers`` is the count of ``Linear(..., hidden) + ReLU`` blocks before
    the final ``Linear(..., 1)``. If ``0``, a single ``Linear(in_features, 1)`` is used.
    """

    def __init__(self, in_features: int, hidden: int, num_hidden_layers: int) -> None:
        super().__init__()
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if num_hidden_layers == 0:
            self.net: nn.Module = nn.Linear(in_features, 1)
        else:
            parts: list[nn.Module] = []
            d = in_features
            for _ in range(num_hidden_layers):
                parts += [nn.Linear(d, hidden), nn.ReLU()]
                d = hidden
            parts.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def as_row_label(v: int | torch.Tensor) -> int:
    """Dataloader may pass a 0-dim or length-1 int tensor; normalize to ``int``."""
    if isinstance(v, torch.Tensor):
        return int(v.view(-1).item())
    return int(v)


@torch.no_grad()
def run_eval(
    model: InstanceMlp,
    loader: DataLoader,
    device: torch.device,
    desc: str = "val",
) -> dict[str, float]:
    model.eval()
    tot_loss = 0.0
    n = 0
    traj_right = 0
    for x, row_label in tqdm(loader, desc=desc, leave=False, ncols=100):
        x = x.to(device)
        y_mil = trajectory_mil_target(as_row_label(row_label))
        logits = model(x)
        loss = mil_hybrid_loss(logits, y_mil)
        tot_loss += loss.item()
        p = torch.sigmoid(logits)
        if prediction_matches_target(p, y_mil):
            traj_right += 1
        n += 1
    return {
        "loss": tot_loss / max(n, 1),
        "traj_acc": traj_right / max(n, 1),
    }


def train_one_epoch(
    model: InstanceMlp,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    tot = 0.0
    n = 0
    traj_right = 0
    for x, row_label in tqdm(loader, desc="train", leave=False, ncols=100):
        x = x.to(device)
        if x.size(0) == 0:
            continue
        y_mil = trajectory_mil_target(as_row_label(row_label))
        optimizer.zero_grad()
        logits = model(x)
        loss = mil_hybrid_loss(logits, y_mil)
        with torch.no_grad():
            p = torch.sigmoid(logits)
            if prediction_matches_target(p, y_mil):
                traj_right += 1
        loss.backward()
        optimizer.step()
        tot += loss.item()
        n += 1
    denom = max(n, 1)
    return tot / denom, traj_right / denom


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    train_path = resolve_data_path(str(cfg.data.train_jsonl))
    val_path = resolve_data_path(str(cfg.data.val_jsonl))

    train_ds = TrajectoryJsonlDataset(train_path)
    val_ds = TrajectoryJsonlDataset(val_path)
    bs = int(cfg.training.batch_size)
    if bs != 1:
        raise ValueError("training.batch_size must be 1 (one trajectory per step).")
    nw = int(cfg.training.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_one_trajectory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        collate_fn=collate_one_trajectory,
    )

    model = InstanceMlp(
        N_FEATURES,
        int(cfg.model.hidden),
        int(cfg.model.num_hidden_layers),
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.training.lr))

    epochs = int(cfg.training.epochs)
    logger.info(
        "MIL: train {} trajectories from {}, val {} from {}, device={}",
        len(train_ds),
        train_path,
        len(val_ds),
        val_path,
        device,
    )

    for ep in trange(epochs, desc="epoch", ncols=100):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer)
        m_val = run_eval(model, val_loader, device, desc="val")
        logger.info(
            "epoch {}/{}  train loss {:.4f}  train acc {:.4f}  |  "
            "val loss {:.4f}  val acc {:.4f}",
            ep + 1,
            epochs,
            tr_loss,
            tr_acc,
            m_val["loss"],
            m_val["traj_acc"],
        )


if __name__ == "__main__":
    main()
