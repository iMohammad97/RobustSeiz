"""EEGNet seizure classifier: 18-ch bipolar 4 s window at 128 Hz -> seizure probability."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from braindecode.models import EEGNet

from eegnet_seizure.preprocessing import WINDOW

N_CHANS = 18
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


class SeizureEEGNet(torch.nn.Module):
    """braindecode EEGNet (default F1=8, D=2, dropout 0.25) with a sigmoid head."""

    def __init__(self):
        super().__init__()
        self.net = EEGNet(n_chans=N_CHANS, n_outputs=1, n_times=WINDOW)

    def forward(self, x):  # (B, 18, 512) -> (B,)
        return torch.sigmoid(self.net(x)).squeeze(-1)


def save(model: SeizureEEGNet, config: dict, out_dir: Path = WEIGHTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load(device: str = "cpu", weights_dir: Path = WEIGHTS_DIR) -> tuple[SeizureEEGNet, dict]:
    model = SeizureEEGNet()
    model.load_state_dict(torch.load(weights_dir / "model.pt", map_location=device, weights_only=True))
    config = json.loads((weights_dir / "config.json").read_text(encoding="utf-8"))
    return model.to(device).eval(), config
