from __future__ import annotations

import os

import torch

from experiments_utils.adversarial import fgsm_batch, pgd_batch


def forward_with_adversarial(model, dataloader, device):
    """Run model forward pass with optional FGSM/PGD perturbations from env vars."""
    attack = os.getenv("ATTACK", "").lower()
    eps = float(os.getenv("EPS", "0") or 0)
    attack_mode = os.getenv("ATTACK_MODE", "flip").lower()
    thresh = float(os.getenv("ATTACK_THRESH", "0.5"))

    total_output = None
    model.eval()
    for data in dataloader:
        data = data.float().to(device)

        if attack == "fgsm" and eps > 0:
            batch = fgsm_batch(model, data, mode=attack_mode, eps=eps, thresh=thresh)
        elif attack == "pgd" and eps > 0:
            alpha = float(os.getenv("ALPHA", "0.005"))
            steps = int(os.getenv("STEPS", "5"))
            batch = pgd_batch(
                model, data, mode=attack_mode, eps=eps, alpha=alpha, steps=steps, thresh=thresh
            )
        else:
            batch = data

        with torch.no_grad():
            output = model(batch)
        total_output = output if total_output is None else torch.cat((total_output, output), 0)

    print(f"[adversarial] attack={attack or 'none'} mode={attack_mode} eps={eps}")
    return total_output
