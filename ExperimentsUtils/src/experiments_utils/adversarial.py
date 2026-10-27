import torch
import torch.nn.functional as F


def _attack_target(out, mode, thresh=0.5):
    if mode == "increase":
        return torch.ones_like(out)
    if mode == "decrease":
        return torch.zeros_like(out)
    if mode == "flip":
        return (out.detach() < thresh).float()
    raise ValueError(f"ATTACK_MODE must be increase|decrease|flip, got {mode}")


def _channel_scale(x):
    """Per-window, per-channel std so eps is relative to the input scale of any model."""
    return x.detach().std(dim=-1, keepdim=True).clamp_min(1e-6)


def fgsm_batch(model, batch, mode="flip", eps=0.02, thresh=0.5):
    """One signed-gradient step that moves predictions toward the attack target."""
    model.eval()
    x = batch.clone().detach().requires_grad_(True)
    out = model(x)
    loss = F.binary_cross_entropy(out, _attack_target(out, mode, thresh))
    loss.backward()
    return (x - eps * _channel_scale(batch) * x.grad.sign()).detach()


def pgd_batch(model, batch, mode="increase", eps=0.02, alpha=0.005, steps=5, thresh=0.5):
    """Iterated targeted signed-gradient steps, projected to an eps * channel-std box."""
    model.eval()
    x0 = batch.detach()
    scale = _channel_scale(x0)
    x = x0.clone() + torch.empty_like(x0).uniform_(-1, 1) * eps * scale
    x.requires_grad_(True)
    for _ in range(steps):
        out = model(x)
        loss = F.binary_cross_entropy(out, _attack_target(out, mode, thresh))
        loss.backward()
        with torch.no_grad():
            x -= alpha * scale * x.grad.sign()
            x.copy_(torch.max(torch.min(x, x0 + eps * scale), x0 - eps * scale))
        x.grad.zero_()
    return x.detach()
