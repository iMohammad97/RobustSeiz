import torch
import torch.nn.functional as F


def fgsm_batch(model, batch, mode="flip", eps=0.02):
    model.eval()
    x = batch.clone().detach().requires_grad_(True)
    out = model(x)
    if mode == "increase":
        target = torch.ones_like(out)
    elif mode == "decrease":
        target = torch.zeros_like(out)
    elif mode == "flip":
        target = (out.detach() < 0.5).float()
    else:
        raise ValueError(f"ATTACK_MODE must be increase|decrease|flip, got {mode}")
    loss = F.binary_cross_entropy(out, target)
    loss.backward()
    return (x + eps * x.grad.sign()).detach()


def pgd_batch(model, batch, mode="increase", eps=0.02, alpha=0.005, steps=5):
    model.eval()
    x0 = batch.detach()
    x = x0.clone().detach() + torch.zeros_like(x0).uniform_(-eps, eps)
    x.requires_grad_(True)
    for _ in range(steps):
        out = model(x)
        if mode == "increase":
            target = torch.ones_like(out)
        elif mode == "decrease":
            target = torch.zeros_like(out)
        elif mode == "flip":
            target = (out.detach() < 0.5).float()
        else:
            raise ValueError("increase|decrease|flip")
        loss = F.binary_cross_entropy(out, target)
        loss.backward()
        with torch.no_grad():
            x += alpha * x.grad.sign()
            x.clamp_(x0 - eps, x0 + eps)
        x.requires_grad_(True)
        x.grad.zero_()
    return x.detach()
