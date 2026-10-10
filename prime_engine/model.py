"""Tiny action-conditioned diffusion next-frame model.

The network is a Phase 0 stand-in for the Runner: a few million parameters,
trained from scratch on frames from the house. It is conditioned on the
previous frame, the action, and an optional Solidify guide render.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(ch: int) -> nn.GroupNorm:
    groups = 4 if ch % 4 == 0 else 1
    return nn.GroupNorm(groups, ch)


class ResBlock(nn.Module):
    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = _group_norm(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = _group_norm(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.to_scale = nn.Linear(emb_dim, ch)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.to_scale(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class TinyUNet(nn.Module):
    """Concatenates noisy RGB, context RGB, guide RGB, and a coverage mask."""

    def __init__(self, emb_dim: int = 128, base: int = 32):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(32, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.act_emb = nn.Embedding(128, emb_dim)
        self.in_conv = nn.Conv2d(10, base, 3, padding=1)
        self.down1 = nn.Conv2d(base, base * 2, 4, stride=2, padding=1)
        self.b1 = ResBlock(base * 2, emb_dim)
        self.down2 = nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1)
        self.b2 = ResBlock(base * 4, emb_dim)
        self.mid = ResBlock(base * 4, emb_dim)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1)
        self.b3 = ResBlock(base * 2, emb_dim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1)
        self.b4 = ResBlock(base, emb_dim)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(sinusoidal(t, 32)) + self.act_emb(action.clamp(0, 127))
        h0 = self.in_conv(x)
        h1 = self.b1(self.down1(h0), emb)
        h2 = self.b2(self.down2(h1), emb)
        h = self.mid(h2, emb)
        h = self.b3(self.up2(h) + h1, emb)
        h = self.b4(self.up1(h) + h0, emb)
        return self.out(h)


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Schedule:
    def __init__(self, steps: int = 64, device: torch.device | None = None):
        self.steps = steps
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(alphas, dim=0).to(device or "cpu")

    def to(self, device: torch.device) -> "Schedule":
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def pack(noisy, context, guide, mask) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    return torch.cat([noisy, context, guide, mask], dim=1)


def train_model(frames, steps: int = 80, batch: int = 8, lr: float = 1e-3, seed: int = 0, device: str | None = None, upto: int | None = None):
    """Train on frames[:upto]. Each target is paired with the previous frame and its guide."""
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    last = upto if upto is not None else len(frames)
    last = max(2, min(last, len(frames)))
    rgb = np.stack([f.rgb for f in frames[:last]]).astype(np.float32) / 255.0
    guide = np.stack([f.guide_rgb for f in frames[:last]]).astype(np.float32) / 255.0
    mask = np.stack([f.guide_mask for f in frames[:last]]).astype(np.float32) / 255.0
    actions = np.array([f.action_keys for f in frames[:last]], np.int64)
    n, h, w, _ = rgb.shape
    model = TinyUNet().to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = Schedule(device=device_t)
    model.train()
    losses = []
    for step in range(steps):
        idx = rng.integers(1, n, size=batch)
        # Drop the guide on some samples so the net can also roll without a cache.
        use_guide = rng.random(batch) > 0.35
        gt = torch.from_numpy(rgb[idx]).permute(0, 3, 1, 2).to(device_t)
        ctx = torch.from_numpy(rgb[idx - 1]).permute(0, 3, 1, 2).to(device_t)
        gd = torch.from_numpy(guide[idx]).permute(0, 3, 1, 2).to(device_t)
        mk = torch.from_numpy(mask[idx])[:, None].to(device_t)
        gd = gd * torch.from_numpy(use_guide.astype(np.float32)).to(device_t)[:, None, None, None]
        mk = mk * torch.from_numpy(use_guide.astype(np.float32)).to(device_t)[:, None, None, None]
        t = torch.randint(0, sched.steps, (batch,), device=device_t)
        noise = torch.randn_like(gt)
        ab = sched.alpha_bar[t][:, None, None, None]
        xt = torch.sqrt(ab) * gt + torch.sqrt(1.0 - ab) * noise
        pred = model(pack(xt, ctx, gd, mk), t, torch.from_numpy(actions[idx]).to(device_t))
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return model, sched, float(np.mean(losses[-10:])), sum(p.numel() for p in model.parameters())


@torch.no_grad()
def sample(model, sched, context, guide, mask, action, start_t: int, device) -> np.ndarray:
    """DDIM from ``start_t``. ``start_t`` near the end of the schedule is pure noise.

    A mid-schedule start is SDEdit from the guide, which is how Solidify asks the
    model to refine the cache instead of inventing the room.
    """
    model.eval()
    ctx = _image_tensor(context, device)
    gd = _image_tensor(guide, device)
    mk = _mask_tensor(mask, device)
    if start_t >= sched.steps - 1:
        x = torch.randn(1, 3, ctx.shape[2], ctx.shape[3], device=device)
        times = [sched.steps - 1, 48, 32, 16, 8, 0]
    else:
        ab = sched.alpha_bar[start_t]
        noise = torch.randn_like(ctx)
        x = torch.sqrt(ab) * gd + torch.sqrt(1.0 - ab) * noise
        times = [t for t in (start_t, start_t // 2, start_t // 4, 0) if t <= start_t]
        if times[-1] != 0:
            times.append(0)
    act = torch.tensor([action], dtype=torch.long, device=device)
    for i, t_from in enumerate(times[:-1]):
        t_to = times[i + 1]
        t = torch.tensor([t_from], dtype=torch.long, device=device)
        eps = model(pack(x, ctx, gd, mk), t, act)
        ab_f = sched.alpha_bar[t_from]
        x0 = (x - torch.sqrt(1.0 - ab_f) * eps) / torch.sqrt(ab_f)
        x0 = x0.clamp(-1.0, 1.5)
        if t_to == 0:
            x = x0
        else:
            ab_t = sched.alpha_bar[t_to]
            x = torch.sqrt(ab_t) * x0 + torch.sqrt(1.0 - ab_t) * eps
    img = x[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def consistency_guard(pred: np.ndarray, guide: np.ndarray, mask: np.ndarray, thresh: float = 0.12) -> np.ndarray:
    """Where the cache covers the pixel and the model disagrees, snap back to the cache."""
    err = np.mean(np.abs(pred.astype(np.float32) - guide.astype(np.float32)), axis=-1) / 255.0
    snap = (mask > 127) & (err > thresh)
    out = pred.copy()
    out[snap] = guide[snap]
    return out


def _image_tensor(img: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return t.to(device)


def _mask_tensor(mask: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(mask.astype(np.float32) / 255.0)[None, None]
    return t.to(device)
