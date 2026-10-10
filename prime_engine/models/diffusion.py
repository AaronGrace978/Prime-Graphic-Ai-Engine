"""Continuous-time cosine diffusion with v-prediction and DDIM sampling."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, min_snr_gamma: float = 5.0, t_min: float = 1e-4, t_max: float = 0.9999):
        self.min_snr_gamma = min_snr_gamma
        self.t_min, self.t_max = t_min, t_max

    @staticmethod
    def alpha_sigma(t: torch.Tensor):
        return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

    def loss(self, model, x0: torch.Tensor, **cond) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device) * (self.t_max - self.t_min) + self.t_min
        a, s = (v[:, None, None, None] for v in self.alpha_sigma(t))
        noise = torch.randn_like(x0)
        x_t = a * x0 + s * noise
        v_target = a * noise - s * x0
        v_pred = model(x_t, t, **cond)
        # Min-SNR weighting expressed for v-prediction: w = min(snr, gamma) / (snr + 1).
        snr = (a / s).pow(2).flatten()
        w = snr.clamp(max=self.min_snr_gamma) / (snr + 1)
        per = F.mse_loss(v_pred.float(), v_target.float(), reduction="none").mean(dim=(1, 2, 3))
        return (per * w).mean()

    @torch.no_grad()
    def sample(self, model, shape, steps: int = 16, generator: torch.Generator | None = None,
               device=None, noise: torch.Tensor | None = None, **cond) -> torch.Tensor:
        x = noise if noise is not None else torch.randn(shape, generator=generator, device=device)
        ts = torch.linspace(self.t_max, 0.0, steps + 1, device=device)
        for i in range(steps):
            t, t_next = ts[i], ts[i + 1]
            a, s = self.alpha_sigma(t)
            v = model(x, t.expand(shape[0]), **cond).float()
            x0 = (a * x - s * v).clamp(-1, 1)
            eps = (x - a * x0) / s
            a_n, s_n = self.alpha_sigma(t_next)
            x = a_n * x0 + s_n * eps
        return x
