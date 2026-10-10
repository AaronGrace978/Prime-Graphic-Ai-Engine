"""Batch sampler for next-frame training with memory views retrieved like the runtime store."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..harness.geometry import POSE_FEAT_DIM, pose_key, relative_pose_features, view_distance
from ..harness.solidify import SolidifyConfig


def precompute_retrieval(poses: np.ndarray, K: int, M: int, hcfg: SolidifyConfig,
                         vis_fn: Callable | None = None):
    """For every (episode, target i): the M memory views the runtime store would return.

    Mirrors SolidifyStore: only *canonical* views are candidates (first frame to land in each
    pose key, i.e. first-write-wins), they must be older than the context window (j < i - K),
    and they are ranked by FOV overlap (``retrieval="fov"``, needs vis_fn) or pose distance.
    Returns idx [N, T1, M] and pose features [N, T1, M, P] (valid flag = 0 when missing).
    """
    N, T1, _ = poses.shape
    idx = np.zeros((N, T1, M), np.int64)
    feats = np.zeros((N, T1, M, POSE_FEAT_DIM), np.float32)
    if M == 0:
        return idx, feats
    use_fov = hcfg.retrieval == "fov" and vis_fn is not None
    j = np.arange(T1)
    for n in range(N):
        P = poses[n].astype(np.float64)
        seen, canon = set(), np.zeros(T1, bool)
        for t in range(T1):
            k = pose_key(P[t], hcfg.pos_cell, hcfg.heading_bins)
            canon[t] = k not in seen
            seen.add(k)
        if use_fov:
            V = np.stack([vis_fn(p) for p in P])
            score = V @ V.T
            ok = score >= hcfg.min_overlap
        else:
            dist = np.stack([view_distance(P[i], P, hcfg.angle_weight) for i in range(T1)])
            score, ok = -dist, dist <= hcfg.max_distance
        score = np.where((j[None, :] < (j[:, None] - K)) & canon[None, :], score, -np.inf)
        order = np.argsort(-score, axis=1, kind="stable")[:, :M]
        valid = np.take_along_axis(ok & np.isfinite(score), order, axis=1)
        idx[n] = order
        for i in range(T1):
            f = relative_pose_features(P[i], P[order[i]])
            f[~valid[i]] = 0.0
            feats[n, i] = f
    return idx, feats


def hue_rotate(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate colours of x [B, ..., 3, H, W] in [-1, 1] about the grey axis by angle [B] (radians)."""
    k = torch.full((3,), 3 ** -0.5, device=x.device)
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=x.device)
    c, s = torch.cos(angle)[:, None, None], torch.sin(angle)[:, None, None]
    R = c * torch.eye(3, device=x.device) + s * K + (1 - c) * torch.outer(k, k)  # Rodrigues, [B, 3, 3]
    shape = x.shape
    flat = x.reshape(shape[0], -1, 3, shape[-2] * shape[-1])
    return torch.einsum("bij,bnjp->bnip", R, flat).reshape(shape).clamp(-1, 1)


class TrajectoryBatcher:
    def __init__(self, data: dict, K: int, M: int, hcfg: SolidifyConfig, vis_fn: Callable | None = None,
                 device="cpu", revisit_age: int = 20):
        frames = torch.from_numpy(data["rgb"])  # [N, T1, H, W, 3] uint8
        self.frames = frames.permute(0, 1, 4, 2, 3).contiguous()  # [N, T1, 3, H, W]
        self.actions = torch.from_numpy(data["action"]).long()  # [N, T]
        self.N, self.T1 = self.frames.shape[:2]
        self.K, self.M = K, M
        idx, feats = precompute_retrieval(data["player_pose"], K, M, hcfg, vis_fn)
        self.mem_idx = torch.from_numpy(idx)
        self.mem_feat = torch.from_numpy(feats)
        self.device = torch.device(device)
        valid = self.mem_feat[..., 0, -1] > 0
        self.memory_coverage = valid[:, K:].float().mean().item()
        # "Revisit" targets: best memory view is old, i.e. the place was seen long before the context window.
        age = torch.arange(self.T1)[None, :] - self.mem_idx[..., 0]
        rev = valid & (age >= revisit_age)
        rev[:, :K] = False
        self.revisit_index = rev.nonzero()  # [R, 2] (episode, frame)
        self.revisit_share = rev[:, K:].float().mean().item()

    def sample(self, batch: int, gen: torch.Generator, max_aug: float, mem_drop: float, slot_drop: float,
               revisit_frac: float = 0.0, ctx_corrupt: float = 0.0) -> dict:
        """ctx_corrupt: prob. of hue-rotating the context (only when memory is present) so the model
        learns to take appearance from Solidify memory when its own recent frames have drifted."""
        n = torch.randint(0, self.N, (batch,), generator=gen)
        i = torch.randint(self.K, self.T1, (batch,), generator=gen)
        n_rev = int(round(batch * revisit_frac)) if len(self.revisit_index) else 0
        if n_rev:
            pick = self.revisit_index[torch.randint(0, len(self.revisit_index), (n_rev,), generator=gen)]
            n[:n_rev], i[:n_rev] = pick[:, 0], pick[:, 1]
        ctx_t = i[:, None] - self.K + torch.arange(self.K)[None]          # frames i-K .. i-1
        ctx = self.frames[n[:, None], ctx_t]
        acts = self.actions[n[:, None], ctx_t]                             # action j leads frame j -> j+1
        target = self.frames[n, i]
        midx = self.mem_idx[n, i]
        mem = self.frames[n[:, None], midx]
        mpose = self.mem_feat[n, i].clone()
        keep = (torch.rand(batch, 1, generator=gen) >= mem_drop) & (torch.rand(batch, self.M, generator=gen) >= slot_drop)
        mpose = mpose * keep[..., None].float()
        level = torch.rand(batch, generator=gen) * max_aug
        corrupt = (torch.rand(batch, generator=gen) < ctx_corrupt) & (mpose[..., -1].sum(1) > 0)
        angle = (torch.rand(batch, generator=gen) * 4 + 1) * (torch.pi / 3)  # 60..300 degrees

        d = self.device
        to_f = lambda x: x.to(d, non_blocking=True).float() / 127.5 - 1.0  # noqa: E731
        ctx, mem, target = to_f(ctx), to_f(mem), to_f(target)
        if corrupt.any():
            ci = corrupt.nonzero().flatten().to(d)
            ctx[ci] = hue_rotate(ctx[ci], angle.to(d)[ci])
        mpose, level = mpose.to(d), level.to(d)
        valid = mpose[..., -1, None, None, None]
        lv = level[:, None, None, None, None]
        ctx = ctx + lv * torch.randn_like(ctx)
        mem = (mem + lv * torch.randn_like(mem)) * valid
        return dict(target=target, context=ctx, actions=acts.to(d), aug_level=level, mem=mem, mem_pose=mpose)
