"""Batch sampler for next-frame training with pose-retrieved memory views."""
from __future__ import annotations

import numpy as np
import torch

from ..harness.geometry import POSE_FEAT_DIM, relative_pose_features, view_distance


def precompute_retrieval(poses: np.ndarray, K: int, M: int, angle_weight: float, max_distance: float):
    """For every (episode, target i): indices of the M closest past views outside the context window.

    Mirrors SpatialMemory.retrieve: candidates j < i - K, ranked by view_distance to pose[i].
    Returns idx [N, T1, M] (int64) and pose features [N, T1, M, P] (valid flag = 0 when missing).
    """
    N, T1, _ = poses.shape
    idx = np.zeros((N, T1, M), np.int64)
    feats = np.zeros((N, T1, M, POSE_FEAT_DIM), np.float32)
    if M == 0:
        return idx, feats
    j = np.arange(T1)
    for n in range(N):
        P = poses[n].astype(np.float64)
        D = np.stack([view_distance(P[i], P, angle_weight) for i in range(T1)])  # [T1, T1]
        D[j[None, :] >= (j[:, None] - K)] = np.inf
        order = np.argsort(D, axis=1)[:, :M]
        dsel = np.take_along_axis(D, order, axis=1)
        idx[n] = order
        for i in range(T1):
            f = relative_pose_features(P[i], P[order[i]])
            f[dsel[i] > max_distance] = 0.0
            feats[n, i] = f
    return idx, feats


class TrajectoryBatcher:
    def __init__(self, data: dict, K: int, M: int, angle_weight: float, max_distance: float, device="cpu"):
        frames = torch.from_numpy(data["rgb"])  # [N, T1, H, W, 3] uint8
        self.frames = frames.permute(0, 1, 4, 2, 3).contiguous()  # [N, T1, 3, H, W]
        self.actions = torch.from_numpy(data["action"]).long()  # [N, T]
        self.N, self.T1 = self.frames.shape[:2]
        self.K, self.M = K, M
        idx, feats = precompute_retrieval(data["player_pose"], K, M, angle_weight, max_distance)
        self.mem_idx = torch.from_numpy(idx)
        self.mem_feat = torch.from_numpy(feats)
        self.device = torch.device(device)
        valid = (self.mem_feat[..., -1] > 0).float()
        self.memory_coverage = valid[:, K:].mean().item()

    def sample(self, batch: int, gen: torch.Generator, max_aug: float, mem_drop: float, slot_drop: float) -> dict:
        n = torch.randint(0, self.N, (batch,), generator=gen)
        i = torch.randint(self.K, self.T1, (batch,), generator=gen)
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

        d = self.device
        to_f = lambda x: x.to(d, non_blocking=True).float() / 127.5 - 1.0  # noqa: E731
        ctx, mem, target = to_f(ctx), to_f(mem), to_f(target)
        mpose, level = mpose.to(d), level.to(d)
        valid = mpose[..., -1, None, None, None]
        lv = level[:, None, None, None, None]
        ctx = ctx + lv * torch.randn_like(ctx)
        mem = (mem + lv * torch.randn_like(mem)) * valid
        return dict(target=target, context=ctx, actions=acts.to(d), aug_level=level, mem=mem, mem_pose=mpose)
