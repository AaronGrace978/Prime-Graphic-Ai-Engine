"""Autoregressive neural rendering loop with optional Solidify memory.

Per engine plan §3.3: the engine owns state (here: the raycaster's kinematics,
collision and cell graph) and hands the runner the *pose* and *cell id* each
frame must be rendered from -- pose never comes from the model. The runner
renders that frame with the diffusion model, conditioned on recent frames, the
action and, if enabled, keyframes retrieved from the Solidify store.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .geometry import POSE_FEAT_DIM
from .solidify import SolidifyConfig, SolidifyStore


def to_tensor(frames_uint8: np.ndarray) -> torch.Tensor:
    """[..., H, W, 3] uint8 -> [..., 3, H, W] float in [-1, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(frames_uint8)).float() / 127.5 - 1.0
    return t.movedim(-1, -3)


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    return ((frames.movedim(-3, -1).clamp(-1, 1) + 1) * 127.5 + 0.5).to(torch.uint8).cpu().numpy()


class WorldModelRunner:
    """Runs B worlds in lockstep. memory_cfg=None disables the Solidify harness."""

    def __init__(self, model, diffusion, *, batch: int = 1, sample_steps: int = 16, aug_level: float = 0.05,
                 memory_cfg: SolidifyConfig | None = None, device="cpu", seed: int = 0, runner_id: str = "v0"):
        self.model, self.diffusion = model.eval(), diffusion
        self.K, self.M = model.K, model.M
        self.B = batch
        self.sample_steps, self.aug_level = sample_steps, aug_level
        self.device = torch.device(device)
        self.memory_cfg = memory_cfg
        self.use_memory = memory_cfg is not None and self.M > 0
        self.runner_id = runner_id
        self.stores: list[SolidifyStore] | None = None
        self.seed = seed

    def reset(self, frames: np.ndarray, actions: np.ndarray, poses: np.ndarray, cell_ids=None,
              cell_graphs: list[dict] | None = None, world_seeds=None, visibility=None):
        """Bootstrap with K real frames per world.

        frames [B, K, H, W, 3] uint8, actions [B, K] (actions[:, j] led *into* frames[:, j]; use 0 for the
        first), poses [B, K, 3], cell_ids [B, K] (engine cell per frame, -1 = doorway; None = Dream-mode grid),
        cell_graphs: per-world {cell: {bounds, neighbors, state}} to register (Anchored mode),
        visibility [B, K, D]: engine visibility vectors for FOV-overlap retrieval (None = pose retrieval).
        """
        self.gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.ctx = deque(to_tensor(frames).to(self.device).unbind(1), maxlen=self.K)
        self.acts = deque(torch.as_tensor(actions, dtype=torch.long, device=self.device).unbind(1), maxlen=self.K)
        self.step_idx = frames.shape[1]
        if self.use_memory:
            seeds = world_seeds if world_seeds is not None else [0] * self.B
            self.stores = [SolidifyStore(self.memory_cfg, world_seed=int(s), runner_id=self.runner_id) for s in seeds]
            for b, store in enumerate(self.stores):
                if cell_graphs is not None:
                    store.register_cells(cell_graphs[b])
                for j in range(frames.shape[1]):
                    store.write(poses[b, j], self.ctx[j][b], step=j,
                                cell_id=None if cell_ids is None else int(cell_ids[b][j]),
                                vis=None if visibility is None else visibility[b][j])

    def _memory_inputs(self, next_poses: np.ndarray, cell_ids, visibility):
        h, w = self.ctx[0].shape[-2:]
        mem = torch.zeros(self.B, self.M, 3, h, w, device=self.device)
        feats = np.zeros((self.B, self.M, POSE_FEAT_DIM), np.float32)
        if self.use_memory:
            for b, store in enumerate(self.stores):
                cid = None if cell_ids is None else int(cell_ids[b])
                payloads, f = store.conditioning(next_poses[b], self.M, current_step=self.step_idx, cell_id=cid,
                                                 vis=None if visibility is None else visibility[b])
                feats[b] = f
                for m, p in enumerate(payloads):
                    if p is not None:
                        mem[b, m] = p
        return mem, torch.from_numpy(feats).to(self.device)

    def _initial_noise(self, shape, next_poses, cell_ids):
        if not (self.use_memory and self.memory_cfg.pose_seeded_noise):
            return torch.randn(shape, generator=self.gen, device=self.device)
        out = []
        for b, store in enumerate(self.stores):
            g = torch.Generator(device=self.device).manual_seed(
                store.noise_seed(next_poses[b], None if cell_ids is None else int(cell_ids[b])))
            out.append(torch.randn(shape[1:], generator=g, device=self.device))
        return torch.stack(out)

    @torch.no_grad()
    def step(self, actions, next_poses, cell_ids=None, visibility=None) -> np.ndarray:
        """Advance every world by one action. next_poses [B, 3], cell_ids [B], visibility [B, D] come from the engine.

        Returns the rendered frames [B, H, W, 3] uint8.
        """
        actions = torch.as_tensor(np.asarray(actions).reshape(self.B), dtype=torch.long, device=self.device)
        next_poses = np.asarray(next_poses, dtype=np.float64).reshape(self.B, 3)
        self.acts.append(actions)
        ctx = torch.stack(list(self.ctx), 1)
        mem, mem_pose = self._memory_inputs(next_poses, cell_ids, visibility)
        level = torch.full((self.B,), self.aug_level, device=self.device)
        if self.aug_level > 0:
            ctx = ctx + self.aug_level * torch.randn(ctx.shape, generator=self.gen, device=self.device)
            mem = mem + self.aug_level * torch.randn(mem.shape, generator=self.gen, device=self.device) * (
                mem_pose[..., -1, None, None, None])
        h, w = ctx.shape[-2:]
        noise = self._initial_noise((self.B, 3, h, w), next_poses, cell_ids)
        frame = self.diffusion.sample(self.model, (self.B, 3, h, w), steps=self.sample_steps, device=self.device,
                                      noise=noise, context=ctx, actions=torch.stack(list(self.acts), 1),
                                      aug_level=level, mem=mem, mem_pose=mem_pose)
        self.ctx.append(frame)
        if self.use_memory:
            for b, store in enumerate(self.stores):
                store.write(next_poses[b], frame[b], step=self.step_idx,
                            cell_id=None if cell_ids is None else int(cell_ids[b]),
                            vis=None if visibility is None else visibility[b])
        self.step_idx += 1
        return to_uint8(frame)
