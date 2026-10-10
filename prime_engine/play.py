"""Interactive session: the engine (raycaster logic) drives pose, the model renders.

Runs two neural renderers in lockstep -- harness off and Solidify on -- next to
the ground-truth raycast, so you can walk out of a room and back and watch the
difference live.
"""
from __future__ import annotations

import numpy as np

from .harness import WorldModelRunner
from .models import Diffusion
from .world.raycaster import NOOP, TURN_RIGHT, RoomWorld


class PlaySession:
    def __init__(self, model, cfg, seed: int = 0, device="cpu", sample_steps: int | None = None):
        self.cfg = cfg
        self.world = RoomWorld(cfg.world, seed=seed)
        cx, cy = self.world.room_center(0)
        self.world.reset(pose=(cx, cy, 0.0))
        steps = sample_steps or cfg.eval.sample_steps
        kw = dict(batch=1, sample_steps=steps, aug_level=cfg.eval.aug_level, device=device, seed=seed)
        self.runners = {"no_harness": WorldModelRunner(model, Diffusion(), memory_cfg=None, **kw),
                        "solidify": WorldModelRunner(model, Diffusion(), memory_cfg=cfg.harness, **kw)}
        K = model.K
        frames, poses, cells, acts = [], [], [], [NOOP]
        for j in range(K):
            if j:
                self.world.step(TURN_RIGHT)
                acts.append(TURN_RIGHT)
            frames.append(self.world.render())
            poses.append(self.world.pose.copy())
            cells.append(self.world.room_at(*self.world.pose[:2]))
        vis = np.stack([self.world.visibility(p) for p in poses])[None]
        for r in self.runners.values():
            r.reset(np.stack(frames)[None], np.array(acts)[None], np.stack(poses)[None], cell_ids=np.array(cells)[None],
                    cell_graphs=[self.world.cell_graph()], world_seeds=[self.world.world_seed], visibility=vis)
        self.last = {"gt": frames[-1], "no_harness": frames[-1], "solidify": frames[-1]}
        self.trail = [p for p in poses]

    def step(self, action: int) -> dict:
        pose = self.world.step(action)
        cell = self.world.room_at(*pose[:2])
        self.trail.append(pose.copy())
        self.last = {"gt": self.world.render()}
        vis = self.world.visibility(pose)[None]
        for name, r in self.runners.items():
            self.last[name] = r.step([action], pose[None], np.array([cell]), visibility=vis)[0]
        return self.last

    def cells(self) -> list[dict]:
        return self.runners["solidify"].stores[0].summary()
