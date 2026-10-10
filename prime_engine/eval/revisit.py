"""The Return benchmark: leave a room, come back, and check it rematerialises consistently.

Per world (random room colours):
  bootstrap  K ground-truth frames in room A (the only real frames the model sees)
  A1         360 degree look-around in room A (first visit, generated)
  go         walk to the next room(s)
  away       360 degree look-around in each away room (room A leaves the context window)
  back       walk back to room A
  A2         360 degree look-around in room A again (revisit, generated)

Metrics on A2 frames, each matched to the A1 frame with the closest pose:
  * revisit_psnr_vs_first: PSNR(generated revisit, generated first visit) -- self-consistency
  * revisit_psnr_vs_gt:    PSNR(generated revisit, ground truth)
  * room_id_acc:           fraction of frames whose dominant wall is painted in the room's true colour
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..harness.geometry import view_distance
from ..harness.rollout import WorldModelRunner
from ..harness.solidify import SolidifyConfig
from ..models import Diffusion
from ..world.policy import scripted_actions_to
from ..world.raycaster import NOOP, TURN_RIGHT, RoomWorld, WorldConfig
from .metrics import psnr, room_identity


@dataclass
class RevisitEpisode:
    world: RoomWorld
    actions: np.ndarray      # [T] action leading frame t -> t+1
    poses: np.ndarray        # [T+1, 3]
    cells: np.ndarray        # [T+1]
    phases: list             # [T+1] phase label per frame
    rgb: np.ndarray          # [T+1, H, W, 3] ground truth
    instance: np.ndarray     # [T+1, H, W]


def build_revisit_episode(cfg: WorldConfig, seed: int, context_frames: int = 4, home: int = 0,
                          away: tuple = (1,)) -> RevisitEpisode:
    rng = np.random.default_rng(seed)
    world = RoomWorld(cfg, seed=int(rng.integers(1 << 31)))
    n_turn = int(round(360 / cfg.turn_deg))
    cx, cy = world.room_center(home)
    heading = rng.integers(n_turn) * np.deg2rad(cfg.turn_deg)
    world.reset(pose=(cx + 0.01, cy + 0.01, heading))
    actions, phases = [], ["bootstrap"]
    sim_pose = world.pose.copy()

    def add(acts, label):
        nonlocal sim_pose
        for a in acts:
            actions.append(a)
            phases.append(label)
            sim_pose = world.next_pose(sim_pose, a)

    add([TURN_RIGHT] * (context_frames - 1), "bootstrap")
    add([TURN_RIGHT] * n_turn, "A1")
    for r in away:
        add(scripted_actions_to(world, sim_pose, world.room_center(r)), "go")
        add([TURN_RIGHT] * n_turn, "away")
    add(scripted_actions_to(world, sim_pose, world.room_center(home)), "back")
    add([TURN_RIGHT] * n_turn, "A2")

    poses, rgb, inst = [world.pose.copy()], [], []
    for a in actions:
        poses.append(world.step(a))
    poses = np.stack(poses)
    for p in poses:
        f, _, i = world.render(p, aux=True)
        rgb.append(f)
        inst.append(i)
    cells = np.array([world.room_at(p[0], p[1]) for p in poses])
    return RevisitEpisode(world, np.array(actions), poses, cells, phases, np.stack(rgb), np.stack(inst))


@torch.no_grad()
def rollout_episodes(model, episodes: list[RevisitEpisode], memory_cfg: SolidifyConfig | None, *,
                     sample_steps: int = 12, aug_level: float = 0.05, device="cpu", seed: int = 0,
                     use_cells: bool = True):
    """Generate every frame after the bootstrap. Returns (frames [B, T+1, H, W, 3], runner)."""
    K = model.K
    B, T1 = len(episodes), len(episodes[0].poses)
    assert all(len(e.poses) == T1 for e in episodes), "batch episodes must share a length"
    runner = WorldModelRunner(model, Diffusion(), batch=B, sample_steps=sample_steps, aug_level=aug_level,
                              memory_cfg=memory_cfg, device=device, seed=seed)
    boot_actions = np.stack([np.concatenate([[NOOP], e.actions[:K - 1]]) for e in episodes])
    runner.reset(np.stack([e.rgb[:K] for e in episodes]), boot_actions, np.stack([e.poses[:K] for e in episodes]),
                 cell_ids=np.stack([e.cells[:K] for e in episodes]) if use_cells else None,
                 cell_graphs=[e.world.cell_graph() for e in episodes] if use_cells else None,
                 world_seeds=[e.world.world_seed for e in episodes])
    out = np.zeros((B, T1) + episodes[0].rgb.shape[1:], np.uint8)
    out[:, :K] = np.stack([e.rgb[:K] for e in episodes])
    for t in range(K, T1):
        out[:, t] = runner.step(np.array([e.actions[t - 1] for e in episodes]),
                                np.stack([e.poses[t] for e in episodes]),
                                np.array([e.cells[t] for e in episodes]) if use_cells else None)
    return out, runner


def revisit_metrics(ep: RevisitEpisode, gen: np.ndarray) -> dict:
    ph = np.array(ep.phases)
    a1, a2 = np.where(ph == "A1")[0], np.where(ph == "A2")[0]
    vs_first, vs_gt, ident_rev, ident_first = [], [], [], []
    for t in a2:
        m = a1[np.argmin(view_distance(ep.poses[t], ep.poses[a1]))]
        vs_first.append(psnr(gen[t], gen[m]))
        vs_gt.append(psnr(gen[t], ep.rgb[t]))
        r = room_identity(gen[t], ep.instance[t], ep.world.room_colors)
        if r is not None:
            ident_rev.append(r[0] == r[1])
    for t in a1:
        r = room_identity(gen[t], ep.instance[t], ep.world.room_colors)
        if r is not None:
            ident_first.append(r[0] == r[1])
    return {
        "revisit_psnr_vs_first": float(np.mean(vs_first)),
        "revisit_psnr_vs_gt": float(np.mean(vs_gt)),
        "room_id_acc_revisit": float(np.mean(ident_rev)) if ident_rev else float("nan"),
        "room_id_acc_first_visit": float(np.mean(ident_first)) if ident_first else float("nan"),
        "first_visit_psnr_vs_gt": float(np.mean([psnr(gen[t], ep.rgb[t]) for t in a1])),
    }


def run_revisit_benchmark(model, world_cfg: WorldConfig, harness_cfg: SolidifyConfig, *, n_worlds: int = 16,
                          sample_steps: int = 12, aug_level: float = 0.05, seed: int = 1234, device="cpu",
                          away: tuple = (1,), extra_modes: dict | None = None) -> dict:
    """Runs the same worlds with the harness off and on; returns per-mode metrics and frames."""
    episodes = [build_revisit_episode(world_cfg, seed + i, model.K, away=away) for i in range(n_worlds)]
    # Episodes can differ in length by a few steps (path following); trim all to the shortest.
    T1 = min(len(e.poses) for e in episodes)
    for e in episodes:
        cut = len(e.poses) - T1
        if cut:  # drop frames from the end of the revisit look-around
            e.poses, e.cells, e.rgb, e.instance = e.poses[:T1], e.cells[:T1], e.rgb[:T1], e.instance[:T1]
            e.actions, e.phases = e.actions[:T1 - 1], e.phases[:T1]
    modes = {"no_harness": None, "solidify": harness_cfg, **(extra_modes or {})}
    results = {"episodes": episodes, "modes": {}}
    for name, mcfg in modes.items():
        frames, runner = rollout_episodes(model, episodes, mcfg, sample_steps=sample_steps, aug_level=aug_level,
                                          device=device, seed=seed)
        per = [revisit_metrics(e, frames[i]) for i, e in enumerate(episodes)]
        agg = {k: float(np.nanmean([p[k] for p in per])) for k in per[0]}
        cells = runner.stores[0].summary() if runner.stores else []
        results["modes"][name] = {"metrics": agg, "per_world": per, "frames": frames, "cells_world0": cells}
    return results
