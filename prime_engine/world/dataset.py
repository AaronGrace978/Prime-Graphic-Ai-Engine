"""Trajectory dataset generation (record layout documented in schema.py)."""
from __future__ import annotations

import dataclasses
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm.auto import tqdm

from .policy import WaypointExplorer
from .raycaster import RoomWorld, WorldConfig
from .schema import (CAPTURE_FPS, SCHEMA_VERSION, action_to_input, camera_intrinsics, cell_state_hash,
                     pose_to_T_world_cam)


def collect_episode(cfg: WorldConfig, length: int, seed: int, episode_id: int = 0) -> dict:
    """One episode with randomly coloured rooms, in the schema.py record layout (length+1 frames)."""
    rng = np.random.default_rng(seed)
    world = RoomWorld(cfg, seed=int(rng.integers(1 << 31)))
    world.reset(random_start=True)
    policy = WaypointExplorer(world, rng)
    T1, H, W = length + 1, cfg.height, cfg.width
    rec = dict(
        rgb=np.empty((T1, H, W, 3), np.uint8), depth=np.empty((T1, H, W), np.float16),
        instance=np.empty((T1, H, W), np.int8), player_pose=np.empty((T1, 3), np.float32),
        T_world_cam=np.empty((T1, 4, 4), np.float32), cell_id=np.empty(T1, np.int8),
        portal_crossed=np.zeros(T1, bool), state_hash=np.empty(T1, np.uint64),
        action=np.empty(length, np.int64), keys=np.empty(length, np.uint8), mouse_dx=np.empty(length, np.float32),
    )
    state_hashes = {c: cell_state_hash(world.world_seed, c, world.cell_state(c)) for c in range(-1, world.n_rooms)}
    last_room = None
    for t in range(T1):
        if t > 0:
            a = policy.act()
            world.step(a)
            rec["action"][t - 1] = a
            rec["keys"][t - 1], rec["mouse_dx"][t - 1] = action_to_input(a, cfg)
        rec["rgb"][t], rec["depth"][t], rec["instance"][t] = world.render(aux=True)
        rec["player_pose"][t] = world.pose
        rec["T_world_cam"][t] = pose_to_T_world_cam(world.pose)
        cell = world.room_at(*world.pose[:2])
        rec["cell_id"][t] = cell
        rec["state_hash"][t] = state_hashes[cell]
        if cell >= 0:
            rec["portal_crossed"][t] = last_room is not None and cell != last_room
            last_room = cell
    rec["frame_id"] = np.arange(T1, dtype=np.int32)
    rec["t"] = (rec["frame_id"] / CAPTURE_FPS).astype(np.float32)
    rec["episode_id"] = np.int64(episode_id)
    rec["world_seed"] = np.int64(world.world_seed)
    rec["cell_state"] = world.room_colors.astype(np.int64)
    return rec


def _collect(args):
    return collect_episode(*args)


def generate_dataset(cfg: WorldConfig, n_episodes: int, length: int, seed: int = 0, workers: int | None = None,
                     out_path: str | None = None) -> dict:
    workers = workers or min(8, os.cpu_count() or 1)
    jobs = [(cfg, length, seed * 100003 + i, i) for i in range(n_episodes)]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            eps = list(tqdm(ex.map(_collect, jobs, chunksize=4), total=n_episodes, desc="collect"))
    else:
        eps = [collect_episode(*j) for j in tqdm(jobs, desc="collect")]
    data = {k: np.stack([e[k] for e in eps]) for k in eps[0]}
    data["camera_K"] = camera_intrinsics(cfg)
    data["schema_version"] = np.int64(SCHEMA_VERSION)
    data["world_config"] = np.array(json.dumps(dataclasses.asdict(cfg)))
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        np.savez_compressed(out_path, **data)
    return data


def load_dataset(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
