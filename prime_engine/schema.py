"""Per-frame capture record from the architecture plan, section 2.3.

Production Solidify hashes use BLAKE3. This prototype uses stdlib BLAKE2s
so the fingerprint has no extra dependency. The field name is ``state_hash``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

RUNNER_ID = "phase0-runner-v0"

# Action bitmask. Mouse axes are stored separately, matching the plan's record.
ACT_FORWARD = 1
ACT_BACK = 2
ACT_TURN_LEFT = 4
ACT_TURN_RIGHT = 8
ACT_STRAFE_LEFT = 16
ACT_STRAFE_RIGHT = 32
ACT_INTERACT = 64


def state_hash(game_state: dict, cell_stats: dict, runner_id: str = RUNNER_ID) -> str:
    """Fingerprint of authoritative state plus the Solidify cache stats."""
    payload = {
        "runner_id": runner_id,
        "game_state": game_state,
        "cells": cell_stats,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2s(raw, digest_size=8).hexdigest()


@dataclass
class FrameRecord:
    frame_id: int
    t: float
    episode_id: str
    world_seed: int
    rgb: np.ndarray  # H W 3 uint8
    depth: np.ndarray  # H W float16, forward depth in meters
    instance_id: np.ndarray  # H W uint8
    K: np.ndarray  # 3x3
    pose: np.ndarray  # x, y, z, yaw, pitch, roll
    player_pose: np.ndarray
    action_keys: int
    mouse_dx: float
    mouse_dy: float
    buttons: int
    cell_id: int
    portal_crossed: bool
    game_state: dict
    state_hash: str
    guide_rgb: np.ndarray  # H W 3 uint8, cache render before this frame was fused
    guide_mask: np.ndarray  # H W uint8

    @property
    def yaw(self) -> float:
        return float(self.pose[3])


def save_episode(path: str, frames: list[FrameRecord]) -> None:
    if not frames:
        raise ValueError("empty episode")
    states = [json.dumps(f.game_state, sort_keys=True) for f in frames]
    np.savez_compressed(
        path,
        frame_id=np.array([f.frame_id for f in frames], np.int32),
        t=np.array([f.t for f in frames], np.float32),
        episode_id=np.array(frames[0].episode_id),
        world_seed=np.array(frames[0].world_seed, np.int64),
        rgb=np.stack([f.rgb for f in frames]),
        depth=np.stack([f.depth for f in frames]),
        instance_id=np.stack([f.instance_id for f in frames]),
        K=frames[0].K.astype(np.float32),
        pose=np.stack([f.pose for f in frames]).astype(np.float32),
        player_pose=np.stack([f.player_pose for f in frames]).astype(np.float32),
        action_keys=np.array([f.action_keys for f in frames], np.int32),
        mouse_dx=np.array([f.mouse_dx for f in frames], np.float32),
        mouse_dy=np.array([f.mouse_dy for f in frames], np.float32),
        buttons=np.array([f.buttons for f in frames], np.int32),
        cell_id=np.array([f.cell_id for f in frames], np.int16),
        portal_crossed=np.array([f.portal_crossed for f in frames], np.bool_),
        game_state=np.array(states),
        state_hash=np.array([f.state_hash for f in frames]),
        guide_rgb=np.stack([f.guide_rgb for f in frames]),
        guide_mask=np.stack([f.guide_mask for f in frames]),
    )


def load_episode(path: str) -> list[FrameRecord]:
    z = np.load(path, allow_pickle=False)
    n = len(z["frame_id"])
    episode_id = str(z["episode_id"])
    world_seed = int(z["world_seed"])
    K = z["K"]
    frames = []
    for i in range(n):
        frames.append(
            FrameRecord(
                frame_id=int(z["frame_id"][i]),
                t=float(z["t"][i]),
                episode_id=episode_id,
                world_seed=world_seed,
                rgb=z["rgb"][i],
                depth=z["depth"][i],
                instance_id=z["instance_id"][i],
                K=K,
                pose=z["pose"][i],
                player_pose=z["player_pose"][i],
                action_keys=int(z["action_keys"][i]),
                mouse_dx=float(z["mouse_dx"][i]),
                mouse_dy=float(z["mouse_dy"][i]),
                buttons=int(z["buttons"][i]),
                cell_id=int(z["cell_id"][i]),
                portal_crossed=bool(z["portal_crossed"][i]),
                game_state=json.loads(str(z["game_state"][i])),
                state_hash=str(z["state_hash"][i]),
                guide_rgb=z["guide_rgb"][i],
                guide_mask=z["guide_mask"][i],
            )
        )
    return frames
