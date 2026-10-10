"""Capture-record schema, aligned with the engine plan §2.3 (capture harness).

Per-frame record in the plan vs. this v0 dataset (``.npz``, one array per field,
leading dims [episode, frame]; per-transition fields have one fewer frame):

  plan field                        v0 key            dtype / shape
  --------------------------------  ----------------  -----------------------------------
  frame_id, t                       frame_id, t       int32 [N,T1], float32 [N,T1] (t = frame / FPS)
  episode_id, world_seed            episode_id,       int64 [N]
                                    world_seed
  rgb                               rgb               uint8 [N,T1,H,W,3]
  depth (linear float16)            depth             float16 [N,T1,H,W]  (z along optical axis)
  instance_id map                   instance          int8 [N,T1,H,W]  (room id | -1 door | -2 floor | -3 ceil)
  camera K                          camera_K          float32 [3,3]  (shared)
  camera pose T_world_cam (6DoF)    T_world_cam       float32 [N,T1,4,4]  (OpenCV camera axes)
  player pose                       player_pose       float32 [N,T1,3]  (x, y, heading)
  action: key bitmask, mouse dx     keys, mouse_dx    uint8 [N,T], float32 [N,T]  (input that led t -> t+1)
                                    action            int64 [N,T]  (discrete id the v0 model consumes)
  cell_id, portal crossed?          cell_id,          int8 [N,T1] (-1 = in doorway), bool [N,T1]
                                    portal_crossed
  game_state record + state_hash    cell_state,       int64 [N,n_cells] (wall colour per cell),
                                    state_hash        uint64 [N,T1] (hash of the current cell's record)

Not yet captured (v0 simplifications): normals, gamepad axes, PNG storage,
state *changes* (doors/props) -- the raycaster has static rooms.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .raycaster import (BACKWARD, CAMERA_HEIGHT, FORWARD, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT, TURN_RIGHT,
                        WorldConfig)

CAPTURE_FPS = 30.0
SCHEMA_VERSION = 1

KEY_W, KEY_S, KEY_A, KEY_D = 1, 2, 4, 8
_ACTION_KEYS = {FORWARD: KEY_W, BACKWARD: KEY_S, STRAFE_LEFT: KEY_A, STRAFE_RIGHT: KEY_D}


def action_to_input(action: int, cfg: WorldConfig) -> tuple[int, float]:
    """Discrete action -> (key bitmask, mouse dx in degrees; positive = turn right)."""
    mouse = {TURN_LEFT: -cfg.turn_deg, TURN_RIGHT: cfg.turn_deg}.get(int(action), 0.0)
    return _ACTION_KEYS.get(int(action), 0), mouse


def camera_intrinsics(cfg: WorldConfig) -> np.ndarray:
    fx = (cfg.width / 2) / np.tan(np.deg2rad(cfg.fov_deg) / 2)
    return np.array([[fx, 0, cfg.width / 2], [0, fx, cfg.height / 2], [0, 0, 1]], dtype=np.float32)


def pose_to_T_world_cam(pose) -> np.ndarray:
    """(x, y, heading) on the map (z up) -> 4x4 camera-to-world with OpenCV axes (x right, y down, z fwd)."""
    x, y, th = pose
    c, s = np.cos(th), np.sin(th)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = (-s, c, 0.0)   # camera right
    T[:3, 1] = (0.0, 0.0, -1.0)  # camera down
    T[:3, 2] = (c, s, 0.0)    # camera forward
    T[:3, 3] = (x, y, CAMERA_HEIGHT)
    return T


def stable_hash(*parts) -> int:
    """64-bit content hash. The plan specifies BLAKE3; v0 uses stdlib BLAKE2b to avoid a dependency."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        elif isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return int.from_bytes(h.digest(), "little")


def cell_state_hash(world_seed: int, cell: int, state: dict) -> int:
    return stable_hash("cell", world_seed, cell, state)
