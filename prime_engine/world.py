"""Deterministic indoor house and pinhole renderer.

The engine owns pose, collision, and prop state. Frames are ground truth for
Solidify, not model output. Six rooms, doorways as portals, one prop each
that can be toppled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from prime_engine.schema import (
    ACT_FORWARD,
    ACT_INTERACT,
    ACT_TURN_LEFT,
    ACT_TURN_RIGHT,
    FrameRecord,
    state_hash,
)

FOV = math.radians(75.0)
WALL_H = 2.5
CEIL_H = 2.5
CAM_Z = 1.05
MAP_W = 20
MAP_H = 18

# id, name, wall rgb, floor rgb
ROOM_DEFS = (
    (0, "foyer", (186, 78, 58), (156, 118, 78)),
    (1, "hall", (214, 196, 170), (132, 128, 120)),
    (2, "study", (58, 108, 86), (92, 74, 54)),
    (3, "kitchen", (210, 206, 190), (176, 154, 108)),
    (4, "bedroom", (104, 82, 124), (126, 100, 92)),
    (5, "gallery", (64, 98, 138), (104, 112, 118)),
)

# Inclusive floor rectangles (x0, y0, x1, y1).
ROOM_RECTS = {
    0: (1, 1, 5, 5),
    1: (7, 1, 12, 5),
    2: (14, 1, 18, 5),
    3: (1, 7, 8, 11),
    4: (10, 7, 18, 11),
    5: (1, 13, 18, 16),
}

# Door cell, room label used when the player stands on it.
DOORS = (
    (6, 3, 1),
    (13, 3, 2),
    (3, 6, 3),
    (8, 6, 3),
    (9, 9, 4),
    (5, 12, 5),
    (14, 12, 5),
)

# Neighbors for the hot/warm residency set.
NEIGHBORS = {
    0: (1, 3),
    1: (0, 2, 3),
    2: (1,),
    3: (0, 1, 4, 5),
    4: (3, 5),
    5: (3, 4),
}

# Canonical standing pose (x, y, yaw) looking across the room at the prop.
CANONICAL = {
    0: (3.5, 4.4, -math.pi / 2),
    1: (9.5, 4.4, -math.pi / 2),
    2: (16.5, 4.4, -math.pi / 2),
    3: (4.5, 10.4, -math.pi / 2),
    4: (14.5, 10.4, -math.pi / 2),
    5: (10.5, 15.6, -math.pi / 2),
}


@dataclass
class Prop:
    name: str
    obj_id: int
    cell_id: int
    x: float
    y: float
    radius: float
    height: float
    color: tuple[int, int, int]
    toppled_height: float
    toppled_radius: float
    toppled_color: tuple[int, int, int]


PROPS = (
    Prop("foyer_vase", 1, 0, 3.5, 2.4, 0.28, 0.92, (210, 42, 48), 0.16, 0.46, (150, 36, 40)),
    Prop("hall_bench", 2, 1, 9.5, 2.3, 0.55, 0.48, (120, 82, 48), 0.14, 0.7, (96, 66, 40)),
    Prop("study_cabinet", 3, 2, 16.5, 2.2, 0.42, 1.15, (42, 78, 150), 0.22, 0.62, (36, 60, 120)),
    Prop("kitchen_stool", 4, 3, 4.6, 8.4, 0.22, 0.62, (230, 186, 48), 0.14, 0.36, (190, 150, 40)),
    Prop("bedroom_lamp", 5, 4, 14.5, 8.3, 0.16, 1.05, (240, 224, 170), 0.12, 0.34, (200, 180, 130)),
    Prop("gallery_plinth", 6, 5, 10.5, 14.2, 0.34, 0.72, (180, 180, 188), 0.16, 0.5, (140, 140, 150)),
)


def camera_k(width: int, height: int, fov: float = FOV) -> np.ndarray:
    fx = (width / 2.0) / math.tan(fov / 2.0)
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], np.float32)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def build_grid() -> np.ndarray:
    grid = np.full((MAP_H, MAP_W), -1, np.int16)
    for rid, (x0, y0, x1, y1) in ROOM_RECTS.items():
        grid[y0 : y1 + 1, x0 : x1 + 1] = rid
    for x, y, rid in DOORS:
        grid[y, x] = rid
    return grid


class House:
    """Authoritative world. Pose never comes from the model."""

    def __init__(self, seed: int = 0, width: int = 64, height: int = 48):
        self.seed = seed
        self.width = width
        self.height = height
        self.grid = build_grid()
        self.K = camera_k(width, height)
        self.fx = float(self.K[0, 0])
        self.fy = float(self.K[1, 1])
        self.cx = float(self.K[0, 2])
        self.cy = float(self.K[1, 2])
        self.wall_color = {rid: np.array(wall, np.float32) for rid, _, wall, _ in ROOM_DEFS}
        self.floor_color = {rid: np.array(floor, np.float32) for rid, _, _, floor in ROOM_DEFS}
        self.room_name = {rid: name for rid, name, _, _ in ROOM_DEFS}
        self.props = {p.name: p for p in PROPS}
        self.prop_state = {p.name: "upright" for p in PROPS}
        x, y, yaw = CANONICAL[0]
        self.x = x
        self.y = y
        self.yaw = yaw
        self.frame_id = 0
        self.episode_id = f"return-{seed}"

    def cell_at(self, x: float, y: float) -> int:
        ix, iy = int(math.floor(x)), int(math.floor(y))
        if ix < 0 or iy < 0 or ix >= MAP_W or iy >= MAP_H:
            return -1
        return int(self.grid[iy, ix])

    @property
    def cell_id(self) -> int:
        return self.cell_at(self.x, self.y)

    def game_state(self) -> dict:
        return {"props": dict(self.prop_state)}

    def pose_vec(self) -> np.ndarray:
        return np.array([self.x, self.y, CAM_Z, self.yaw, 0.0, 0.0], np.float32)

    def topple(self, name: str = "foyer_vase") -> None:
        if name not in self.prop_state:
            raise KeyError(name)
        self.prop_state[name] = "toppled"

    def _prop_geom(self, prop: Prop) -> tuple[float, float, tuple[int, int, int]]:
        if self.prop_state[prop.name] == "toppled":
            return prop.toppled_radius, prop.toppled_height, prop.toppled_color
        return prop.radius, prop.height, prop.color

    def render(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w, h = self.width, self.height
        rgb = np.zeros((h, w, 3), np.float32)
        depth = np.full((h, w), np.inf, np.float32)
        inst = np.zeros((h, w), np.uint8)

        us = (np.arange(w, dtype=np.float32) + 0.5 - self.cx) / self.fx
        vs = (np.arange(h, dtype=np.float32) + 0.5 - self.cy) / self.fy
        uu, vv = np.meshgrid(us, vs)
        fwd = np.array([math.cos(self.yaw), math.sin(self.yaw)], np.float32)
        right = np.array([-math.sin(self.yaw), math.cos(self.yaw)], np.float32)
        # Camera ray, unit length. Image +v is down, world +z is up.
        dir_x = fwd[0] + right[0] * uu
        dir_y = fwd[1] + right[1] * uu
        dir_z = -vv
        norm = np.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm

        # Floor z=0 and ceiling z=CEIL_H. Compute hits only on rays that face the plane.
        hit_x = np.zeros_like(dir_x)
        hit_y = np.zeros_like(dir_y)
        z_floor = np.full_like(dir_x, np.inf)
        face_floor = dir_z < -1e-4
        s_floor = -CAM_Z / dir_z[face_floor]
        hit_x[face_floor] = self.x + s_floor * dir_x[face_floor]
        hit_y[face_floor] = self.y + s_floor * dir_y[face_floor]
        z_floor[face_floor] = (hit_x[face_floor] - self.x) * fwd[0] + (hit_y[face_floor] - self.y) * fwd[1]
        valid_floor = face_floor & (z_floor > 0.05)
        ix = np.floor(hit_x).astype(np.int32)
        iy = np.floor(hit_y).astype(np.int32)
        inside = valid_floor & (ix >= 0) & (iy >= 0) & (ix < MAP_W) & (iy < MAP_H)
        room = np.full((h, w), -1, np.int16)
        room[inside] = self.grid[iy[inside], ix[inside]]
        walkable = inside & (room >= 0)
        checker = ((np.floor(hit_x * 2.0).astype(np.int32) + np.floor(hit_y * 2.0).astype(np.int32)) & 1).astype(np.float32)
        for rid, color in self.floor_color.items():
            m = walkable & (room == rid)
            shade = 0.82 + 0.18 * checker[m]
            rgb[m] = color * shade[:, None]
            depth[m] = z_floor[m]
        face_ceil = dir_z > 1e-4
        s_ceil = (CEIL_H - CAM_Z) / dir_z[face_ceil]
        z_ceil = np.full_like(dir_x, np.inf)
        z_ceil[face_ceil] = s_ceil * (dir_x[face_ceil] * fwd[0] + dir_y[face_ceil] * fwd[1])
        ceil = face_ceil & (z_ceil > 0.05) & (z_ceil < depth)
        rgb[ceil] = (228, 226, 220)
        depth[ceil] = z_ceil[ceil]

        self._paint_columns(rgb, depth, inst, fwd, right)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        depth_out = depth.astype(np.float16)
        depth_out[~np.isfinite(depth)] = np.float16(0)
        return rgb, depth_out, inst

    def _paint_columns(self, rgb: np.ndarray, depth: np.ndarray, inst: np.ndarray, fwd: np.ndarray, right: np.ndarray) -> None:
        w, h = self.width, self.height
        for u in range(w):
            x_cam = (u + 0.5 - self.cx) / self.fx
            ray = fwd + right * x_cam
            ray_norm = float(np.linalg.norm(ray))
            ray = ray / ray_norm
            wall = self._march(float(ray[0]), float(ray[1]))
            prop_hit = self._nearest_prop(float(ray[0]), float(ray[1]))
            # z_cam = euclidean * dot(unit_ray, forward). ray here is unit in xy,
            # and the horizontal camera vector (x_cam, 1) has length ray_norm before
            # normalizing the xy part... Using dot with forward is the perspective depth
            # of the xy hit because forward is unit and ray is the unit xy direction.
            if wall is not None:
                dist, hx, hy, side, prev_room = wall
                z_cam = dist * float(np.dot(ray, fwd))
                if z_cam > 0.05 and prev_room >= 0:
                    color = self.wall_color[prev_room] * (0.78 if side == 1 else 1.0)
                    self._paint_span(rgb, depth, inst, u, z_cam, 0.0, WALL_H, color, 0)
            if prop_hit is not None:
                dist, prop, radius, height, color = prop_hit
                z_cam = dist * float(np.dot(ray, fwd))
                if z_cam > 0.05:
                    self._paint_span(rgb, depth, inst, u, z_cam, 0.0, height, np.array(color, np.float32), prop.obj_id)

    def _paint_span(self, rgb, depth, inst, u, z_cam, z0, z1, color, obj_id) -> None:
        v0 = int(math.floor(self.cy - self.fy * ((z1 - CAM_Z) / z_cam)))
        v1 = int(math.ceil(self.cy - self.fy * ((z0 - CAM_Z) / z_cam)))
        v0 = max(0, v0)
        v1 = min(self.height, v1)
        if v1 <= v0:
            return
        closer = z_cam < depth[v0:v1, u]
        if not np.any(closer):
            return
        rows = np.arange(v0, v1)
        sel = rows[closer]
        rgb[sel, u] = color
        depth[sel, u] = z_cam
        inst[sel, u] = obj_id

    def _march(self, dx: float, dy: float):
        if abs(dx) < 1e-8:
            dx = 1e-8
        if abs(dy) < 1e-8:
            dy = 1e-8
        map_x = int(math.floor(self.x))
        map_y = int(math.floor(self.y))
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        t_delta_x = abs(1.0 / dx)
        t_delta_y = abs(1.0 / dy)
        if dx > 0:
            t_max_x = (map_x + 1 - self.x) * t_delta_x
        else:
            t_max_x = (self.x - map_x) * t_delta_x
        if dy > 0:
            t_max_y = (map_y + 1 - self.y) * t_delta_y
        else:
            t_max_y = (self.y - map_y) * t_delta_y
        prev_room = self.cell_at(self.x, self.y)
        dist = 0.0
        for _ in range(48):
            if t_max_x < t_max_y:
                map_x += step_x
                dist = t_max_x
                t_max_x += t_delta_x
                side = 0
            else:
                map_y += step_y
                dist = t_max_y
                t_max_y += t_delta_y
                side = 1
            if dist > 24.0 or map_x < 0 or map_y < 0 or map_x >= MAP_W or map_y >= MAP_H:
                return None
            rid = int(self.grid[map_y, map_x])
            if rid < 0:
                hx = self.x + dx * dist
                hy = self.y + dy * dist
                return dist, hx, hy, side, prev_room
            prev_room = rid
        return None

    def _nearest_prop(self, dx: float, dy: float):
        best = None
        for prop in PROPS:
            radius, height, color = self._prop_geom(prop)
            t = _ray_circle(self.x, self.y, dx, dy, prop.x, prop.y, radius)
            if t is None:
                continue
            if best is None or t < best[0]:
                best = (t, prop, radius, height, color)
        return best

    def _occupied(self, x: float, y: float) -> bool:
        return self.cell_at(x, y) < 0

    def try_move(self, nx: float, ny: float) -> None:
        if not self._occupied(nx, self.y):
            self.x = nx
        if not self._occupied(self.x, ny):
            self.y = ny

    def apply(self, action: int) -> tuple[int, float]:
        """Apply one control. Returns (action, mouse_dx yaw delta)."""
        yaw0 = self.yaw
        step = 0.28
        turn = 0.22
        if action & ACT_TURN_LEFT:
            self.yaw = wrap_angle(self.yaw - turn)
        elif action & ACT_TURN_RIGHT:
            self.yaw = wrap_angle(self.yaw + turn)
        elif action & ACT_FORWARD:
            self.try_move(self.x + math.cos(self.yaw) * step, self.y + math.sin(self.yaw) * step)
        elif action & ACT_INTERACT:
            self._interact()
        return action, wrap_angle(self.yaw - yaw0)

    def _interact(self) -> None:
        for prop in PROPS:
            if prop.cell_id != self.cell_id:
                continue
            if math.hypot(prop.x - self.x, prop.y - self.y) < 2.4:
                self.prop_state[prop.name] = "toppled"

    def bfs(self, goal: tuple[int, int]) -> list[tuple[int, int]]:
        start = (int(math.floor(self.x)), int(math.floor(self.y)))
        if start == goal:
            return [start]
        q = [start]
        prev = {start: None}
        while q:
            cx, cy = q.pop(0)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in prev:
                    continue
                if nx < 0 or ny < 0 or nx >= MAP_W or ny >= MAP_H:
                    continue
                if int(self.grid[ny, nx]) < 0:
                    continue
                prev[(nx, ny)] = (cx, cy)
                if (nx, ny) == goal:
                    path = [(nx, ny)]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append((nx, ny))
        return []


def _ray_circle(ox, oy, dx, dy, cx, cy, r) -> float | None:
    fx, fy = ox - cx, oy - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    t1 = (-b - root) * 0.5
    if t1 > 0.05:
        return t1
    t2 = (-b + root) * 0.5
    if t2 > 0.05:
        return t2
    return None


class Capture:
    """Walks the house and emits section-2.3 frames. Solidify is optional."""

    def __init__(self, house: House, solidify=None, fps: float = 30.0):
        self.house = house
        self.solidify = solidify
        self.fps = fps
        self.frames: list[FrameRecord] = []
        self._prev_cell = house.cell_id

    def shoot(self, action: int = 0, mouse_dx: float = 0.0) -> FrameRecord:
        guide_rgb = np.zeros((self.house.height, self.house.width, 3), np.uint8)
        guide_mask = np.zeros((self.house.height, self.house.width), np.uint8)
        cell_stats = {}
        if self.solidify is not None:
            guide_rgb, guide_mask = self.solidify.guide(self.house)
            cell_stats = self.solidify.cell_stats()
        rgb, depth, inst = self.house.render()
        cell = self.house.cell_id
        portal = cell != self._prev_cell
        self._prev_cell = cell
        game_state = self.house.game_state()
        digest = state_hash(game_state, cell_stats)
        pose = self.house.pose_vec()
        frame = FrameRecord(
            frame_id=self.house.frame_id,
            t=self.house.frame_id / self.fps,
            episode_id=self.house.episode_id,
            world_seed=self.house.seed,
            rgb=rgb,
            depth=depth,
            instance_id=inst,
            K=self.house.K.copy(),
            pose=pose,
            player_pose=pose.copy(),
            action_keys=action,
            mouse_dx=mouse_dx,
            mouse_dy=0.0,
            buttons=1 if action & ACT_INTERACT else 0,
            cell_id=cell,
            portal_crossed=portal,
            game_state=game_state,
            state_hash=digest,
            guide_rgb=guide_rgb,
            guide_mask=guide_mask,
        )
        self.house.frame_id += 1
        self.frames.append(frame)
        if self.solidify is not None:
            self.solidify.observe(self.house, frame)
        return frame

    def act(self, action: int) -> FrameRecord:
        action, mouse_dx = self.house.apply(action)
        return self.shoot(action, mouse_dx)

    def go_to_cell(self, tx: int, ty: int, max_steps: int = 280) -> None:
        for _ in range(max_steps):
            if int(math.floor(self.house.x)) == tx and int(math.floor(self.house.y)) == ty:
                return
            path = self.house.bfs((tx, ty))
            if len(path) < 2:
                raise RuntimeError(f"no path to {(tx, ty)} from {(self.house.x, self.house.y)}")
            wx, wy = path[1][0] + 0.5, path[1][1] + 0.5
            desired = math.atan2(wy - self.house.y, wx - self.house.x)
            err = wrap_angle(desired - self.house.yaw)
            if abs(err) > 0.28:
                self.act(ACT_TURN_LEFT if err < 0 else ACT_TURN_RIGHT)
            else:
                self.act(ACT_FORWARD)
        raise RuntimeError(f"failed to reach {(tx, ty)}")

    def go_to_pose(self, x: float, y: float, yaw: float) -> None:
        self.go_to_cell(int(math.floor(x)), int(math.floor(y)))
        # Settle on the exact canonical point. Collision is cell-based, and the
        # cell is already walkable, so this does not tunnel through walls.
        self.house.x = x
        self.house.y = y
        for _ in range(24):
            err = wrap_angle(yaw - self.house.yaw)
            if abs(err) < 0.12:
                break
            self.act(ACT_TURN_LEFT if err < 0 else ACT_TURN_RIGHT)
        self.house.yaw = yaw
        self.shoot(0, 0.0)

    def scan(self, turns: int = 12) -> None:
        step = ACT_TURN_RIGHT
        for _ in range(turns):
            self.act(step)


def capture_the_return(width: int = 64, height: int = 48, seed: int = 0):
    """Scripted episode: see the house, topple the foyer vase, leave, come back.

    Returns (capture, indexes) where indexes mark the locked hashes and the
    foyer frame used as ground truth for the revisit metric.
    """
    from prime_engine.solidify import Solidify

    house = House(seed=seed, width=width, height=height)
    solidify = Solidify()
    cap = Capture(house, solidify)
    # First sight of every room, including a spin so the cache covers the walls.
    for rid in (0, 1, 2, 3, 4, 5):
        x, y, yaw = CANONICAL[rid]
        cap.go_to_pose(x, y, yaw)
        cap.scan(12)
        solidify.lock(rid)
    # The scan ends in the gallery. Walk back before editing the foyer.
    fx, fy, fyaw = CANONICAL[0]
    cap.go_to_pose(fx, fy, fyaw)
    cap.shoot(0, 0.0)
    hash_before = cap.frames[-1].state_hash
    # Knock the vase over and look at it long enough to re-lock the foyer.
    cap.act(ACT_INTERACT)
    solidify.invalidate(0, PROPS[0].obj_id)
    cap.scan(8)
    solidify.lock(0)
    cap.shoot(0, 0.0)
    locked_index = len(cap.frames) - 1
    hash_locked = cap.frames[-1].state_hash
    # Leave through three rooms.
    for rid in (1, 2, 3):
        x, y, yaw = CANONICAL[rid]
        cap.go_to_pose(x, y, yaw)
    # Come back to the foyer pose from before the walk.
    fx, fy, fyaw = CANONICAL[0]
    cap.go_to_pose(fx, fy, fyaw)
    return_index = len(cap.frames) - 1
    return cap, {
        "hash_before": hash_before,
        "hash_locked": hash_locked,
        "hash_return": cap.frames[return_index].state_hash,
        "locked_index": locked_index,
        "return_index": return_index,
        "foyer_cell": 0,
    }
