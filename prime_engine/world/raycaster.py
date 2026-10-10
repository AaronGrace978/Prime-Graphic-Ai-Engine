"""Multi-room 2.5D raycaster world.

This is the "game logic" side of the PoC: it owns the map, collision and the
camera pose. It can render ground-truth frames (used for dataset generation and
evaluation), while the diffusion model acts as the neural renderer at rollout
time.

Each episode samples a random wall colour per room from a fixed hue palette, so
a room's appearance cannot be inferred from the map alone: once a room is out
of the model's short context window, only memory (the Solidify harness) can
bring it back consistently.
"""
from __future__ import annotations

import colorsys
from collections import deque
from dataclasses import dataclass

import numpy as np

# Action ids. Keep in sync with ACTION_NAMES.
NOOP, FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STRAFE_LEFT, STRAFE_RIGHT = range(7)
ACTION_NAMES = ["noop", "forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
NUM_ACTIONS = len(ACTION_NAMES)

# Eight well-separated hues; room colours are drawn from these without replacement.
PALETTE_HUES = np.array([0.0, 0.08, 0.16, 0.33, 0.5, 0.62, 0.75, 0.87])
PALETTE_RGB = np.array([colorsys.hsv_to_rgb(h, 0.75, 0.95) for h in PALETTE_HUES], dtype=np.float32)

DOOR_ROOM = -1
FLOOR_ID = -2
CEIL_ID = -3
CAMERA_HEIGHT = 0.5  # walls are 1 unit tall
FLOOR_RGB = np.array([0.42, 0.38, 0.34], dtype=np.float32)
CEIL_RGB = np.array([0.20, 0.22, 0.27], dtype=np.float32)
DOOR_RGB = np.array([0.55, 0.55, 0.55], dtype=np.float32)


@dataclass
class WorldConfig:
    width: int = 64
    height: int = 64
    supersample: int = 2
    fov_deg: float = 66.0
    rooms_x: int = 2
    rooms_y: int = 2
    room_size: int = 6
    door_width: int = 2
    move_step: float = 0.3
    turn_deg: float = 15.0
    agent_radius: float = 0.2
    max_ray_steps: int = 64


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class RoomWorld:
    """Grid of rectangular rooms connected by doorways (a ring for 2x2)."""

    def __init__(self, cfg: WorldConfig, room_colors: np.ndarray | None = None, seed: int | None = None):
        self.cfg = cfg
        self.world_seed = int(seed) if seed is not None else int(np.random.default_rng().integers(1 << 31))
        self.rng = np.random.default_rng(self.world_seed)
        self.n_rooms = cfg.rooms_x * cfg.rooms_y
        self._build_map()
        if room_colors is None:
            room_colors = self.rng.choice(len(PALETTE_RGB), size=self.n_rooms, replace=False)
        self.room_colors = np.asarray(room_colors, dtype=np.int64)
        self.pose = np.array([*self.room_center(0), 0.0], dtype=np.float64)

    # ------------------------------------------------------------------ map
    def _build_map(self):
        c = self.cfg
        R = c.room_size
        gw = c.rooms_x * (R + 1) + 1
        gh = c.rooms_y * (R + 1) + 1
        grid = np.ones((gh, gw), dtype=np.int8)  # [y, x], 1 = wall
        room_id = np.full((gh, gw), DOOR_ROOM, dtype=np.int64)
        for ry in range(c.rooms_y):
            for rx in range(c.rooms_x):
                x0, y0 = 1 + rx * (R + 1), 1 + ry * (R + 1)
                grid[y0:y0 + R, x0:x0 + R] = 0
                room_id[y0:y0 + R, x0:x0 + R] = ry * c.rooms_x + rx
        d0 = (R - c.door_width) // 2
        self.portals = []  # (cell_a, cell_b, (x0, y0, x1, y1) doorway bounds)
        for ry in range(c.rooms_y):
            for rx in range(c.rooms_x):
                x0, y0 = 1 + rx * (R + 1), 1 + ry * (R + 1)
                r = ry * c.rooms_x + rx
                if rx + 1 < c.rooms_x:  # door east
                    grid[y0 + d0:y0 + d0 + c.door_width, x0 + R] = 0
                    self.portals.append((r, r + 1, (x0 + R, y0 + d0, x0 + R + 1, y0 + d0 + c.door_width)))
                if ry + 1 < c.rooms_y:  # door south
                    grid[y0 + R, x0 + d0:x0 + d0 + c.door_width] = 0
                    self.portals.append((r, r + c.rooms_x, (x0 + d0, y0 + R, x0 + d0 + c.door_width, y0 + R + 1)))
        self.cell_bounds = {
            ry * c.rooms_x + rx: (1 + rx * (R + 1), 1 + ry * (R + 1), 1 + rx * (R + 1) + R, 1 + ry * (R + 1) + R)
            for ry in range(c.rooms_y) for rx in range(c.rooms_x)}
        self.neighbors = {r: sorted({b for a, b, _ in self.portals if a == r} | {a for a, b, _ in self.portals if b == r})
                          for r in self.cell_bounds}
        self.grid = grid
        self.room_id = room_id
        self.free_cells = np.argwhere(grid == 0)  # (y, x)

    def room_center(self, r: int) -> tuple[float, float]:
        R = self.cfg.room_size
        rx, ry = r % self.cfg.rooms_x, r // self.cfg.rooms_x
        return 1 + rx * (R + 1) + R / 2.0, 1 + ry * (R + 1) + R / 2.0

    def room_at(self, x: float, y: float) -> int:
        return int(self.room_id[int(y), int(x)])

    def cell_graph(self) -> dict:
        """Cells + portals in the form SolidifyStore.register_cells expects (Anchored mode)."""
        return {c: {"bounds": self.cell_bounds[c], "neighbors": self.neighbors[c], "state": self.cell_state(c)}
                for c in self.cell_bounds}

    def cell_state(self, cell: int) -> dict:
        """Authoritative per-cell game state (plan §3.2 `state`). v0: just the wall appearance."""
        return {"wall_color": int(self.room_colors[cell])} if cell >= 0 else {}

    def is_wall(self, x: float, y: float) -> bool:
        ix, iy = int(np.floor(x)), int(np.floor(y))
        if iy < 0 or ix < 0 or iy >= self.grid.shape[0] or ix >= self.grid.shape[1]:
            return True
        return bool(self.grid[iy, ix])

    def _collides(self, x: float, y: float, r: float | None = None) -> bool:
        r = self.cfg.agent_radius if r is None else r
        return any(self.is_wall(x + dx, y + dy) for dx in (-r, r) for dy in (-r, r))

    # ------------------------------------------------------------- dynamics
    def reset(self, pose=None, random_start: bool = False):
        if pose is not None:
            self.pose = np.asarray(pose, dtype=np.float64).copy()
        elif random_start:
            cy, cx = self.free_cells[self.rng.integers(len(self.free_cells))]
            heading = self.rng.integers(int(round(360 / self.cfg.turn_deg))) * np.deg2rad(self.cfg.turn_deg)
            self.pose = np.array([cx + 0.5, cy + 0.5, wrap_angle(heading)], dtype=np.float64)
        else:
            self.pose = np.array([*self.room_center(0), 0.0], dtype=np.float64)
        return self.pose.copy()

    def next_pose(self, pose, action: int) -> np.ndarray:
        """Pure kinematics + collision: the pose the engine would reach."""
        x, y, th = pose
        c = self.cfg
        turn = np.deg2rad(c.turn_deg)
        dx = dy = 0.0
        if action == FORWARD:
            dx, dy = np.cos(th) * c.move_step, np.sin(th) * c.move_step
        elif action == BACKWARD:
            dx, dy = -np.cos(th) * c.move_step, -np.sin(th) * c.move_step
        elif action == STRAFE_LEFT:
            dx, dy = np.sin(th) * c.move_step, -np.cos(th) * c.move_step
        elif action == STRAFE_RIGHT:
            dx, dy = -np.sin(th) * c.move_step, np.cos(th) * c.move_step
        elif action == TURN_LEFT:
            th = th - turn
        elif action == TURN_RIGHT:
            th = th + turn
        # Axis-separated sliding collision.
        if dx and not self._collides(x + dx, y):
            x += dx
        if dy and not self._collides(x, y + dy):
            y += dy
        return np.array([x, y, float(wrap_angle(th))], dtype=np.float64)

    def step(self, action: int) -> np.ndarray:
        self.pose = self.next_pose(self.pose, action)
        return self.pose.copy()

    # ------------------------------------------------------------ rendering
    def render(self, pose=None, aux: bool = False):
        """Render an HxWx3 uint8 frame from `pose` (defaults to current pose).

        With aux=True also returns linear depth (float16, world units along the
        optical axis) and an instance map (int8: room id for walls, DOOR_ROOM for
        door jambs, FLOOR_ID / CEIL_ID).
        """
        c = self.cfg
        ss = c.supersample
        W, H = c.width * ss, c.height * ss
        x, y, th = self.pose if pose is None else pose
        half = np.tan(np.deg2rad(c.fov_deg) / 2)
        cam = (2 * (np.arange(W) + 0.5) / W - 1) * half
        dir_x, dir_y = np.cos(th), np.sin(th)
        # Screen x grows to the right == clockwise in our (y-down) map frame.
        plane_x, plane_y = -dir_y, dir_x
        rdx = dir_x + plane_x * cam
        rdy = dir_y + plane_y * cam
        with np.errstate(divide="ignore"):
            ddx = np.where(rdx == 0, 1e30, np.abs(1 / rdx))
            ddy = np.where(rdy == 0, 1e30, np.abs(1 / rdy))
        mx = np.full(W, int(np.floor(x)))
        my = np.full(W, int(np.floor(y)))
        step_x = np.where(rdx < 0, -1, 1)
        step_y = np.where(rdy < 0, -1, 1)
        sdx = np.where(rdx < 0, (x - mx) * ddx, (mx + 1 - x) * ddx)
        sdy = np.where(rdy < 0, (y - my) * ddy, (my + 1 - y) * ddy)
        hit = np.zeros(W, bool)
        side = np.zeros(W, np.int8)
        prev_x, prev_y = mx.copy(), my.copy()
        gh, gw = self.grid.shape
        for _ in range(c.max_ray_steps):
            active = ~hit
            if not active.any():
                break
            go_x = active & (sdx < sdy)
            go_y = active & ~go_x
            prev_x = np.where(active, mx, prev_x)
            prev_y = np.where(active, my, prev_y)
            sdx = np.where(go_x, sdx + ddx, sdx)
            mx = np.where(go_x, mx + step_x, mx)
            sdy = np.where(go_y, sdy + ddy, sdy)
            my = np.where(go_y, my + step_y, my)
            side = np.where(go_x, 0, np.where(go_y, 1, side)).astype(np.int8)
            mxc, myc = np.clip(mx, 0, gw - 1), np.clip(my, 0, gh - 1)
            hit |= active & (self.grid[myc, mxc] == 1)
        perp = np.where(side == 0, sdx - ddx, sdy - ddy)
        perp = np.maximum(perp, 1e-3)
        # Wall texture coordinate for subtle vertical panels (motion cue).
        wall_u = np.where(side == 0, y + perp * rdy, x + perp * rdx)
        wall_u = wall_u - np.floor(wall_u)
        rooms = self.room_id[np.clip(prev_y, 0, gh - 1), np.clip(prev_x, 0, gw - 1)]
        col = np.where((rooms >= 0)[:, None], PALETTE_RGB[self.room_colors[np.maximum(rooms, 0)]], DOOR_RGB[None])
        shade = np.where(side == 1, 0.72, 1.0) * (1.0 / (1.0 + 0.06 * perp))
        panel = 1.0 - 0.18 * (np.floor(wall_u * 3) % 2)
        col = col * (shade * panel)[:, None]

        line_h = H / perp
        top = (H / 2 - line_h / 2)[None, :]
        bot = (H / 2 + line_h / 2)[None, :]
        rows = (np.arange(H) + 0.5)[:, None]
        is_wall = (rows >= top) & (rows < bot)
        is_ceil = rows < top
        rd = np.abs(rows - H / 2) / (H / 2)  # 0 at horizon, 1 at border
        floor = FLOOR_RGB[None, None] * (0.45 + 0.55 * rd)[..., None]
        ceil = CEIL_RGB[None, None] * (0.6 + 0.4 * rd)[..., None]
        img = np.where(is_ceil[..., None], np.broadcast_to(ceil, (H, 1, 3)), np.broadcast_to(floor, (H, 1, 3)))
        img = np.where(is_wall[..., None], col[None, :, :], img)
        if ss > 1:
            img = img.reshape(c.height, ss, c.width, ss, 3).mean(axis=(1, 3))
        rgb = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        if not aux:
            return rgb
        # Camera at half wall height: floor/ceiling depth = (H/2) / |row - H/2|.
        plane_depth = (H / 2) / np.maximum(np.abs(rows - H / 2), 0.5)
        depth = np.where(is_wall, perp[None, :], np.broadcast_to(plane_depth, (H, W)))
        inst = np.where(is_wall, rooms[None, :], np.where(is_ceil, CEIL_ID, FLOOR_ID)).astype(np.int8)
        if ss > 1:
            depth = depth.reshape(c.height, ss, c.width, ss).mean(axis=(1, 3))
            inst = inst[ss // 2::ss, ss // 2::ss]
        return rgb, depth.astype(np.float16), inst

    # ------------------------------------------------------------ planning
    def bfs_path(self, start_xy, goal_cell) -> list[tuple[int, int]]:
        """4-connected BFS over free cells; returns list of (x, y) cells."""
        sx, sy = int(start_xy[0]), int(start_xy[1])
        gx, gy = goal_cell
        prev = {(sx, sy): None}
        q = deque([(sx, sy)])
        while q:
            cx, cy = q.popleft()
            if (cx, cy) == (gx, gy):
                break
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (nx, ny) not in prev and not self.grid[ny, nx]:
                    prev[(nx, ny)] = (cx, cy)
                    q.append((nx, ny))
        if (gx, gy) not in prev:
            return []
        path, node = [], (gx, gy)
        while node is not None:
            path.append(node)
            node = prev[node]
        return path[::-1]
