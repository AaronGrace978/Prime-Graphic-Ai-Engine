"""Exploration policies used to collect trajectories (frames + actions + poses)."""
from __future__ import annotations

import numpy as np

from .raycaster import (BACKWARD, FORWARD, NOOP, NUM_ACTIONS, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT, TURN_RIGHT,
                        RoomWorld, wrap_angle)


class WaypointExplorer:
    """Walks to random goals (biased toward other rooms) with look-arounds and noise.

    This produces lots of room exits and re-entries, which is exactly the data
    the Solidify memory conditioning needs to learn to copy from past views.
    """

    def __init__(self, world: RoomWorld, rng: np.random.Generator, noise: float = 0.08,
                 look_around_prob: float = 0.015, other_room_prob: float = 0.75):
        self.w = world
        self.rng = rng
        self.noise = noise
        self.look_around_prob = look_around_prob
        self.other_room_prob = other_room_prob
        self.path: list[tuple[int, int]] = []
        self.queue: list[int] = []
        self.last_xy = None
        self.stuck = 0
        self.auto_goal = True

    def _new_goal(self):
        x, y, _ = self.w.pose
        cur = self.w.room_at(x, y)
        cells = self.w.free_cells
        if cur >= 0 and self.rng.random() < self.other_room_prob:
            others = cells[self.w.room_id[cells[:, 0], cells[:, 1]] != cur]
            cells = others if len(others) else cells
        gy, gx = cells[self.rng.integers(len(cells))]
        self.path = self.w.bfs_path((x, y), (int(gx), int(gy)))[1:]

    def act(self) -> int:
        if self.queue:
            return self.queue.pop()
        r = self.rng.random()
        if r < self.look_around_prob:
            self.queue = [self.rng.choice([TURN_LEFT, TURN_RIGHT])] * int(self.rng.integers(4, 25))
            return self.queue.pop()
        if r < self.look_around_prob + self.noise:
            return int(self.rng.choice([NOOP, BACKWARD, STRAFE_LEFT, STRAFE_RIGHT, TURN_LEFT, TURN_RIGHT, FORWARD]))
        return self.navigate()

    def navigate(self) -> int:
        x, y, th = self.w.pose
        xy = (round(x, 3), round(y, 3))
        if self.last_xy is not None:
            self.stuck = self.stuck + 1 if xy == self.last_xy else 0
        self.last_xy = None
        if self.stuck > 3:
            # Replan to the same goal from here; the first waypoint (current cell centre) pulls us off the jamb.
            self.path = self.w.bfs_path((x, y), self.path[-1]) if self.path else []
            self.stuck = 0
        while self.path:
            cx, cy = self.path[0]
            if np.hypot(cx + 0.5 - x, cy + 0.5 - y) < 0.35:
                self.path.pop(0)
            else:
                break
        if not self.path:
            if not self.auto_goal:
                return NOOP
            self._new_goal()
            if not self.path:
                return int(self.rng.integers(NUM_ACTIONS))
        # String-pulling: aim at the farthest upcoming waypoint in clear line of sight.
        target = 0
        for i in range(min(len(self.path), 8)):
            if self._clear(x, y, self.path[i][0] + 0.5, self.path[i][1] + 0.5):
                target = i
        del self.path[:target]
        cx, cy = self.path[0]
        want = np.arctan2(cy + 0.5 - y, cx + 0.5 - x)
        diff = float(wrap_angle(want - th))
        # Headings live on a turn_deg lattice; a threshold above half a step avoids zigzagging.
        thresh = np.deg2rad(self.w.cfg.turn_deg) * 0.8
        if diff > thresh:
            return TURN_RIGHT
        if diff < -thresh:
            return TURN_LEFT
        self.last_xy = xy
        return FORWARD

    def _clear(self, x0, y0, x1, y1) -> bool:
        n = int(np.ceil(np.hypot(x1 - x0, y1 - y0) / 0.15)) + 1
        r = self.w.cfg.agent_radius + 0.15  # margin: headings are quantised, so the walked line wobbles
        return not any(self.w._collides(x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, r) for s in np.linspace(0, 1, n))


def scripted_actions_to(world: RoomWorld, pose, goal_xy, max_steps: int = 200) -> list[int]:
    """Deterministic shortest-path controller (no noise) used for eval scripts."""
    sim = RoomWorld(world.cfg, room_colors=world.room_colors)
    sim.pose = np.asarray(pose, dtype=np.float64).copy()
    explorer = WaypointExplorer(sim, np.random.default_rng(0), noise=0.0, look_around_prob=0.0)
    explorer.auto_goal = False
    explorer.path = sim.bfs_path(sim.pose[:2], (int(goal_xy[0]), int(goal_xy[1])))
    actions = []
    for _ in range(max_steps):
        a = explorer.navigate()
        if a == NOOP:
            break
        actions.append(a)
        sim.step(a)
    return actions
