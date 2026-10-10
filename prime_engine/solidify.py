"""Solidify v0: cells, a voxel cache, pose-indexed keyframes, guide renders.

The cache is canonical geometry. Once a cell is SOLID it is not rewritten,
so a later visit is rendered from stored data instead of from the model's memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from prime_engine.world import CAM_Z, NEIGHBORS

VOXEL = 0.12


@dataclass
class _Acc:
    rgb: np.ndarray
    n: int
    obj_id: int


@dataclass
class Cell:
    id: int
    status: str = "UNSEEN"  # UNSEEN | FORMING | SOLID | DIRTY
    tier: str = "cold"  # hot | warm | cold
    voxels: dict = field(default_factory=dict)
    keyframes: list = field(default_factory=list)
    checksum: int = 0
    _xyz: np.ndarray | None = None
    _rgb: np.ndarray | None = None

    def touch(self) -> None:
        self._xyz = None
        self._rgb = None

    def packed(self) -> tuple[np.ndarray, np.ndarray]:
        if self._xyz is None:
            if not self.voxels:
                self._xyz = np.zeros((0, 3), np.float32)
                self._rgb = np.zeros((0, 3), np.float32)
            else:
                keys = list(self.voxels.keys())
                accs = [self.voxels[k] for k in keys]
                xyz = np.array([[(k[0] + 0.5) * VOXEL, (k[1] + 0.5) * VOXEL, (k[2] + 0.5) * VOXEL] for k in keys], np.float32)
                rgb = np.clip(np.stack([a.rgb / max(a.n, 1) for a in accs]), 0, 255).astype(np.float32)
                self._xyz = xyz
                self._rgb = rgb
        return self._xyz, self._rgb

    def rehash(self) -> None:
        total = 0
        for (ix, iy, iz), acc in self.voxels.items():
            color = acc.rgb / max(acc.n, 1)
            total = (total + ix * 3 + iy * 5 + iz * 7 + int(color[0]) + int(color[1]) * 3 + int(color[2]) * 9) & 0xFFFFFFFF
        self.checksum = int(total)


class Solidify:
    def __init__(self):
        self.cells = {i: Cell(id=i) for i in range(6)}
        self._kf_stride = 0

    def cell_stats(self) -> dict:
        # Tier is streaming residency, not canonical state, so it stays out of the hash.
        return {
            str(c.id): {"status": c.status, "voxels": len(c.voxels), "sum": c.checksum}
            for c in self.cells.values()
        }

    def update_residency(self, cell_id: int) -> None:
        hot = {cell_id, *NEIGHBORS.get(cell_id, ())}
        for c in self.cells.values():
            if c.id in hot:
                c.tier = "hot"
            elif c.tier == "hot":
                c.tier = "warm"
            elif c.tier != "warm":
                c.tier = "cold"

    def lock(self, cell_id: int) -> None:
        cell = self.cells[cell_id]
        cell.rehash()
        cell.status = "SOLID"

    def invalidate(self, cell_id: int, obj_id: int) -> None:
        """Drop surfels of one object so an edit can re-solidify that region."""
        cell = self.cells[cell_id]
        cell.voxels = {k: v for k, v in cell.voxels.items() if v.obj_id != obj_id}
        cell.keyframes = [kf for kf in cell.keyframes if not kf["obj_touched"]]
        cell.status = "DIRTY"
        cell.touch()
        cell.rehash()

    def observe(self, house, frame) -> None:
        self.update_residency(frame.cell_id)
        self._fuse(house, frame)
        self._maybe_keyframe(frame)
        cell = self.cells[frame.cell_id]
        if cell.status == "UNSEEN":
            cell.status = "FORMING"

    def _fuse(self, house, frame) -> None:
        depth = np.asarray(frame.depth, np.float32)
        rgb = frame.rgb
        inst = frame.instance_id
        h, w = depth.shape
        pose = frame.pose
        cam_x, cam_y, cam_z, yaw = float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])
        fwd = np.array([math.cos(yaw), math.sin(yaw)], np.float32)
        right = np.array([-math.sin(yaw), math.cos(yaw)], np.float32)
        fx = float(frame.K[0, 0])
        fy = float(frame.K[1, 1])
        cx = float(frame.K[0, 2])
        cy = float(frame.K[1, 2])
        # Stride keeps the cache small. Surfaces still cover a 64px frame.
        vs = np.arange(0, h, 2)
        us = np.arange(0, w, 2)
        uu, vv = np.meshgrid(us, vs)
        z = depth[vv, uu]
        good = np.isfinite(z) & (z > 0.08) & (z < 12.0)
        if not np.any(good):
            return
        uu, vv, z = uu[good], vv[good], z[good]
        x_cam = (uu + 0.5 - cx) / fx
        y_cam = (vv + 0.5 - cy) / fy
        # Forward depth z, image +v is down.
        world_x = cam_x + fwd[0] * z + right[0] * x_cam * z
        world_y = cam_y + fwd[1] * z + right[1] * x_cam * z
        world_z = cam_z - y_cam * z
        colors = rgb[vv, uu].astype(np.float32)
        objs = inst[vv, uu].astype(np.int32)
        ix = np.floor(world_x / VOXEL).astype(np.int32)
        iy = np.floor(world_y / VOXEL).astype(np.int32)
        iz = np.floor(world_z / VOXEL).astype(np.int32)
        for i in range(len(z)):
            cid = house.cell_at(float(world_x[i]), float(world_y[i]))
            if cid < 0:
                # Attribute a wall hit to the player's cell so rooms still fill in.
                cid = int(frame.cell_id)
            cell = self.cells[cid]
            if cell.status == "SOLID":
                continue
            key = (int(ix[i]), int(iy[i]), int(iz[i]))
            acc = cell.voxels.get(key)
            obj = int(objs[i])
            if acc is None:
                cell.voxels[key] = _Acc(rgb=colors[i].copy(), n=1, obj_id=obj)
            else:
                acc.rgb += colors[i]
                acc.n += 1
                if obj:
                    acc.obj_id = obj
            cell.touch()
        # Bound memory. Distant, low-count samples go first.
        for cell in self.cells.values():
            if len(cell.voxels) > 8000:
                ranked = sorted(cell.voxels.items(), key=lambda kv: kv[1].n)
                cell.voxels = dict(ranked[-6000:])
                cell.touch()

    def _maybe_keyframe(self, frame) -> None:
        cell = self.cells[frame.cell_id]
        if cell.status == "SOLID" and len(cell.keyframes) >= 8:
            return
        pose = frame.pose
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[3])
        for kf in cell.keyframes:
            if math.hypot(kf["x"] - x, kf["y"] - y) < 0.45 and abs(_wrap(kf["yaw"] - yaw)) < 0.35:
                return
        if len(cell.keyframes) >= 24:
            cell.keyframes.pop(0)
        touched = bool(np.any(frame.instance_id > 0))
        cell.keyframes.append(
            {
                "x": x,
                "y": y,
                "yaw": yaw,
                "rgb": frame.rgb.copy(),
                "obj_touched": touched,
            }
        )

    def retrieve(self, cell_id: int, x: float, y: float, yaw: float):
        cell = self.cells[cell_id]
        if not cell.keyframes:
            return None
        best, score = None, 1e9
        for kf in cell.keyframes:
            s = math.hypot(kf["x"] - x, kf["y"] - y) + 0.6 * abs(_wrap(kf["yaw"] - yaw))
            if s < score:
                best, score = kf, s
        return best

    def guide(self, house) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize the cache at the current pose. Returns RGB and coverage mask."""
        h, w = house.height, house.width
        rgb = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w), np.uint8)
        chunks_xyz = []
        chunks_rgb = []
        for cell in self.cells.values():
            xyz, cols = cell.packed()
            if len(xyz):
                chunks_xyz.append(xyz)
                chunks_rgb.append(cols)
        if not chunks_xyz:
            return rgb, mask
        pts_a = np.concatenate(chunks_xyz)
        cols_a = np.concatenate(chunks_rgb)
        fwd = np.array([math.cos(house.yaw), math.sin(house.yaw)], np.float32)
        right = np.array([-math.sin(house.yaw), math.cos(house.yaw)], np.float32)
        rel_x = pts_a[:, 0] - house.x
        rel_y = pts_a[:, 1] - house.y
        z_cam = rel_x * fwd[0] + rel_y * fwd[1]
        x_cam = rel_x * right[0] + rel_y * right[1]
        y_up = pts_a[:, 2] - CAM_Z
        ok = z_cam > 0.12
        if not np.any(ok):
            return rgb, mask
        z_cam, x_cam, y_up, cols_a = z_cam[ok], x_cam[ok], y_up[ok], cols_a[ok]
        u = house.cx + house.fx * (x_cam / z_cam)
        v = house.cy - house.fy * (y_up / z_cam)
        # Expand each surfel to a 3x3 so the guide is dense enough to condition on.
        offsets = ((0, 0, 0.0), (1, 0, 0.01), (-1, 0, 0.01), (0, 1, 0.01), (0, -1, 0.01))
        us, vs, zs, cs = [], [], [], []
        for du, dv, bias in offsets:
            us.append(np.rint(u + du).astype(np.int32))
            vs.append(np.rint(v + dv).astype(np.int32))
            zs.append(z_cam + bias)
            cs.append(cols_a)
        u_i = np.concatenate(us)
        v_i = np.concatenate(vs)
        z_i = np.concatenate(zs)
        c_i = np.concatenate(cs)
        inside = (u_i >= 0) & (v_i >= 0) & (u_i < w) & (v_i < h)
        u_i, v_i, z_i, c_i = u_i[inside], v_i[inside], z_i[inside], c_i[inside]
        if len(z_i) == 0:
            return rgb, mask
        order = np.argsort(z_i)  # near first
        u_i, v_i, c_i = u_i[order], v_i[order], c_i[order]
        lin = v_i * w + u_i
        _, first = np.unique(lin, return_index=True)
        rgb[v_i[first], u_i[first]] = c_i[first].astype(np.uint8)
        mask[v_i[first], u_i[first]] = 255
        return _dilate_guide(rgb, mask, iters=3)


def _dilate_guide(rgb: np.ndarray, mask: np.ndarray, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Spread cache colors into empty pixels so the guide is dense enough to condition on."""
    out = rgb.copy()
    known = mask > 0
    h, w = known.shape
    for _ in range(iters):
        acc = np.zeros((*known.shape, 3), np.float32)
        weight = np.zeros(known.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src_m = np.roll(np.roll(known, dy, axis=0), dx, axis=1)
            src = np.roll(np.roll(out, dy, axis=0), dx, axis=1)
            if dy > 0:
                src_m[:dy] = False
            elif dy < 0:
                src_m[dy:] = False
            if dx > 0:
                src_m[:, :dx] = False
            elif dx < 0:
                src_m[:, dx:] = False
            acc += src.astype(np.float32) * src_m[..., None]
            weight += src_m
        fill = (~known) & (weight > 0)
        out[fill] = (acc[fill] / weight[fill, None]).astype(np.uint8)
        known = known | fill
    filled = np.where(known, np.uint8(255), np.uint8(0))
    return out, filled


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = pred.astype(np.float32)
    b = gt.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
        b = b / 255.0
    if mask is not None:
        m = mask > 0
        if not np.any(m):
            return 0.0
        a = a[m]
        b = b[m]
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-10:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def perceptual(pred: np.ndarray, gt: np.ndarray) -> float:
    """Downsampled L1. Lower is better. Stand-in for LPIPS with no extra dependency."""
    a = pred.astype(np.float32)
    b = gt.astype(np.float32)
    if a.max() > 1.5:
        a /= 255.0
        b /= 255.0
    h, w = a.shape[:2]
    h2, w2 = h // 4 * 4, w // 4 * 4
    a = a[:h2, :w2]
    b = b[:h2, :w2]

    def pool(x):
        return x.reshape(h2 // 4, 4, w2 // 4, 4, 3).mean(axis=(1, 3))

    return float(np.mean(np.abs(pool(a) - pool(b))))
