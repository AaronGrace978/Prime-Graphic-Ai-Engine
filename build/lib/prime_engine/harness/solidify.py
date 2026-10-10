"""Solidify v0: cell-structured, pose-keyed memory of rendered views.

Follows the engine plan §3.2 data model, minus the 3D splat cache:

  Cell { id, bounds, neighbors, seed, keyframes{pose_key -> view}, state, status, version, state_hash }

* Cells come from the engine (Anchored mode: room ids from level data). If the
  engine gives no cell id, a coarse hash-grid cell is used (Dream mode).
* Every generated frame is offered to its cell under a discretised pose key
  (position bin x heading bin). With ``first_write`` the first view of a place
  is canonical; later visits retrieve it instead of their own re-generations.
* A cell becomes SOLID once its heading coverage passes ``lock_coverage``
  (v0 proxy for "surface coverage from >= 2 views"); SOLID cells are read-only.
  ``set_state`` applies an edit: bumps version and marks the cell DIRTY so it
  re-forms and re-locks.
* ``state_hash = H(keyframe content hashes, state record, runner_id)`` -- same
  hash means the room will be rematerialised from the same canonical data.
* Retrieval searches the current cell plus its portal neighbours, ranked by the
  same pose distance used at training time.

Payloads are opaque (pixels today, latents later).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..world.schema import stable_hash
from .geometry import POSE_FEAT_DIM, relative_pose_features, view_distance


class CellStatus(str, Enum):
    UNSEEN = "UNSEEN"
    FORMING = "FORMING"
    SOLID = "SOLID"
    DIRTY = "DIRTY"


@dataclass
class Keyframe:
    pose: np.ndarray
    payload: Any
    step: int
    content_hash: int
    hits: int = 0


@dataclass
class Cell:
    id: Any
    seed: int
    bounds: tuple | None = None
    neighbors: list = field(default_factory=list)
    keyframes: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    status: CellStatus = CellStatus.UNSEEN
    version: int = 0

    def heading_coverage(self, heading_bins: int) -> float:
        return len({k[2] for k in self.keyframes}) / heading_bins


@dataclass
class SolidifyConfig:
    pos_cell: float = 0.5          # world units per keyframe position bin
    heading_bins: int = 24         # 15 degree bins
    policy: str = "first_write"    # "first_write" (solidify) | "overwrite" (latest wins, no locking)
    angle_weight: float = 1.5      # must match training retrieval
    max_distance: float = 3.0      # views farther than this are not returned
    exclude_recent: int = 4        # skip views still inside the model's context window (= context_frames)
    lock_coverage: float = 0.9     # heading coverage at which a cell turns SOLID
    neighbor_search: bool = True   # retrieve from current cell + portal neighbours only
    dream_cell_size: float = 7.0   # hash-grid cell size when the engine provides no cell id
    pose_seeded_noise: bool = False  # Exact-Replay flavour: initial noise = H(cell seed, quantised pose)


def _content_hash(payload) -> int:
    try:
        import torch
        if isinstance(payload, torch.Tensor):
            payload = ((payload.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return stable_hash(np.asarray(payload))


class SolidifyStore:
    def __init__(self, cfg: SolidifyConfig | None = None, world_seed: int = 0, runner_id: str = "v0"):
        self.cfg = cfg or SolidifyConfig()
        self.world_seed = world_seed
        self.runner_id = runner_id
        self.cells: dict[Any, Cell] = {}

    # ---------------------------------------------------------------- cells
    def register_cells(self, graph: dict):
        """graph: {cell_id: {"bounds": ..., "neighbors": [...], "state": {...}}} from the engine."""
        for cid, info in graph.items():
            cell = self._cell(cid)
            cell.bounds = info.get("bounds")
            cell.neighbors = list(info.get("neighbors", []))
            cell.state = dict(info.get("state", {}))

    def _cell(self, cid) -> Cell:
        if cid not in self.cells:
            self.cells[cid] = Cell(id=cid, seed=stable_hash("cell-seed", self.world_seed, cid))
        return self.cells[cid]

    def cell_id_for(self, pose, cell_id=None):
        if cell_id is not None and cell_id >= 0:
            return int(cell_id)
        if cell_id is not None:  # engine says "doorway / between cells"
            return None
        s = self.cfg.dream_cell_size
        return f"g{int(np.floor(pose[0] / s))}_{int(np.floor(pose[1] / s))}"

    def pose_key(self, pose) -> tuple[int, int, int]:
        x, y, th = pose
        hb = int(np.round((th % (2 * np.pi)) / (2 * np.pi) * self.cfg.heading_bins)) % self.cfg.heading_bins
        return int(np.floor(x / self.cfg.pos_cell)), int(np.floor(y / self.cfg.pos_cell)), hb

    def state_hash(self, cid) -> int:
        c = self.cells[cid]
        kf = sorted((k, v.content_hash) for k, v in c.keyframes.items())
        return stable_hash(kf, c.state, self.runner_id)

    def set_state(self, cid, **state):
        """Apply an authoritative state edit (door opened, prop moved...)."""
        c = self._cell(cid)
        c.state.update(state)
        c.version += 1
        c.keyframes.clear()  # v0 has no per-object regions: the whole cell re-forms
        c.status = CellStatus.DIRTY

    # ---------------------------------------------------------- write/read
    def write(self, pose, payload, step: int, cell_id=None) -> bool:
        """Offer a rendered view to its cell. Returns True if the store changed."""
        cid = self.cell_id_for(pose, cell_id)
        if cid is None:
            return False
        c = self._cell(cid)
        if c.status == CellStatus.SOLID and self.cfg.policy == "first_write":
            return False
        k = self.pose_key(pose)
        if k in c.keyframes and self.cfg.policy == "first_write":
            return False
        c.keyframes[k] = Keyframe(np.asarray(pose, dtype=np.float64).copy(), payload, step, _content_hash(payload))
        if c.status in (CellStatus.UNSEEN, CellStatus.DIRTY):
            c.status = CellStatus.FORMING
        if (self.cfg.policy == "first_write" and c.status == CellStatus.FORMING
                and c.heading_coverage(self.cfg.heading_bins) >= self.cfg.lock_coverage):
            c.status = CellStatus.SOLID
            c.version += 1
        return True

    def _candidate_cells(self, cid):
        if cid is None or not self.cfg.neighbor_search or cid not in self.cells:
            return list(self.cells.values())
        ids = [cid, *self.cells[cid].neighbors]
        return [self.cells[i] for i in ids if i in self.cells]

    def retrieve(self, query_pose, k: int, current_step: int | None = None, cell_id=None):
        """Up to k closest stored views as [(Keyframe, distance)], skipping ones still in context."""
        cid = self.cell_id_for(query_pose, cell_id)
        items = [kf for c in self._candidate_cells(cid) for kf in c.keyframes.values()]
        if current_step is not None and self.cfg.exclude_recent:
            items = [e for e in items if e.step < current_step - self.cfg.exclude_recent]
        if not items:
            return []
        d = view_distance(np.asarray(query_pose, dtype=np.float64), np.stack([e.pose for e in items]),
                          self.cfg.angle_weight)
        out = [(items[i], float(d[i])) for i in np.argsort(d)[:k] if d[i] <= self.cfg.max_distance]
        for e, _ in out:
            e.hits += 1
        return out

    def conditioning(self, query_pose, k: int, current_step: int | None = None, cell_id=None):
        """Retrieve and pack as (payloads list, pose features [k, POSE_FEAT_DIM]); empty slots are invalid."""
        hits = self.retrieve(query_pose, k, current_step, cell_id)
        feats = np.zeros((k, POSE_FEAT_DIM), np.float32)
        payloads = [None] * k
        for i, (e, _) in enumerate(hits):
            payloads[i] = e.payload
            feats[i] = relative_pose_features(np.asarray(query_pose), e.pose[None])[0]
        return payloads, feats

    def noise_seed(self, pose, cell_id=None) -> int:
        cid = self.cell_id_for(pose, cell_id)
        seed = self.cells[cid].seed if cid in self.cells else self.world_seed
        return stable_hash("noise", seed, self.pose_key(pose)) & 0x7FFFFFFFFFFFFFFF

    def summary(self) -> list[dict]:
        return [{"cell": c.id, "status": c.status.value, "version": c.version, "keyframes": len(c.keyframes),
                 "coverage": round(c.heading_coverage(self.cfg.heading_bins), 2),
                 "state": c.state, "state_hash": f"{self.state_hash(c.id):016x}"} for c in self.cells.values()]

    def __len__(self):
        return sum(len(c.keyframes) for c in self.cells.values())

    def clear(self):
        self.cells.clear()
