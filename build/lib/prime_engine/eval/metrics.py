"""Image metrics for revisit consistency."""
from __future__ import annotations

import numpy as np

from ..world.raycaster import PALETTE_HUES


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def _hue_sat(rgb: np.ndarray):
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    delta = mx - mn
    sat = np.where(mx > 0, delta / np.maximum(mx, 1e-6), 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    d = np.maximum(delta, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6.0
    return h, sat


def wall_color_class(rgb: np.ndarray, mask: np.ndarray) -> int:
    """Palette index of the dominant (saturation-weighted circular mean) hue inside mask; -1 if no colour."""
    if mask.sum() == 0:
        return -1
    h, s = _hue_sat(rgb)
    h, s = h[mask], s[mask]
    if s.mean() < 0.15:
        return -1
    ang = 2 * np.pi * h
    mean = np.arctan2((s * np.sin(ang)).sum(), (s * np.cos(ang)).sum()) / (2 * np.pi) % 1.0
    d = np.abs((PALETTE_HUES - mean + 0.5) % 1.0 - 0.5)
    return int(np.argmin(d))


def room_identity(gen: np.ndarray, instance: np.ndarray, room_colors: np.ndarray):
    """Does the generated frame paint the dominant visible room in that room's true colour?

    Uses the ground-truth instance map only to pick *where* to look (pixels of the
    most-visible room's walls) and which colour is correct. Returns (pred, true) palette ids,
    or None when no room wall dominates the view.
    """
    walls = instance[instance >= 0]
    if walls.size < 0.1 * instance.size:
        return None
    room = int(np.bincount(walls).argmax())
    mask = instance == room
    return wall_color_class(gen, mask), int(room_colors[room])
