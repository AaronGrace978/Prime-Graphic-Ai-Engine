"""Side-by-side figures and GIFs for revisit rollouts."""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from ..harness.geometry import view_distance

LABELS = {"gt": "ground truth", "no_harness": "no harness", "solidify": "Solidify"}


def _up(img: np.ndarray, scale: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST))


def _label(img: np.ndarray, text: str, pad: int = 14) -> np.ndarray:
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + pad), (12, 12, 16))
    canvas.paste(Image.fromarray(img), (0, pad))
    ImageDraw.Draw(canvas).text((3, 1), text, fill=(235, 235, 235))
    return np.asarray(canvas)


def map_image(world, poses: np.ndarray | None = None, scale: int = 16) -> np.ndarray:
    """Top-down view: walls dark, floor tinted by room colour, optional trajectory in white."""
    from ..world.raycaster import PALETTE_RGB
    g = world.grid
    img = np.full(g.shape + (3,), 40, np.uint8)
    for r in range(world.n_rooms):
        img[world.room_id == r] = (PALETTE_RGB[world.room_colors[r]] * 200).astype(np.uint8)
    img[(g == 0) & (world.room_id < 0)] = 140
    img = _up(img, scale).copy()
    if poses is not None:
        canvas = Image.fromarray(img)
        d = ImageDraw.Draw(canvas)
        pts = [(float(p[0] * scale), float(p[1] * scale)) for p in poses]
        d.line(pts, fill=(255, 255, 255), width=2)
        x, y, th = poses[-1]
        d.line([pts[-1], ((x + 0.6 * np.cos(th)) * scale, (y + 0.6 * np.sin(th)) * scale)], fill=(255, 40, 40), width=3)
        img = np.asarray(canvas)
    return img


def comparison_gif(results: dict, world: int, path: str, scale: int = 4, fps: int = 8,
                   modes=("no_harness", "solidify")):
    ep = results["episodes"][world]
    rows = []
    for t in range(len(ep.poses)):
        tiles = [_label(_up(ep.rgb[t], scale), "ground truth")]
        for m in modes:
            tiles.append(_label(_up(results["modes"][m]["frames"][world, t], scale), LABELS.get(m, m)))
        sep = np.full((tiles[0].shape[0], 4, 3), 12, np.uint8)
        row = np.concatenate(sum([[tl, sep] for tl in tiles], [])[:-1], axis=1)
        rows.append(_label(row, f"t={t:03d}  phase={ep.phases[t]}  cell={ep.cells[t]}"))
    imageio.mimsave(path, rows, duration=1000 / fps, loop=0)
    return path


def revisit_figure(results: dict, world: int, path: str, n_cols: int = 6, scale: int = 4,
                   modes=("no_harness", "solidify")):
    """Rows: (first visit, revisit) x (ground truth, each mode); columns: matched headings."""
    ep = results["episodes"][world]
    ph = np.array(ep.phases)
    a1, a2 = np.where(ph == "A1")[0], np.where(ph == "A2")[0]
    cols = a2[np.linspace(0, len(a2) - 1, n_cols).astype(int)]
    match = [a1[np.argmin(view_distance(ep.poses[t], ep.poses[a1]))] for t in cols]
    sources = [("ground truth", ep.rgb)] + [(LABELS.get(m, m), results["modes"][m]["frames"][world]) for m in modes]
    rows = []
    for visit, idx in (("first visit", match), ("revisit", cols)):
        for name, frames in sources:
            tiles = [_up(frames[t], scale) for t in idx]
            sep = np.full((tiles[0].shape[0], 2, 3), 12, np.uint8)
            rows.append(_label(np.concatenate(sum([[tl, sep] for tl in tiles], [])[:-1], axis=1),
                               f"{visit} - {name}"))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    return path
