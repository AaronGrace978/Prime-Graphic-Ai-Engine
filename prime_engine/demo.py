"""Run "The Return": topple a vase, leave through three rooms, come back.

The on-screen state hash after the return matches the hash locked when the
vase was toppled. A short-context rollout drifts. Solidify does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prime_engine.model import consistency_guard, sample, train_model
from prime_engine.solidify import perceptual, psnr
from prime_engine.world import capture_the_return


def _blank_like(frame) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.rgb.shape[:2]
    return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint8)


def evaluate(cap, info, model, sched, device) -> dict:
    frames = cap.frames
    ret = frames[info["return_index"]]
    gt = ret.rgb
    guide, mask = ret.guide_rgb, ret.guide_mask
    # Short context: the last frame that was still outside the foyer.
    outside = [f for f in frames[info["locked_index"] : info["return_index"]] if f.cell_id != 0]
    stray = outside[-1].rgb if outside else frames[info["locked_index"]].rgb
    retrieved = cap.solidify.retrieve(0, float(ret.pose[0]), float(ret.pose[1]), float(ret.pose[3]))
    memory = retrieved["rgb"] if retrieved is not None else guide

    empty_g, empty_m = _blank_like(ret)
    diffusion_only = sample(model, sched, stray, empty_g, empty_m, ret.action_keys, start_t=sched.steps - 1, device=device)
    diffusion_solidify = sample(
        model, sched, memory, guide, mask, ret.action_keys, start_t=max(8, sched.steps // 5), device=device
    )
    guarded = consistency_guard(diffusion_solidify, guide, mask)

    def pack(pred):
        return {"psnr": round(psnr(pred, gt), 2), "perceptual": round(perceptual(pred, gt), 4)}

    metrics = {
        "state_hash_before_edit": info["hash_before"],
        "state_hash_locked": info["hash_locked"],
        "state_hash_return": info["hash_return"],
        "hash_matches_on_return": info["hash_locked"] == info["hash_return"],
        "hash_changed_on_edit": info["hash_before"] != info["hash_locked"],
        "frames": len(frames),
        "foyer_status": cap.solidify.cells[0].status,
        "foyer_voxels": len(cap.solidify.cells[0].voxels),
        "guide": pack(guide),
        "short_context": pack(stray),
        "diffusion_only": pack(diffusion_only),
        "diffusion_solidify": pack(diffusion_solidify),
        "solidify_guard": pack(guarded),
    }
    images = {
        "gt": gt,
        "guide": guide,
        "short_context": stray,
        "diffusion_only": diffusion_only,
        "diffusion_solidify": diffusion_solidify,
        "solidify_guard": guarded,
    }
    return metrics, images


def save_strip(images: dict, path: Path) -> None:
    keys = ("gt", "guide", "short_context", "diffusion_only", "solidify_guard")
    labels = {
        "gt": "return (truth)",
        "guide": "solidify cache",
        "short_context": "short context",
        "diffusion_only": "model, no memory",
        "solidify_guard": "model + solidify",
    }
    tiles = [images[k] for k in keys]
    h, w = tiles[0].shape[:2]
    scale = 3
    pad = 18
    canvas = Image.new("RGB", (scale * w * len(tiles), scale * h + pad), (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    for i, (key, tile) in enumerate(zip(keys, tiles)):
        im = Image.fromarray(tile).resize((w * scale, h * scale), Image.Resampling.NEAREST)
        canvas.paste(im, (i * w * scale, pad))
        draw.text((i * w * scale + 4, 2), labels[key], fill=(240, 236, 220))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_gif(cap, info, path: Path) -> None:
    # First foyer scan, then the return frame held so the room is visibly the same.
    foyer = [f.rgb for f in cap.frames if f.cell_id == 0][:24]
    ret = cap.frames[info["return_index"]].rgb
    frames = foyer + [ret] * 8
    images = [Image.fromarray(f).resize((f.shape[1] * 3, f.shape[0] * 3), Image.Resampling.NEAREST) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=90, loop=0)


def run(steps: int = 80, seed: int = 0, out_dir: str = "media", width: int = 64, height: int = 48) -> dict:
    cap, info = capture_the_return(width=width, height=height, seed=seed)
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sched, last_loss, nparams = train_model(
        cap.frames, steps=steps, seed=seed, device=str(device), upto=info["locked_index"]
    )
    sched.to(device)
    metrics, images = evaluate(cap, info, model, sched, device)
    metrics["train_steps"] = steps
    metrics["train_loss_last10"] = round(last_loss, 4)
    metrics["parameters"] = nparams
    metrics["device"] = str(device)
    out = Path(out_dir)
    save_strip(images, out / "return_strip.png")
    save_gif(cap, info, out / "return.gif")
    (out / "smoke_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Prime Engine Phase 0 — The Return")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="media")
    args = parser.parse_args()
    metrics = run(steps=args.steps, seed=args.seed, out_dir=args.out)
    print(json.dumps(metrics, indent=2))
    if not metrics["hash_matches_on_return"]:
        raise SystemExit("state hash changed on return")
    if metrics["solidify_guard"]["psnr"] <= metrics["diffusion_only"]["psnr"]:
        raise SystemExit("Solidify did not beat the no-memory rollout")


if __name__ == "__main__":
    main()
