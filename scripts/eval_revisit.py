#!/usr/bin/env python
"""Run the revisit ("The Return") benchmark with and without the Solidify harness."""
import argparse
import dataclasses
import json
import os

import torch

from prime_engine.eval import comparison_gif, revisit_figure, run_revisit_benchmark
from prime_engine.training import load_checkpoint, pick_device


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/cpu_smoke/model.pt")
    ap.add_argument("--worlds", type=int, default=0, help="override eval.worlds")
    ap.add_argument("--away", default="1", help="comma-separated rooms to visit before returning, e.g. 1,3")
    ap.add_argument("--out", default="", help="output dir (default: next to the checkpoint)")
    ap.add_argument("--gif-worlds", type=int, default=2)
    ap.add_argument("--pose-noise", action="store_true", help="also run Solidify + pose-seeded noise")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    cfg, model = load_checkpoint(args.ckpt, device)
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(out, exist_ok=True)
    extra = {}
    if args.pose_noise:
        extra["solidify_pose_noise"] = dataclasses.replace(cfg.harness, pose_seeded_noise=True)
    res = run_revisit_benchmark(model, cfg.world, cfg.harness, n_worlds=args.worlds or cfg.eval.worlds,
                                sample_steps=cfg.eval.sample_steps, aug_level=cfg.eval.aug_level,
                                seed=cfg.eval.seed, device=device,
                                away=tuple(int(r) for r in args.away.split(",")), extra_modes=extra)
    summary = {m: r["metrics"] for m, r in res["modes"].items()}
    summary["_meta"] = {"ckpt": args.ckpt, "worlds": len(res["episodes"]), "away": args.away,
                        "frames_per_world": int(len(res["episodes"][0].poses)),
                        "cells_world0_solidify": res["modes"]["solidify"]["cells_world0"]}
    with open(os.path.join(out, "revisit_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({m: summary[m] for m in res["modes"]}, indent=2))
    for w in range(min(args.gif_worlds, len(res["episodes"]))):
        print("wrote", comparison_gif(res, w, os.path.join(out, f"revisit_world{w}.gif")))
        print("wrote", revisit_figure(res, w, os.path.join(out, f"revisit_world{w}.png")))


if __name__ == "__main__":
    main()
