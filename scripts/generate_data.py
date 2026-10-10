#!/usr/bin/env python
"""Collect a multi-room trajectory dataset (rgb, depth, poses, inputs, cell state)."""
import argparse

from prime_engine.config import load_config
from prime_engine.world import generate_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cpu_smoke.yaml")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.episodes=50")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    d = generate_dataset(cfg.world, cfg.data.episodes, cfg.data.length, seed=cfg.data.seed,
                         workers=cfg.data.workers or None, out_path=cfg.data.path)
    print(f"wrote {cfg.data.path}: rgb {d['rgb'].shape}, portal crossings/episode "
          f"{d['portal_crossed'].sum(1).mean():.1f}")


if __name__ == "__main__":
    main()
