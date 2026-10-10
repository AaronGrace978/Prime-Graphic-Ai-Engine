#!/usr/bin/env python
"""Train the action-conditioned diffusion world model (generates the dataset if missing)."""
import argparse

import torch

from prime_engine.config import load_config
from prime_engine.training import train


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cpu_smoke.yaml")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for torch (0 = default)")
    ap.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.steps=200")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config, args.overrides)
    print("checkpoint:", train(cfg))


if __name__ == "__main__":
    main()
