"""Training loop for the action-conditioned diffusion world model."""
from __future__ import annotations

import copy
import json
import math
import os
import time

import torch

from ..config import Config, config_from_dict, config_to_dict
from ..models import Diffusion, WorldUNet
from ..world.dataset import generate_dataset, load_dataset
from ..world.raycaster import NUM_ACTIONS, RoomWorld
from .data import TrajectoryBatcher


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: Config) -> WorldUNet:
    m = cfg.model
    return WorldUNet(NUM_ACTIONS, context_frames=m.context_frames, memory_slots=m.memory_slots, base_ch=m.base_ch,
                     ch_mults=tuple(m.ch_mults), num_res_blocks=m.num_res_blocks, attn_levels=tuple(m.attn_levels),
                     cond_dim=m.cond_dim, dropout=m.dropout)


def ensure_dataset(cfg: Config) -> dict:
    if os.path.exists(cfg.data.path):
        data = load_dataset(cfg.data.path)
        rgb = data.get("rgb")
        if rgb is not None and rgb.shape[2:4] == (cfg.world.height, cfg.world.width) and len(rgb) >= cfg.data.episodes:
            return data
        print(f"[data] {cfg.data.path} does not match config; regenerating")
    print(f"[data] generating {cfg.data.episodes} episodes x {cfg.data.length} steps at "
          f"{cfg.world.width}x{cfg.world.height}")
    return generate_dataset(cfg.world, cfg.data.episodes, cfg.data.length, seed=cfg.data.seed,
                            workers=cfg.data.workers or None, out_path=cfg.data.path)


def save_checkpoint(path: str, cfg: Config, model, ema, step: int, extra: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": config_to_dict(cfg), "model": model.state_dict(), "ema": ema.state_dict(), "step": step,
                **(extra or {})}, path)


def load_checkpoint(path: str, device="cpu", use_ema: bool = True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    return cfg, model.to(device).eval()


def train(cfg: Config, data: dict | None = None, progress: bool = True) -> str:
    t_cfg = cfg.train
    device = pick_device(t_cfg.device)
    torch.manual_seed(t_cfg.seed)
    data = data if data is not None else ensure_dataset(cfg)
    vis_world = RoomWorld(cfg.world, seed=0)  # visibility depends on geometry only, not on room colours
    batcher = TrajectoryBatcher(data, cfg.model.context_frames, cfg.model.memory_slots, cfg.harness,
                                vis_fn=vis_world.visibility, device=device)
    print(f"[data] {batcher.N} episodes x {batcher.T1} frames; memory coverage {batcher.memory_coverage:.2f}, "
          f"revisit targets {batcher.revisit_share:.2f} (sampled at {t_cfg.revisit_frac:.2f})")

    model = build_model(cfg).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.2f}M params on {device}")
    diffusion = Diffusion()
    opt = torch.optim.AdamW(model.parameters(), lr=t_cfg.lr, weight_decay=t_cfg.weight_decay, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, t_cfg.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / t_cfg.steps)))))

    use_amp = t_cfg.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    fwd = torch.compile(model) if t_cfg.compile else model
    gen = torch.Generator().manual_seed(t_cfg.seed)
    os.makedirs(t_cfg.out_dir, exist_ok=True)
    log_path = os.path.join(t_cfg.out_dir, "train_log.jsonl")
    ckpt_path = os.path.join(t_cfg.out_dir, "model.pt")

    t0, running = time.time(), 0.0
    for step in range(1, t_cfg.steps + 1):
        b = batcher.sample(t_cfg.batch_size, gen, t_cfg.max_aug, t_cfg.mem_drop, t_cfg.slot_drop, t_cfg.revisit_frac)
        target = b.pop("target")
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            loss = diffusion.loss(fwd, target, **b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        decay = min(t_cfg.ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        running += loss.item()
        if step % t_cfg.log_every == 0 or step == t_cfg.steps:
            n = t_cfg.log_every if step % t_cfg.log_every == 0 else step % t_cfg.log_every
            rec = {"step": step, "loss": running / n, "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
            running = 0.0
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if progress:
                print(f"[train] step {step}/{t_cfg.steps} loss {rec['loss']:.4f} "
                      f"({rec['elapsed_s'] / step:.3f}s/step)", flush=True)
        if step % t_cfg.ckpt_every == 0 or step == t_cfg.steps:
            save_checkpoint(ckpt_path, cfg, model, ema, step)
    return ckpt_path
