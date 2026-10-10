"""Typed experiment config loaded from YAML with dotted CLI overrides."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import yaml

from .harness.solidify import SolidifyConfig
from .world.raycaster import WorldConfig


@dataclass
class DataConfig:
    path: str = "data/rooms.npz"
    episodes: int = 400
    length: int = 200
    seed: int = 0
    workers: int = 0  # 0 = auto


@dataclass
class ModelConfig:
    context_frames: int = 4
    memory_slots: int = 2
    base_ch: int = 64
    ch_mults: tuple = (1, 2, 2)
    num_res_blocks: int = 2
    attn_levels: tuple = (2,)
    cond_dim: int = 256
    dropout: float = 0.0


@dataclass
class TrainConfig:
    out_dir: str = "runs/default"
    steps: int = 20000
    batch_size: int = 64
    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup: int = 500
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    max_aug: float = 0.3       # max noise level added to context/memory frames (anti-drift)
    mem_drop: float = 0.25     # prob. of dropping all memory slots (teaches harness-off mode)
    slot_drop: float = 0.15    # prob. of dropping each memory slot independently
    amp: bool = True
    compile: bool = False
    log_every: int = 100
    ckpt_every: int = 2000
    seed: int = 0
    device: str = "auto"


@dataclass
class EvalConfig:
    worlds: int = 16
    sample_steps: int = 16
    aug_level: float = 0.05
    seed: int = 1234


@dataclass
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    harness: SolidifyConfig = field(default_factory=SolidifyConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _apply(dc, values: dict):
    for k, v in values.items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key {type(dc).__name__}.{k}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur):
            _apply(cur, v)
        else:
            if isinstance(cur, tuple) and isinstance(v, list):
                v = tuple(v)
            setattr(dc, k, v)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    cfg = Config()
    if path:
        with open(path) as f:
            _apply(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        *parents, leaf = key.split(".")
        node = {leaf: yaml.safe_load(val)}
        for p in reversed(parents):
            node = {p: node}
        _apply(cfg, node)
    cfg.harness.exclude_recent = cfg.model.context_frames
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def config_from_dict(d: dict) -> Config:
    cfg = Config()
    _apply(cfg, d)
    return cfg
