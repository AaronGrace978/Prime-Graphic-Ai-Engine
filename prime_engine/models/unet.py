"""Action-conditioned UNet denoiser with optional Solidify memory inputs.

Inputs (all images in [-1, 1], NCHW):
  x_t        noisy target frame                        [B, 3, H, W]
  context    last K frames (channel-concatenated)      [B, K, 3, H, W]
  actions    last K actions (int)                      [B, K]
  aug_level  noise-augmentation level of context/mem   [B]
  mem        M retrieved memory views                  [B, M, 3, H, W]
  mem_pose   relative pose features per memory slot    [B, M, P]   (last feature = valid flag)

Memory views are concatenated as extra input channels together with their
relative pose broadcast as constant planes, and the pose features are also
summed into the global FiLM conditioning vector. Invalid slots are zeros, so
"harness off" is the same network with every slot marked invalid.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def group_norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(32, ch // 4) if ch >= 8 else 1, ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = group_norm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(cond_dim, 2 * cout)
        self.norm2 = group_norm(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, c):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = group_norm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).unbind(1)
        out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(out.transpose(-1, -2).reshape(b, c, h, w))


class WorldUNet(nn.Module):
    def __init__(self, num_actions: int, context_frames: int = 4, memory_slots: int = 2, pose_dim: int = 5,
                 base_ch: int = 64, ch_mults=(1, 2, 2), num_res_blocks: int = 2, attn_levels=(2,),
                 cond_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.K, self.M, self.P = context_frames, memory_slots, pose_dim
        in_ch = 3 + 3 * context_frames + memory_slots * (3 + pose_dim)
        self.time_mlp = nn.Sequential(nn.Linear(base_ch, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.aug_mlp = nn.Sequential(nn.Linear(base_ch, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.action_emb = nn.Embedding(num_actions, cond_dim // 4)
        self.action_mlp = nn.Sequential(nn.Linear(context_frames * (cond_dim // 4), cond_dim), nn.SiLU(),
                                        nn.Linear(cond_dim, cond_dim))
        self.mem_slot_emb = nn.Parameter(torch.zeros(max(memory_slots, 1), pose_dim))
        self.mem_mlp = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.base_ch = base_ch

        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down = nn.ModuleList()
        chans = [base_ch]
        ch = base_ch
        for lvl, mult in enumerate(ch_mults):
            for _ in range(num_res_blocks):
                blk = nn.ModuleList([ResBlock(ch, base_ch * mult, cond_dim, dropout)])
                ch = base_ch * mult
                if lvl in attn_levels:
                    blk.append(SelfAttention(ch))
                self.down.append(blk)
                chans.append(ch)
            if lvl != len(ch_mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                chans.append(ch)
        self.mid = nn.ModuleList([ResBlock(ch, ch, cond_dim, dropout), SelfAttention(ch),
                                  ResBlock(ch, ch, cond_dim, dropout)])
        self.up = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(ch_mults))):
            for i in range(num_res_blocks + 1):
                blk = nn.ModuleList([ResBlock(ch + chans.pop(), base_ch * mult, cond_dim, dropout)])
                ch = base_ch * mult
                if lvl in attn_levels:
                    blk.append(SelfAttention(ch))
                if lvl != 0 and i == num_res_blocks:
                    blk.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blk.append(nn.Conv2d(ch, ch, 3, padding=1))
                self.up.append(blk)
        self.out = nn.Sequential(group_norm(ch), nn.SiLU(), nn.Conv2d(ch, 3, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x_t, t, context, actions, aug_level, mem=None, mem_pose=None):
        b, _, h, w = x_t.shape
        if mem is None:
            mem = x_t.new_zeros(b, self.M, 3, h, w)
            mem_pose = x_t.new_zeros(b, self.M, self.P)
        c = self.time_mlp(sinusoidal(t * 1000, self.base_ch)) + self.aug_mlp(sinusoidal(aug_level * 1000, self.base_ch))
        c = c + self.action_mlp(self.action_emb(actions).flatten(1))
        if self.M:
            valid = mem_pose[..., -1:]
            c = c + (self.mem_mlp(mem_pose + self.mem_slot_emb[None] * valid) * valid).sum(1)
            pose_planes = mem_pose[..., None, None].expand(b, self.M, self.P, h, w)
            mem_in = torch.cat([mem, pose_planes], dim=2).flatten(1, 2)
            inp = torch.cat([x_t, context.flatten(1, 2), mem_in], dim=1)
        else:
            inp = torch.cat([x_t, context.flatten(1, 2)], dim=1)

        hs = [self.stem(inp)]
        x = hs[-1]
        for blk in self.down:
            for layer in blk:
                x = layer(x, c) if isinstance(layer, ResBlock) else layer(x)
            hs.append(x)
        for layer in self.mid:
            x = layer(x, c) if isinstance(layer, ResBlock) else layer(x)
        for blk in self.up:
            x = torch.cat([x, hs.pop()], dim=1)
            for layer in blk:
                x = layer(x, c) if isinstance(layer, ResBlock) else layer(x)
        return self.out(x)
