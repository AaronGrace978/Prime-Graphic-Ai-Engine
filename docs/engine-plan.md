# Prime Graphic AI Engine — Architecture Plan & Roadmap

Repo: [AaronGrace978/Prime-Graphic-Ai-Engine](https://github.com/AaronGrace978/Prime-Graphic-Ai-Engine) · Target dev rig: RTX 5060 Ti 16 GB + 32 GB RAM · Plan date: Oct 2026

---

## TL;DR — the decisions

1. **The engine remembers; the model renders.** Game logic, camera pose, collision, and persistent world state live in a deterministic native engine. The diffusion model is a *neural renderer* conditioned on that state. This one principle is what makes "Solidify" (rooms that rematerialize exactly) achievable, and it is what separates a commercial engine from a research demo.
2. **"13B" is real, but it is not the real-time renderer.** A 13–14B video model on a 16 GB card runs at well under 1 FPS at 480p even at 4-bit. Use it as the **Architect** tier: world authoring, keyframes, style, distillation teacher, and async refinement. The real-time **Runner** is a 0.5–1.3B causal, few-step-distilled model.
3. **Recommended base models (all commercially usable):** Runner from **Wan2.1-T2V-1.3B** (Apache-2.0) via the Self-Forcing recipe (Apache-2.0). Architect is **Wan2.1/2.2 14B** (Apache-2.0). For research baselines and teachers, use **Matrix-Game 2.0** (MIT, 1.8B) and **Matrix-Game 3.0** (Apache-2.0, 5B, memory-augmented). Avoid the Hunyuan family for a global product, because its license excludes the EU, UK, and South Korea.
4. **Solidify = a streaming level system for generated worlds.** The world is split into cells joined by portals. Each cell has a 3D Gaussian/point cache, pose-indexed keyframes, a small authoritative state record, and a state hash. Generation is conditioned on renders of the cache plus retrieved keyframes. Cells lock once they are covered, and they stream between VRAM, RAM, and NVMe like Uncharted-era level chunks.
5. **Stack:** a native **C++20 + Vulkan** runtime with **TensorRT** (FP8/NVFP4) inference, CUDA↔Vulkan interop, and a **Dear ImGui** editor. **No Electron in the hot path.** A web or Electron "Studio" can come later as an optional authoring and dashboard shell over IPC.
6. **Training path:** Colab Pro+ covers Phases 0–1, which are cheap (tens to low hundreds of dollars). Move to rented multi-GPU (8×H100) for Runner distillation in Phase 2 (about $1–3k). Don't full-fine-tune 14B until funded; use LoRA.
7. **Data:** capture your own gameplay from an engine you control (Godot 4, MIT) with CC0 assets. Log RGB, depth, pose, actions, cell ID, and state hash. That gives clean IP plus the ground-truth depth and pose that Solidify needs.
8. **Head-turner demo, "The Return":** play on a consumer 16 GB GPU, knock something over, leave through three rooms, come back minutes later, and find the room identical (same state hash on screen). Show it split-screen against a baseline whose world drifts.

---

## 1. Model sizing reality

### 1.1 What a 13B model means on a 16 GB card

The dev card is an RTX 5060 Ti 16 GB (Blackwell, GDDR7, **448 GB/s**). Dense tensor throughput is roughly **~47 TFLOPS BF16**, **~95–190 TFLOPS FP8**, and **~190–380 TFLOPS FP4**; the range depends on accumulate mode, and GeForce halves FP32-accumulate rates. NVIDIA's "759 AI TOPS" figure is FP4 with sparsity. For comparison, an H100 has about 7.5× the bandwidth and 10–20× the dense tensor throughput.

| Model class | Precision | Weights only | Fits in 16 GB alongside a game? | Expected speed on 5060 Ti* |
|---|---|---|---|---|
| 13–14B DiT (Wan 14B, LTXV-13B, Cosmos-14B) | BF16 | 26–28 GB | No | — |
| | FP8 | 13–14 GB | Only alone, with block offload | ~0.1–0.5 FPS @480p, 4 steps |
| | NVFP4 / INT4 | 7–8 GB | Alone or during pauses | ~0.3–1 FPS @480p, 4 steps; seconds/frame at full quality |
| 5B (Wan2.2-TI2V-5B, Matrix-Game 3.0) | FP8 | ~5 GB | Tight | ~2–6 FPS @480p, few-step |
| 1.3–1.8B (Wan2.1-1.3B, Matrix-Game 2.0) | FP8 | ~1.5–2 GB | Yes | **~8–20 FPS @360p, 1–2 steps, TensorRT** |
| 0.4–0.5B (DIAMOND/Oasis class, "Runner-S") | FP8 | <0.6 GB | Yes | **~25–40 FPS @256–360p** |

\*These are back-of-envelope estimates scaled from published numbers, not measurements. Self-Forcing (1.3B, 832×480, 4 steps) runs **17 FPS on one H100**. Matrix-Game 3.0 (5B) reaches **40 FPS at 720p only on 8 DiT GPUs + 1 VAE GPU**. DIAMOND (381M) runs **~10 FPS on an RTX 3090**. Oasis 500M was shown at ~20 FPS on datacenter hardware. **Phase 0 must replace this table with measured numbers.**

Large text encoders are not counted above. Wan's umT5-XXL alone is ~11 GB in BF16. An action-conditioned model drops the text branch, as Matrix-Game did, and Architect jobs can precompute text embeddings.

**Verdict:** a 13B model will never be the per-frame renderer on this card. It is valuable as a teacher, an author, and a refiner. That is where Aaron's "13B" goes.

### 1.2 Recommended tiered design

```
            ┌──────────────────────── ENGINE (authoritative, deterministic) ───────────────────────┐
 input ───► │ controller/physics → camera pose · game state · cell graph · Solidify cache/streamer │
            └───────────────┬───────────────────────────────┬─────────────────────────────────────┘
                            │ guide render (RGB-D+mask),    │ async jobs (pauses / cloud / 2nd GPU)
                            │ retrieved keyframes, actions  ▼
                 ┌──────────▼──────────┐          ┌──────────────────────────┐
                 │ T1 RUNNER 0.5–1.3B  │          │ T2 ARCHITECT 13–14B      │
                 │ causal, 1–2 steps,  │◄─distill─│ keyframes, room authoring│
                 │ RGB-D output, FP8   │          │ style LoRAs, cache refine│
                 └──────────┬──────────┘          └──────────────────────────┘
                            ▼
                 T3 FINISHER: tiny VAE decoder → temporal 2× upscaler → depth reprojection to 60–120 Hz
                            ▼
                 native HUD/UI composited by engine (never generated)
```

| Tier | Role | Recommended base | License | Notes |
|---|---|---|---|---|
| **T1 Runner** | Real-time frames at 20–30 Hz, 360p internal | **Wan2.1-T2V-1.3B**, converted to causal + action/pose/guide-conditioned with the **Self-Forcing** recipe; later distilled to a ~0.5B **Runner-S** | Apache-2.0 / Apache-2.0 | Cleanest license and a documented real-time recipe. Predicts **RGB + depth** jointly (like Voyager) so it feeds Solidify. |
| T1 alt (research only) | Fast start / baseline | **Matrix-Game 2.0** (1.8B, action-conditioned, 25 FPS on datacenter GPU) | MIT | Trained on ~1200 h of Unreal **and GTA5** data, so it carries a provenance risk. Benchmark against it, don't ship it. |
| **T2 Architect** | Offline/async: author first-visit keyframes per cell, style LoRAs, teacher for distillation, refine caches while the player is elsewhere | **Wan2.1-14B / Wan2.2-A14B** | Apache-2.0 | The "13B" tier. |
| T2 alt | World-model teacher with actions | **LingBot-World** (28B MoE, ~14B active) or **Matrix-Game 3.0** (5B, Wan2.2-based, memory-augmented) | Apache-2.0 / Apache-2.0 | Good distillation teachers. Matrix-Game 3.0 includes "AAA game" data, so the same provenance caveat applies. |
| T2 alt | Physics-flavored world model | **NVIDIA Cosmos-Predict2.5 2B/14B** | NVIDIA Open Model License | Commercial use allowed. Requires a "Built on NVIDIA Cosmos" notice. Revocable-license terms; read before shipping. |
| T2 alt | 13B video | **LTX-Video 13B** | LTXV Open Weights License | Free for commercial use **only while company revenue < $10M/yr**, paid license above. OK for now; plan an exit. |
| ❌ Avoid | | HunyuanVideo 13B / 1.5, Hunyuan-GameCraft, HunyuanWorld-Voyager | Tencent community licenses | **Not licensed in the EU, UK, or South Korea**, plus MAU caps. Disqualifying for a global product. |
| **T3 Finisher** | Decode + upscale + frame pacing | Distilled tiny VAE decoder (Matrix-Game 3.0 showed decoder distillation is a major speedup); own 2× temporal upscaler trained on own engine data; depth-based reprojection | Own | Reprojection lets diffusion frames arrive at 20–30 Hz while the display updates camera motion at 60–120 Hz, like VR timewarp. |
| Depth (only if the model can't predict it) | Lifting frames to 3D | **Depth Anything V2 Small / Video-Depth-Anything Small** | Apache-2.0 | ⚠️ **Base/Large/Giant are CC-BY-NC-4.0 (non-commercial).** DUSt3R/MASt3R are also non-commercial (verify before use). Prefer training the Runner to output depth from engine ground truth. |

### 1.3 Runtime VRAM budget (16 GB, Runner at 360p)

| Item | Budget |
|---|---|
| Runner 1.3B FP8 (NVFP4 ≈ half) | 1.3–1.5 GB |
| KV cache: ~16 context frames + 4 retrieved memory frames (FP8 KV) | 1.5–3 GB |
| TensorRT workspace / activations | 1–2 GB |
| Tiny VAE decoder + upscaler | 0.3 GB |
| Solidify hot set (current + adjacent cells: splats + keyframe latents) | 0.5–1 GB |
| Vulkan swapchain, render targets, splat raster | 0.3–0.5 GB |
| OS/desktop overhead | 0.5–1 GB |
| **Total** | **~6–9 GB** (headroom for a Runner-L or 5B experiments) |

The Architect does **not** co-reside with the live Runner at FP8. It runs during natural pauses (doors, menus, "loading" corridors, which is exactly the Uncharted trick), on a second machine, or in the cloud.

---

## 2. Training & fine-tuning

### 2.1 Colab reality (Oct 2026)

| Plan | Price | What you get | Use it for |
|---|---|---|---|
| Free | $0 | T4 (15 GB, no BF16) when available, ≤12 h sessions, no guarantees | Smoke tests, toy inference only |
| Pay-as-you-go | $9.99 / 100 CU | Same GPUs as Pro | Top-ups |
| Pro | $9.99/mo, 100 CU | Premium GPUs, ~24 h sessions | Light experiments |
| **Pro+ (recommended Phases 0–1)** | **$49.99/mo, 600 CU** | Priority GPUs + **background execution up to 24 h** | Real training runs |

Measured burn rates (community-reported, Mar 2026; Google doesn't publish them; 1 CU ≈ $0.10, ≈ $0.083 on Pro+):

| GPU | CU/h | ≈ $/h | Notes |
|---|---|---|---|
| T4 15 GB | 1.19 | $0.12 | Turing; FP16 only |
| L4 22.5 GB | 1.71 | $0.17 | Good for data encoding / small models |
| A100 40 GB | 5.4 | $0.54 | Runner-S training |
| A100 80 GB | 7.52 | $0.75 | 1.3B full fine-tune, 14B LoRA (FP8 base) |
| **G4 = RTX PRO 6000 Blackwell 96 GB** | 8.71 | **$0.87** | **Best value for 14B LoRA**; also has FP4 |
| H100 80 GB | ~18 | ~$1.80 | Fastest single GPU; similar price to rental |

Colab is **single-GPU per session**, preemptible, and availability is not guaranteed. Rules:

- Checkpoint to Drive or the HF Hub every ~15 minutes.
- Pre-encode the dataset to VAE latents in sharded files, and copy shards to local disk at session start.
- Make every notebook resumable.

When a phase needs 8 GPUs (distillation), use a GPU rental (A100 80 GB ≈ $1.5/h, H100 ≈ $2–3/h on the market today).

**Local 5060 Ti:** fine for data capture, latent encoding, inference, Runner-S experiments, and small LoRAs. Blackwell consumer cards (sm_120) need CUDA 12.8+ and matching PyTorch builds. The 32 GB of system RAM limits CPU offload of 14B models: FP8 14B plus activations is workable, BF16 is not.

### 2.2 LoRA vs full fine-tune — what to do per tier

| Tier | Method | Why | Where | Rough cost |
|---|---|---|---|---|
| Runner (1.3B): add actions, pose (Plücker rays), guide-render + mask, memory-frame inputs, causal attention, depth head | **Full fine-tune** (new input modules can't be LoRA'd in meaningfully) | ~16–18 B/param with AdamW ≈ 22 GB + activations | Colab A100-80 / G4 / H100 for dev; 8×H100 rental for the real run | Dev runs $50–200; main run ~300–1000 H100-h ≈ **$700–2,500** |
| Runner distillation to 1–2 steps (Self-Forcing / DMD) | Distill from a bidirectional or many-step teacher | The authors trained in <2 h on 64 H100s (~128 H100-h); ~16 h on 8 H100s with gradient accumulation | 8×H100 rental | **~$300–600** per run |
| Runner-S (0.5B) | Distill from Runner | 30+ FPS on the 5060 Ti | Colab A100/H100 | $50–200 |
| Architect (14B): style/world LoRAs | **LoRA rank 16–64**, FP8 or 4-bit base (QLoRA-style) | Full fine-tune needs ~250 GB+ across 8 GPUs (FSDP) | Colab **G4 96 GB** or A100-80; local 16 GB possible with block swapping (musubi-tuner / diffusion-pipe / ai-toolkit class tools), slowly | 20–60 GPU-h ≈ **$20–110** per LoRA |
| Architect full fine-tune | **Defer** until funded | | 8×H100 cluster | $5k–20k+ |

**Phase 0/1 scratch model.** DIAMOND's CS:GO world model is the honest yardstick: 381M parameters, 87 h of gameplay, **12 days on one RTX 4090** (~290 GPU-hours). Expect about 150–300 A100-hours (~$110–230 on Colab A100-80) for a comparable action-conditioned model on your own data.

### 2.3 Data collection — the capture harness

Build the data engine in **Godot 4 (MIT)** with **CC0 assets** (Poly Haven, Kenney, ambientCG). That gives provenance you own outright. The same scenes later become the "graybox" for Anchored mode (see §3.6). Unreal is fine technically, but check its EULA for ML clauses. Fab assets tagged **"NoAI" may not be used in AI training datasets**, and Fab's terms bar using its *services* to train models.

Per-frame log record at 30 Hz:

```
frame_id, t, episode_id, world_seed
rgb (PNG/lossless-ish, 640×360 capture → train at 256–360p)
depth (linear float16), normals (optional), instance_id map
camera K, camera pose T_world_cam (6DoF), player pose
action: key bitmask, mouse dx/dy, buttons, (gamepad axes)
cell_id, portal crossed?, game_state record + state_hash
```

How to collect it:

- **Agents:** scripted explorers plus an RL agent (GameNGen used PPO-agent play), plus human sessions for natural motion.
- **Revisit trajectories:** deliberately *leave rooms and come back* after varying delays, and change state (open doors, move props) before leaving. This is what Solidify trains and is evaluated on.
- **Volumes:** Phase 1 ≈ 50–100 h (DIAMOND scale). Phase 2 ≈ 300–1000 h (Matrix-Game 2.0 used ~1200 h).
- **Storage:** DIAMOND's 95 h was ~660 GB raw. Store raw on NVMe or a NAS, train from pre-encoded latents.
- **Domain:** one genre first. **Indoor first-person exploration** (house, museum, station) is recommended because rooms make persistence visible and demonstrable.

---

## 3. "Solidify" — the persistence harness

### 3.1 What Uncharted 2 actually did, and the lesson

Uncharted 2 (PS3: 256 MB main RAM + 256 MB video RAM) famously played without loading screens. It **streamed environments in chunks** as you moved; the train sequence, for example, ran through "a streaming series of environments." Gameplay state was driven by Naughty Dog's **per-object state scripts and game-object properties**, plus **checkpoints**. A room looks the same when you return **not because the renderer remembers it, but because it is re-rendered from canonical authored data** streamed back from disc, **plus a small persisted state delta** (what's open, broken, or dead).

The general engine pattern: either keep lightweight game objects resident while their heavy mesh/texture resources stream with the region, or save object state on unload and restore it on re-entry. Public sources don't document Uncharted 2's exact streaming internals, so treat this as the established pattern rather than a claim about their code.

**Lesson for a diffusion engine:** pure world models (GameNGen, Oasis, DIAMOND, even Genie 3's "a few minutes" of consistency) keep the world in the model's context window, so it decays. DIAMOND's authors note that turning toward a wall makes it forget the map. **Solidify turns generated content into canonical data the first time it's seen, stores state as small authoritative records, and streams both.** The model then only has to *render* the room again, never *remember* it.

### 3.2 Data model

The world is a graph of **cells** (rooms/zones) linked by **portals**. In Anchored mode, cells come from level data. In Dream mode, they come from a sparse 3D hash grid (e.g. 6–8 m cells) merged at detected doorways.

```
Cell {
  id, bounds, portals[], neighbors[]
  seed        = H(world_seed, id)                 // deterministic noise per cell
  cache       : 3D Gaussian splats / surfels      // xyz, color(SH0–1), opacity, scale, normal,
                                                  // confidence, n_views, last_seen, object_id
  keyframes[] : {pose, latent, depth, coverage}   // pose-indexed episodic memory (≈16–64)
  state       : {object_id → state}               // doors, props, NPC flags — authoritative, tiny
  status      : UNSEEN | FORMING | SOLID | DIRTY
  version, state_hash = H(cache_hash, state, runner_id, lora_id)
}
```

Per-room footprint is about 200–500k splats. Compressed SH0 splats run ~4–10 MB, plus 32 keyframe latents (~4 MB), so **~10–25 MB per room**. A hundred rooms fit in about 2 GB of system RAM.

### 3.3 Per-frame loop

1. **Engine step.** Actions → controller + collision → **camera pose** and **state** updates, both deterministic. Pose never comes from the model. In Dream mode, collision runs against the cache, with a coarse occupancy grid built from depth.
2. **Visibility and streaming.** Portal culling finds the visible cells, and the streamer ensures they're hot.
3. **Guide render.** Rasterize the cache at the current pose (Gaussian splat raster in Vulkan, ~2–3 ms at 360p) to get **RGB, depth, and coverage mask** (GEN3C / Voyager-style conditioning).
4. **Retrieval.** Pick the top-k keyframes by **FOV overlap with the current pose** (WorldMem, Context-as-Memory) or by **surfel votes** (VMem: each surfel remembers which views saw it).
5. **Generate.** The Runner denoises the next frame conditioned on:
   - working memory (recent frames, via the KV cache);
   - actions;
   - pose (Plücker embedding);
   - guide RGB-D + mask;
   - retrieved keyframes (memory attention);
   - noise seeded from `H(cell.seed, quantized_pose, frame_index)`.

   It outputs **RGB + depth**.
6. **Consistency guard.** In covered pixels, compare the generated frame with the guide render (reprojection error plus a cheap perceptual distance). Above a threshold, **snap back**: re-denoise from a partially noised guide render (SDEdit-style) or composite the guide in those pixels.
7. **Lift and fuse.** Unproject depth and add splats **only where coverage is missing or confidence is low**. Filter dynamic content out of the static cache (as in "Video World Models with Long-term Spatial Memory"); dynamics live in the state record. Update keyframes if the view adds coverage.
8. **Lock.** When coverage exceeds ~90% of the cell's visible surface from ≥2 views and the consistency error stays low, set `status = SOLID`. The cache becomes read-only, and future frames in the cell are *render-conditioned refinement*. Following GEN3C's insight, the model spends its capacity on unseen regions, view-dependent effects, and dynamics, not on recall.

### 3.4 State-hash locking, edits, determinism

- **Hash:** `state_hash = BLAKE3(cache_blob_hash ‖ state_record ‖ runner_id ‖ lora_id)`. Show it in the debug HUD. The hash is the save-game key, the multiplayer sync token (send hash + deltas), and the regression-test oracle: same hash means same room.
- **Edits:** state changes (vase knocked over, door opened) are **deltas on object-tagged splat groups**. Object IDs come from the engine in Anchored mode, or from instance segmentation per splat cluster in Dream mode. Applying an edit bumps `version`, marks the affected region DIRTY, lets the Runner re-solidify only that region, and then re-locks.
- **"Exact" rematerialization, stated honestly:**
  - **Geometry and appearance** of a SOLID cell are canonical: they come from the cache, not the model's memory. That is the strong guarantee.
  - **Pixel-identical** frames for a returning pose need identical inputs: same cache version, same seed from the hash and quantized pose, and deterministic kernels (fixed TensorRT engine and tactics, deterministic cuBLAS). Recent-frame context differs on re-entry, so use an **Exact-Replay mode**: when the pose matches a stored keyframe within tolerance, condition dominantly on that keyframe and the guide render, or show the cached frame directly.
  - The claim to make publicly: "*the room comes back the same — same layout, same objects, same state, same hash*."
- **Model upgrades:** a new Runner or LoRA changes the hash. Either keep caches (they're model-agnostic 3D data, which is a strength) or offer a "re-bake" pass using the Architect.

### 3.5 Streaming & eviction (the Uncharted part)

| Tier | Contents | Medium |
|---|---|---|
| **Hot** | Current + portal-adjacent cells: full splats, keyframe latents, guide-render buffers | VRAM (0.5–1 GB budget) |
| **Warm** | Recently visited / predicted cells: compressed splats + latents | System RAM (budget ~4–8 GB of the 32 GB) |
| **Cold** | Everything else: quantized splats (zstd), keyframes, state | NVMe, one file per cell, memory-mapped |

- **Always resident:** the cell graph and **all state records**. They're tiny, and this matches the "keep game objects alive, stream heavy resources" pattern.
- **Prefetch:** portal graph distance plus player velocity/heading predicts the next cells; loads are async (Vulkan transfer queue).
- **Eviction:** LRU weighted by graph distance. Evicting never loses information: Hot is written back to Warm, Warm to Cold.
- **Architect jobs** (async, low priority) refine Cold/Warm caches while the player is elsewhere: fill holes and upgrade texture detail with the 14B model. Each job bumps the version, and the cell is re-locked on the next visit.

### 3.6 Two modes, one conditioning interface

- **Dream mode** (the wow demo): the world is invented by the model and solidified from its own RGB-D.
- **Anchored mode** (the commercial product): a designer's graybox or level blockout plus game logic is rasterized by the engine as the guide render. The diffusion model renders final pixels and style, and Solidify caches the result.

Both modes feed the Runner the same input: **guide RGB-D + coverage mask + keyframes**. One model, two products.

### 3.7 Research this builds on (verified)

| Work | What to take |
|---|---|
| **WorldMem** (Xiao et al., NeurIPS 2025, [arXiv 2504.12369](https://arxiv.org/abs/2504.12369)) | Memory bank of frames + states (pose, timestamp), state-aware memory attention, FOV/time retrieval. Code and weights are public. |
| **Context-as-Memory** ([arXiv 2506.03141](https://arxiv.org/abs/2506.03141)) | Store raw frames; retrieve by camera FOV overlap; concatenate as context. Simple and strong. |
| **VMem** (Li, Torr, Vedaldi, Jakab, [arXiv 2506.18903](https://arxiv.org/abs/2506.18903)) | Surfel-indexed view memory: retrieve the past views that saw the surfaces you're looking at. |
| **Video World Models with Long-term Spatial Memory** (NeurIPS 2025, [arXiv 2506.05284](https://arxiv.org/abs/2506.05284)) | Working + spatial (static point cloud) + episodic (keyframes) memory; filter dynamics before storing. |
| **GEN3C** (NVIDIA, CVPR 2025, [arXiv 2503.03751](https://arxiv.org/abs/2503.03751)) | 3D cache of point clouds from depth; condition on cache renders so the model "neither has to remember… nor infer structure." |
| **Voyager** (Tencent, [arXiv 2506.04225](https://arxiv.org/abs/2506.04225)) | Joint RGB-D generation, world cache with point culling. The idea is useful; the code and weights are **Tencent-licensed (EU/UK/KR excluded)**, so reimplement rather than reuse. |
| **Matrix-Game 3.0** ([project](https://matrix-game-v3.github.io/)) | Camera-aware memory retrieval plus error-buffer self-correction at real-time rates. The closest open system to this plan. |
| **Self-Forcing** ([arXiv 2506.08009](https://arxiv.org/abs/2506.08009)) | Train on own rollouts with KV cache; few-step real-time AR video. The Runner's training recipe. |
| **GameNGen** ([arXiv 2408.14837](https://arxiv.org/abs/2408.14837)), **DIAMOND** ([repo](https://github.com/eloialonso/diamond)), **Oasis** ([site](https://oasis-model.github.io/)) | Action conditioning, context-noise augmentation against drift, two-stage low-res dynamics + upsampler. |
| **Genie 3** ([DeepMind](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)) | The bar: 720p/24 FPS with "a few minutes" of emergent consistency. Closed. Solidify's pitch is *explicit, unbounded* consistency instead. |

---

## 4. Runtime & presentation

### 4.1 Recommendation: native C++/Vulkan runtime, ImGui editor, no Electron in the loop

| Layer | Choice |
|---|---|
| Core | C++20, Vulkan 1.3, a job system; Windows first (Aaron's rig), Linux for training/servers |
| Inference | **TensorRT** engines (FP8, then NVFP4 on Blackwell), built per GPU arch, with fixed tactics for determinism. Fallback: **ONNX Runtime** (TensorRT/CUDA EP; DirectML for non-NVIDIA later) |
| Interop | CUDA ↔ Vulkan external memory and semaphores: the Runner writes straight into a Vulkan image. **Zero CPU copies.** |
| Rendering | Gaussian-splat rasterizer (compute) for guide renders, composite, upscaler, **depth reprojection** to the display rate, native HUD/UI |
| Editor | **Dear ImGui** (MIT) docked editor showing the cell graph, cache coverage heatmap, state-hash panel, keyframe browser, frame-time budget, model/LoRA switcher, and a capture-harness toggle |
| Training/research | Python/PyTorch only. Export through ONNX → TensorRT. The Python prototype (Phases 0–1) uses the same data schema. |

Frame budget at 30 Hz (33 ms) on the 5060 Ti:

| Step | Time |
|---|---|
| Engine | ~1 ms |
| Guide render | 2–3 ms |
| Retrieval | <0.5 ms |
| Runner, 1–2 steps | 15–22 ms |
| Tiny VAE decode | 3–5 ms |
| Upscale | 2–3 ms |
| Present | ~1 ms |

The display runs at 60–120 Hz through reprojection, so camera input *feels* instant even when neural frames arrive at 20–30 Hz.

### 4.2 Why not Electron

Electron puts Chromium between the GPU and the screen. Frames produced by native CUDA must reach a WebGL/WebGPU canvas through extra copies or shared-texture workarounds, which adds latency and jitter. Chromium also costs several hundred MB of RAM and contends for the same GPU. Input latency is the number one complaint about neural game engines, so don't add to it.

Electron (or a plain browser UI) is fine **later** for a "Prime Studio": asset/world browser, cloud-job dashboard, LoRA training UI. It would talk to the native runtime over a local gRPC/WebSocket API and never sit in the frame path.

A native runtime also unlocks the best commercial channel: **Unreal/Unity plugins** that embed the same C++ core.

---

## 5. Commercialization

### 5.1 License map

| Status | Components |
|---|---|
| ✅ **Green** (permissive) | Wan2.1 / Wan2.2 (Apache-2.0) · Matrix-Game 2.0 (MIT) · Matrix-Game 3.0 (Apache-2.0) · LingBot-World (Apache-2.0) · Oasis 500M (MIT) · DIAMOND (MIT) · Self-Forcing code and weights (Apache-2.0) · Depth Anything V2 **Small** / Video-Depth-Anything **Small** (Apache-2.0) · Godot (MIT) · Dear ImGui (MIT) · ONNX Runtime (MIT) · CC0 assets |
| 🟡 **Yellow** (commercial with conditions) | NVIDIA Cosmos (Open Model License: commercial OK, "Built on NVIDIA Cosmos" attribution, revocable terms) · LTX-Video 13B (free under $10M company revenue, paid above) · Stable Diffusion 1.x (OpenRAIL-M use restrictions) · TensorRT (NVIDIA SDK license; redistributable runtime) |
| ❌ **Red** (non-commercial or territory-restricted) | **Hunyuan family** (no EU/UK/South Korea; MAU caps) · **Depth Anything V2 Base/Large/Giant and Video-Depth-Anything Base/Large (CC-BY-NC-4.0)** · DUSt3R/MASt3R (non-commercial; verify) · Genie 3 (not available) · Fab assets tagged NoAI, for training |

### 5.2 IP & data provenance

- **Permissive weights ≠ clean outputs.** Matrix-Game 2.0 learned from GTA5, Matrix-Game 3.0 from AAA games, Oasis from Minecraft, DIAMOND from CS:GO. Fine-tuning on them can reproduce trademarked or copyrighted looks. Ship only models whose *fine-tuning* data you own, and treat game-trained checkpoints as research baselines. Base video models (Wan, Cosmos) were pretrained on undisclosed web video; that's an industry-wide risk, mitigated by heavy fine-tuning on owned data and by indemnity-minded customers choosing Anchored mode.
- **Data ledger from day one:** every clip carries a source, license, consent, capture build, and content hash. Each training run writes a **provenance manifest** listing the data hashes, base model, and licenses. Enterprise buyers and investors will ask for it.
- Keep the Solidify implementation proprietary. A patent on the *combination* (cell streaming + lock + state hash + exact replay + render-conditioned diffusion) may be possible, but prior art is dense (GEN3C, WorldMem, VMem, Voyager); talk to a patent attorney before publishing details.
- **Weights distribution:** an encrypted TensorRT engine plus an EULA for local SKUs, or keep the Architect cloud-only.

### 5.3 Differentiators that turn heads

1. **Persistence you can verify:** rooms come back identical with an on-screen state hash. Nobody else's world model ships explicit, unbounded persistence.
2. **Runs on a 16 GB consumer GPU.** Open competitors need H100s: Matrix-Game 3.0 uses 8 GPUs for its headline rate, and LingBot-World says enterprise GPUs are required.
3. **Anchored mode:** blockout in, styled playable world out, with art-direction LoRAs ("render this graybox as rainy noir").
4. **Feels responsive:** 60–120 Hz camera through reprojection, with HUD/UI drawn natively and crisply.
5. **Deterministic saves and shareable worlds:** seed + state records + cache = a save file you can send.

### 5.4 Go-to-market and demo strategy

- **First market:** previs/prototyping and "playable concept art" for indie studios, game jams, and film previs. Quality expectations are lower there, and speed-to-idea is the value. Then Unreal/Unity plugins, then a cloud "playable worlds" API.
- **Hero demo, "The Return" (90 s):**
  1. Walk a generated mansion on an RTX 5060 Ti (sticker on screen).
  2. Knock over a vase and open a cabinet.
  3. Leave through three rooms and wait 5 minutes.
  4. Return: identical room, vase still on the floor, **same state hash**.

  Run it split-screen against Solidify-off, where the world drifts.
- **Proof, not vibes:** publish revisit metrics (PSNR/LPIPS of revisited views vs. first visit, the way WorldMem evaluates), FPS, and motion-to-photon latency. Release a small open **"Solidify-Bench"** (revisit trajectories in CC0 scenes) to define the metric others get measured on.
- **Launch:** a tech-report PDF (or arXiv), a YouTube/X video, a downloadable Windows build of the demo house, and a Colab notebook for the Phase 0 toy.

---

## 6. Phased roadmap

Each phase has an exit gate; don't advance until it passes.

### Phase 0 — Foundations (local 5060 Ti + Colab Pro+)
- Godot capture harness with the §2.3 schema; a 4–6 room CC0 test house; scripted explorer with revisit routes.
- Toy action-conditioned diffusion world model (DIAMOND-style, 64–128 px, ~50–100M params). The Colab prototype being built in the repo in parallel should adopt this schema.
- Solidify v0 in Python: point cloud from **ground-truth** depth, FOV retrieval, guide-render conditioning.
- **Measure** real 5060 Ti throughput for Wan2.1-1.3B and Matrix-Game 2.0 inference, and replace the §1.1 estimates.
- **Exit:** toy model playable; revisit LPIPS clearly better with Solidify than without; benchmark numbers recorded.

### Phase 1 — Runner v0 + Solidify v1 (Colab A100/G4/H100, ~$100–300)
- ~400M scratch model, or the Wan2.1-1.3B fine-tune at 256p, on 50–100 h of own data. Predicts RGB-D.
- Solidify v1: model-predicted depth, splat cache, locking, consistency guard, state deltas, Warm/Cold streaming to disk.
- First rough cut of "The Return" (Python, not real-time).
- **Exit:** a 10-minute session with ≥5 revisits per room shows no structural drift; edits persist; hashes match.

### Phase 2 — Real-time Runner (rented 8×H100, ~$1–3k)
- Causal conversion plus Self-Forcing/DMD distillation to 1–2 steps; Architect (Wan 14B + own LoRA) as teacher.
- Runner-S (~0.5B) distillation; tiny VAE decoder; 2× upscaler.
- Export through ONNX → TensorRT FP8/NVFP4.
- **Exit:** ≥20 FPS at 360p internal on the 5060 Ti, with Solidify on.

### Phase 3 — Native runtime ("Prime Runtime")
- C++/Vulkan core, CUDA-Vulkan interop, splat rasterizer, cell streamer (VRAM/RAM/NVMe), depth reprojection to 60+ Hz, ImGui editor, deterministic TensorRT engines.
- **Exit:** "The Return" runs natively and live, within the frame budget and motion-to-photon targets; the hero video is recorded.

### Phase 4 — Architect & Anchored mode
- Wan 14B style/world LoRAs (Colab G4); async cache refinement during pauses.
- Graybox import (Godot/glTF) → Anchored mode; designer workflow in the editor.
- **Exit:** a designer blocks out a level and gets a styled, persistent, playable world.

### Phase 5 — Commercial
- SDK and EULA, Unreal plugin, cloud Architect API, provenance manifests, Solidify-Bench release, tech report, funding conversations.
- Scale up (Runner-L at 480–720p on 5090-class cards, multi-GPU cloud tier), and do a full Architect fine-tune once funded.

---

## 7. Top risks & mitigations

| Risk | Mitigation |
|---|---|
| FPS falls short on the 5060 Ti | Runner-S, 256p internal + upscaler, high-compression VAE (Wan2.2-style 16×16×4), NVFP4, reprojection to hide the frame rate |
| Autoregressive drift | Self-Forcing training, context-noise augmentation (GameNGen), consistency guard, guide-render dominance in SOLID cells |
| Input latency | Async pipeline + reprojection; pose comes from the engine, not the model |
| Bad depth corrupts the cache | Train depth from engine ground truth; confidence-weighted fusion; only lock after multi-view agreement |
| Dynamic objects / NPCs | Keep them out of the static cache: engine-side state, rendered as guide proxies |
| Text and UI garbled by the model | The engine draws all HUD/UI natively |
| License changes | Pin model versions, archive license texts in the repo, keep green alternatives per tier |

---

## 8. Decisions Aaron needs to make

1. **Accept the tier split:** the 13–14B model is the Architect (authoring/teacher), not the real-time renderer. (Recommended: yes.)
2. **Lead product mode:** Anchored mode (graybox → styled world) for the commercial product, with Dream mode as the viral demo. (Recommended: build both on the shared interface; *sell* Anchored.)
3. **Runner base:** clean **Wan2.1-1.3B** (more work, shippable) or **Matrix-Game 2.0** (faster start, GTA5 provenance, research only). (Recommended: Wan; use Matrix-Game only as a baseline.)
4. **First content domain:** indoor first-person exploration (recommended) or something else.
5. **Data engine:** **Godot 4 + CC0** (recommended) or Unreal.
6. **Budget envelope:** ~$50/mo Colab Pro+ through Phase 1, then ~$1–3k of rented GPUs for Phase 2.
7. **Platform:** Windows-first native runtime (recommended, matches his rig), with Linux for training.
