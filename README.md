# Prime Graphic AI Engine

A diffusion-model graphics engine: an **action-conditioned diffusion world model** that renders interactive game frames, plus **Solidify**, a persistence harness. With Solidify, a room you walk out of comes back the way you first saw it instead of being re-hallucinated.

Guiding principle: **the engine remembers; the model renders.** Game logic owns camera pose, collision, cells (rooms) and state. The diffusion model is a neural renderer conditioned on that state and on what Solidify has stored.

This repository is the **v0 proof of concept**. It is small enough to train on a free Colab T4, or in minutes on a CPU at 32×32.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AaronGrace978/Prime-Graphic-Ai-Engine/blob/main/notebooks/prime_engine_poc.ipynb)

## What's in v0

| Piece | Where | What it does |
|---|---|---|
| Multi-room world | `prime_engine/world/` | 2×2 rooms joined by doorways, rendered by a NumPy raycaster. Each episode gets random wall colours per room, so a room's look can't be guessed from geometry; it has to be remembered. Provides kinematics/collision (the "engine"), depth, instance maps and a cell graph. |
| Capture harness | `prime_engine/world/dataset.py`, `schema.py` | Scripted explorer that leaves and re-enters rooms. Logs RGB, linear depth, instance id, camera K, `T_world_cam` (6DoF), player pose, key bitmask + mouse dx, discrete action, cell id, portal crossings, per-cell state record + state hash. |
| World model | `prime_engine/models/` | UNet denoiser (v-prediction, cosine schedule, min-SNR loss, DDIM sampling). Conditioned on the last K frames, the last K actions, context-noise level (GameNGen-style anti-drift augmentation) and M memory views with their relative poses. When memory is present, the context is sometimes hue-corrupted in training, so the model learns to trust memory over its own drifted frames. |
| Solidify v0 | `prime_engine/harness/solidify.py` | Cells (rooms from the engine, or a hash grid in "Dream" mode). Each cell holds **pose-keyed keyframes** (position bin × heading bin, first write wins), a status (`UNSEEN → FORMING → SOLID`; `DIRTY` after a state edit), a version and a **state hash** = H(keyframe content, state record, runner id). SOLID cells are read-only. Retrieval searches the current cell and its portal neighbours and ranks views by **FOV overlap**: the cosine overlap of the engine's histograms of visible wall faces, a surfel-vote analogue. This finds room A's stored views even when you look back at A through a doorway from far away. Pose-distance retrieval is still available as a fallback. |
| Rollout | `prime_engine/harness/rollout.py`, `play.py` | Batched autoregressive renderer: the engine supplies pose and cell id, the model draws the frame, and Solidify stores and retrieves views. `PlaySession` runs harness-off and harness-on side by side for interactive play. |
| Benchmark | `prime_engine/eval/` | "The Return": look around room A, go to room B, come back, look around A again. Scores revisit consistency with vs. without Solidify. |
| Runtime | `runtime/` | Placeholder for the native C++/Vulkan runtime (not started on purpose). |

Training-time memory retrieval mirrors the runtime store. Only canonical (first-write) views older than the context window are candidates, ranked by the same FOV overlap, with the same relative-pose features (`harness/geometry.py`). Half of each batch comes from true revisits, where the place was last seen long before the context window. Memory is randomly dropped during training, so one checkpoint serves both "harness off" and "harness on".

## Results (v0 smoke run)

The model is tiny (1.3M parameters, 32×32) and trained on a 4-core CPU (no GPU): 6k steps (about 17 min), then a 4k-step fine-tune with context-corruption augmentation (`train.ctx_corrupt=0.2`, about 10 min). The benchmark is "The Return" on 32 held-out worlds with random room colours. Each world has 136 frames; only the first 4 are real, and everything else is generated autoregressively. Both modes use the **same checkpoint and the same noise**; the harness is the only difference.

| Metric (revisit look-around in room A) | No harness | **Solidify** | Paired diff (± s.e.) | Solidify better in |
|---|---|---|---|---|
| PSNR, revisit vs. first visit (self-consistency) | 14.86 dB | **16.61 dB** | +1.75 ± 0.21 dB | 31 / 32 worlds |
| PSNR, revisit vs. ground truth | 14.81 dB | **15.46 dB** | +0.65 ± 0.29 dB | 24 / 32 worlds |
| Room-colour accuracy (walls painted in the room's true colour; chance = 0.125) | 0.12 | **0.26** | +0.15 ± 0.04 | — |

What this does and doesn't show:

- With Solidify, a room comes back **looking like it did on the first visit** in 31 of 32 worlds. Without it, the model re-hallucinates the room, and revisit colour accuracy falls to chance.
- Absolute fidelity is low because the CPU model is small. It already drifts during the first 360° look-around (first-visit colour accuracy is only about 0.3). Solidify then faithfully restores the drifted version. Fidelity is a model-capacity problem, so expect it to improve with the T4 / 5060 Ti configs.
- The context-corruption fine-tune taught the model to trust memory over its own drifting frames. Before it, the gain in self-consistency PSNR was the same (+1.72 dB), but the colour-accuracy gain was smaller (+0.10). It costs some first-visit fidelity.

Reproduce:

```bash
python scripts/train.py --config configs/cpu_smoke.yaml train.steps=6000 train.ctx_corrupt=0 train.out_dir=runs/stage1
python scripts/train.py --config configs/cpu_smoke.yaml train.steps=4000 train.init_from=runs/stage1/model.pt \
    train.out_dir=runs/stage2 train.warmup=100 train.lr=3e-4
python scripts/eval_revisit.py --ckpt runs/stage2/model.pt --worlds 32
```

(`configs/cpu_smoke.yaml` on its own runs one 10k-step schedule with the augmentation from the start.)

## Run it in Colab

1. Click the badge above (or open `notebooks/prime_engine_poc.ipynb` in Colab).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all. The notebook clones the repo, generates data, trains with `configs/colab_t4.yaml` (64×64), runs the revisit benchmark, and opens an interactive play widget. Set `QUICK = True` for a few-minute run. To try an unmerged branch, set `BRANCH`.

## Run it locally (RTX 5060 Ti 16 GB or CPU)

```bash
git clone https://github.com/AaronGrace978/Prime-Graphic-Ai-Engine.git && cd Prime-Graphic-Ai-Engine
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate

# PyTorch first. RTX 50-series (Blackwell) needs CUDA 12.8+ wheels:
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu128
# (CPU only: --index-url https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt
pip install -e . --no-deps

# CPU run (~30 min on 4 cores; add train.steps=1000 for a quick check): data -> train -> benchmark
python scripts/train.py --config configs/cpu_smoke.yaml
python scripts/eval_revisit.py --ckpt runs/cpu_smoke/model.pt

# Full local run on the 5060 Ti (bf16, torch.compile)
python scripts/train.py --config configs/rtx5060ti.yaml
python scripts/eval_revisit.py --ckpt runs/rtx5060ti/model.pt --away 1,3   # leave through two rooms

pytest -q tests
```

Any config value can be overridden from the CLI, for example `python scripts/train.py --config configs/colab_t4.yaml train.steps=5000 world.width=96 world.height=96`. `scripts/generate_data.py` only builds the dataset. Training generates it automatically if it's missing.

Outputs: `runs/<name>/model.pt` (contains config + EMA weights), `train_log.jsonl`, and `eval/revisit_metrics.json` with a side-by-side GIF/PNG per world.

## Layout

```
prime_engine/            Python package
  world/                 raycaster world, explorer policy, capture schema + dataset
  models/                UNet denoiser, diffusion (v-pred, DDIM)
  training/              batch sampler with pose-retrieved memory, training loop
  harness/               Solidify store, pose geometry, autoregressive runner
  eval/                  revisit benchmark, metrics, figures
  play.py                interactive session (GT | no harness | Solidify)
configs/                 cpu_smoke / colab_t4 / rtx5060ti presets
scripts/                 generate_data.py, train.py, eval_revisit.py
notebooks/               Colab notebook (+ build_notebook.py that generates it)
runtime/                 placeholder for the C++/Vulkan runtime
tests/                   smoke tests (pytest)
```

## Known v0 simplifications

- Pixel-space diffusion at 32–64 px, trained from scratch. No pretrained backbone and no VAE latents yet. The memory payload is opaque, so it can hold latents later.
- Memory views are channel-concatenated. FOV overlap comes from engine geometry (Anchored mode), and Dream mode would need model-predicted depth. Next steps: memory attention, a 3D cache with guide renders and a consistency guard.
- Cell "coverage" is approximated by heading coverage at keyframe positions. State records are static (room colours), and the state hash uses BLAKE2b (stdlib) instead of BLAKE3.
- `pose_seeded_noise` (Exact-Replay flavour) is available in `SolidifyConfig` but off by default. Enable it with `eval_revisit.py --pose-noise` to ablate it.

## Licenses (all commercially permissive)

No pretrained models or third-party datasets are used. All weights are trained from scratch on procedurally generated data from this repo.

| Dependency | License |
|---|---|
| PyTorch | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| imageio | BSD-2-Clause |
| PyYAML | MIT |
| tqdm | MPL-2.0 and MIT |
| matplotlib (notebook) | PSF-based matplotlib license (BSD-compatible) |
| ipywidgets (notebook, preinstalled on Colab) | BSD-3-Clause |
| pytest (dev) | MIT |
| nbformat (notebook builder, dev) | BSD-3-Clause |

The project's own license is still to be chosen by the owner. Until then, all rights are reserved.
