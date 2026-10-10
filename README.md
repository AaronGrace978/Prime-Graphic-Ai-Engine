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
| World model | `prime_engine/models/` | UNet denoiser (v-prediction, cosine schedule, min-SNR loss, DDIM sampling). Conditioned on the last K frames, the last K actions, context-noise level (GameNGen-style anti-drift augmentation) and M memory views with their relative poses. |
| Solidify v0 | `prime_engine/harness/solidify.py` | Cells (rooms from the engine, or a hash grid in "Dream" mode). Each cell holds **pose-keyed keyframes** (position bin × heading bin, first write wins), a status (`UNSEEN → FORMING → SOLID`; `DIRTY` after a state edit), a version and a **state hash** = H(keyframe content, state record, runner id). Retrieval returns the closest views by pose from the current cell and its portal neighbours. SOLID cells are read-only. |
| Rollout | `prime_engine/harness/rollout.py`, `play.py` | Batched autoregressive renderer: the engine supplies pose and cell id, the model draws the frame, and Solidify stores and retrieves views. `PlaySession` runs harness-off and harness-on side by side for interactive play. |
| Benchmark | `prime_engine/eval/` | "The Return": look around room A, go to room B, come back, look around A again. Scores revisit consistency with vs. without Solidify. |
| Runtime | `runtime/` | Placeholder for the native C++/Vulkan runtime (not started on purpose). |

Training-time memory retrieval uses exactly the same pose distance and relative-pose features as runtime retrieval (`harness/geometry.py`). Memory is randomly dropped during training, so one checkpoint serves both "harness off" and "harness on".

## Results (v0 smoke run)

RESULTS_PLACEHOLDER

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

# CPU smoke test (~15 min on 4 cores): data -> train -> benchmark
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
- Memory views are channel-concatenated, and retrieval uses pose distance. Next steps: FOV-overlap retrieval, memory attention, a 3D cache with guide renders and a consistency guard.
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
