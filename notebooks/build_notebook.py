"""Regenerates prime_engine_poc.ipynb (keeps the notebook diff-able and in sync with the package)."""
import nbformat as nbf

REPO = "AaronGrace978/Prime-Graphic-Ai-Engine"
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(f"""# Prime Graphic AI Engine — diffusion world model + Solidify PoC

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/prime_engine_poc.ipynb)

This notebook runs the v0 proof of concept end to end:

1. **Collect data** from a multi-room 2.5D raycaster (RGB, depth, camera pose, inputs, cell id, state hash).
2. **Train** a small action-conditioned diffusion model that renders the next frame.
3. **Benchmark** "The Return": leave a room, come back, and measure whether it rematerialises consistently, **with vs. without** the Solidify harness.
4. **Play** interactively: the engine moves the camera, the model draws the frames.

**Runtime → Change runtime type → T4 GPU** (free tier works). On CPU it falls back to the tiny 32×32 config."""),
    md("## 1. Setup"),
    code(f"""import os, subprocess, sys
REPO_URL = "https://github.com/{REPO}.git"
BRANCH = "main"  # set to a PR branch to try unmerged work
if not os.path.exists("prime_engine") and not os.path.exists("Prime-Graphic-Ai-Engine"):
    subprocess.run(["git", "clone", "-b", BRANCH, "--depth", "1", REPO_URL], check=True)
if os.path.exists("Prime-Graphic-Ai-Engine"):
    os.chdir("Prime-Graphic-Ai-Engine")
in_colab = "google.colab" in sys.modules
req = "requirements-colab.txt" if in_colab else "requirements.txt"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], check=True)
sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())"""),
    code("""import torch, numpy as np, matplotlib.pyplot as plt
from IPython.display import Image as IPImage, display
from prime_engine.config import load_config
from prime_engine.training import ensure_dataset, train, load_checkpoint, pick_device

device = pick_device()
CONFIG = "configs/colab_t4.yaml" if device.type == "cuda" else "configs/cpu_smoke.yaml"
QUICK = False  # True = a few-minute run just to see everything work
overrides = ["train.steps=1500", "data.episodes=200"] if QUICK else []
cfg = load_config(CONFIG, overrides)
print(device, CONFIG, f"{cfg.world.width}x{cfg.world.height}", "steps:", cfg.train.steps)"""),
    md("## 2. Dataset: frames + depth + camera pose + inputs + room state"),
    code("""data = ensure_dataset(cfg)
print({k: (v.shape, str(v.dtype)) for k, v in data.items() if hasattr(v, "shape") and v.ndim})
print("portal crossings / episode:", data["portal_crossed"].sum(1).mean())

from prime_engine.world import RoomWorld
from prime_engine.eval import map_image
ep = 0
w = RoomWorld(cfg.world, room_colors=data["cell_state"][ep])
fig, ax = plt.subplots(2, 7, figsize=(16, 5))
for i, t in enumerate(np.linspace(0, data["rgb"].shape[1] - 1, 7).astype(int)):
    ax[0, i].imshow(data["rgb"][ep, t]); ax[0, i].set_title(f"t={t} cell={data['cell_id'][ep, t]}")
    ax[1, i].imshow(data["depth"][ep, t].astype(np.float32), cmap="magma_r")
for a in ax.flat: a.axis("off")
plt.show()
plt.figure(figsize=(4, 4)); plt.imshow(map_image(w, data["player_pose"][ep])); plt.axis("off"); plt.title("episode 0 path"); plt.show()"""),
    md("## 3. Train the action-conditioned diffusion model\n\nOn a T4 the default config takes a while; checkpoints are written every `train.ckpt_every` steps to `runs/`, so you can stop early and continue with the cells below."),
    code("""ckpt = train(cfg, data)
import json
log = [json.loads(l) for l in open(os.path.join(cfg.train.out_dir, "train_log.jsonl"))]
plt.plot([r["step"] for r in log], [r["loss"] for r in log]); plt.yscale("log"); plt.xlabel("step"); plt.ylabel("loss"); plt.show()"""),
    md("## 4. The Return: revisit consistency with vs. without Solidify\n\nEach world: look around room A, walk to room B, look around, walk back, look around A again. Only the first 4 frames are real; everything else is generated. We compare revisit frames against the first-visit frames at the same pose and against ground truth, and check whether walls get the room's true colour (chance = 1/8)."),
    code("""from prime_engine.eval import run_revisit_benchmark, comparison_gif, revisit_figure
cfg_eval, model = load_checkpoint(os.path.join(cfg.train.out_dir, "model.pt"), device)
res = run_revisit_benchmark(model, cfg_eval.world, cfg_eval.harness, n_worlds=cfg_eval.eval.worlds,
                            sample_steps=cfg_eval.eval.sample_steps, aug_level=cfg_eval.eval.aug_level,
                            seed=cfg_eval.eval.seed, device=device)
metrics = {m: r["metrics"] for m, r in res["modes"].items()}
print(f"{'metric':28s}" + "".join(f"{m:>14s}" for m in metrics))
for k in next(iter(metrics.values())):
    print(f"{k:28s}" + "".join(f"{metrics[m][k]:14.3f}" for m in metrics))
print("\\nSolidify cells (world 0):")
for c in res["modes"]["solidify"]["cells_world0"]:
    print(c)"""),
    code("""os.makedirs("runs/eval", exist_ok=True)
display(IPImage(revisit_figure(res, 0, "runs/eval/revisit_world0.png")))
display(IPImage(comparison_gif(res, 0, "runs/eval/revisit_world0.gif")))"""),
    md("## 5. Play: the engine moves the camera, the model renders\n\nLeft to right: ground-truth raycast, model without memory, model with Solidify. Walk out of the first room and come back."),
    code("""import io, ipywidgets as W
from PIL import Image
from prime_engine.play import PlaySession
from prime_engine.world.raycaster import FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STRAFE_LEFT, STRAFE_RIGHT

session = PlaySession(model, cfg_eval, seed=7, device=device)
view, status = W.Image(format="png"), W.HTML()

def render():
    s = 6
    tiles = [np.asarray(Image.fromarray(session.last[k]).resize((cfg_eval.world.width * s, cfg_eval.world.height * s), Image.NEAREST))
             for k in ("gt", "no_harness", "solidify")]
    m = np.asarray(Image.fromarray(map_image(session.world, np.stack(session.trail), scale=8)).resize((tiles[0].shape[0],) * 2))
    buf = io.BytesIO(); Image.fromarray(np.concatenate(tiles + [m], 1)).save(buf, "PNG"); view.value = buf.getvalue()
    status.value = "<br>".join(f"cell {c['cell']}: {c['status']} kf={c['keyframes']} hash={c['state_hash']}" for c in session.cells())

def press(a):
    def f(_):
        session.step(a); render()
    return f

buttons = [(name, a) for name, a in [("⟲ left", TURN_LEFT), ("↑ fwd", FORWARD), ("⟳ right", TURN_RIGHT),
                                       ("← strafe", STRAFE_LEFT), ("↓ back", BACKWARD), ("strafe →", STRAFE_RIGHT)]]
btns = []
for name, a in buttons:
    b = W.Button(description=name); b.on_click(press(a)); btns.append(b)
render()
display(W.VBox([W.HTML("<b>ground truth | no harness | Solidify | map</b>"), view, W.HBox(btns[:3]), W.HBox(btns[3:]), status]))"""),
    md("""## Next steps
- Scale `configs/colab_t4.yaml` (resolution, `base_ch`, steps) or use `configs/rtx5060ti.yaml` locally.
- The architecture plan swaps pixel memory for a 3D cache + guide renders and a pretrained video backbone; the dataset already logs depth, intrinsics and 6DoF camera poses for that."""),
]
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
               "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/prime_engine_poc.ipynb")
print("wrote notebooks/prime_engine_poc.ipynb")
