import numpy as np
import torch

from prime_engine.config import load_config
from prime_engine.harness import CellStatus, SolidifyConfig, SolidifyStore, WorldModelRunner
from prime_engine.models import Diffusion
from prime_engine.training import TrajectoryBatcher, build_model
from prime_engine.world import RoomWorld, WorldConfig, collect_episode, generate_dataset, scripted_actions_to
from prime_engine.world.raycaster import FORWARD, TURN_RIGHT

SMALL = WorldConfig(width=16, height=16)


def test_render_and_kinematics():
    w = RoomWorld(SMALL, seed=0)
    rgb, depth, inst = w.render(aux=True)
    assert rgb.shape == (16, 16, 3) and depth.shape == (16, 16) and inst.shape == (16, 16)
    p0 = w.pose.copy()
    w.step(FORWARD)
    assert w.pose[0] > p0[0]
    for _ in range(100):  # walls stop the agent
        w.step(FORWARD)
    assert not w.is_wall(*w.pose[:2])


def test_scripted_route_reaches_every_room():
    w = RoomWorld(SMALL, seed=1)
    for r in (1, 3, 2, 0):
        for a in scripted_actions_to(w, w.pose, w.room_center(r)):
            w.step(a)
        assert w.room_at(*w.pose[:2]) == r


def test_episode_schema():
    ep = collect_episode(SMALL, 20, seed=3)
    assert ep["rgb"].shape == (21, 16, 16, 3) and ep["action"].shape == (20,)
    assert ep["T_world_cam"].shape == (21, 4, 4) and ep["state_hash"].dtype == np.uint64
    np.testing.assert_allclose(ep["T_world_cam"][:, :2, 3], ep["player_pose"][:, :2])


def test_solidify_store_locks_and_hashes():
    cfg = SolidifyConfig(exclude_recent=0)
    store = SolidifyStore(cfg, world_seed=7)
    store.register_cells(RoomWorld(SMALL, seed=7).cell_graph())
    pose = np.array([4.0, 4.0, 0.0])
    for i in range(22):  # 22/24 heading bins >= lock_coverage 0.9
        assert store.cells[0].status != CellStatus.SOLID
        assert store.write(pose + [0, 0, i * np.pi / 12], torch.full((3, 4, 4), i / 24), step=i, cell_id=0)
    assert store.cells[0].status == CellStatus.SOLID
    h = store.state_hash(0)
    assert not store.write(pose, torch.zeros(3, 4, 4), step=99, cell_id=0)  # SOLID cells are read-only
    assert store.state_hash(0) == h
    hits = store.retrieve(pose + [0.1, 0, 0.05], k=2, cell_id=0)
    assert hits[0][0].step == 0
    store.set_state(0, door="open")
    assert store.cells[0].status == CellStatus.DIRTY and store.state_hash(0) != h


def test_train_step_and_rollout():
    cfg = load_config(None, ["model.base_ch=16", "model.cond_dim=32", "model.num_res_blocks=1"])
    model = build_model(cfg)
    data = generate_dataset(SMALL, 2, 30, workers=1)
    batcher = TrajectoryBatcher(data, cfg.model.context_frames, cfg.model.memory_slots, cfg.harness,
                                vis_fn=RoomWorld(SMALL).visibility)
    b = batcher.sample(4, torch.Generator().manual_seed(0), 0.3, 0.25, 0.15, revisit_frac=0.5)
    loss = Diffusion().loss(model, b.pop("target"), **b)
    loss.backward()
    assert torch.isfinite(loss)

    runner = WorldModelRunner(model, Diffusion(), batch=1, sample_steps=2, memory_cfg=cfg.harness)
    K = cfg.model.context_frames
    w = RoomWorld(SMALL)
    vis = np.stack([w.visibility(p) for p in data["player_pose"][0, :K + 1]])[None]
    runner.reset(data["rgb"][:1, :K], np.zeros((1, K), int), data["player_pose"][:1, :K].astype(np.float64),
                 cell_ids=data["cell_id"][:1, :K], visibility=vis[:, :K])
    frame = runner.step([TURN_RIGHT], data["player_pose"][:1, K], cell_ids=data["cell_id"][:1, K],
                        visibility=vis[:, K])
    assert frame.shape == (1, 16, 16, 3) and len(runner.stores[0]) >= 1
