# runtime/ (placeholder)

The native **C++20 / Vulkan** runtime ("Prime Runtime") will live here. It is intentionally empty in v0.

The Python package defines the interfaces it will need to match:

| Python (v0) | Runtime responsibility |
|---|---|
| `prime_engine.world.RoomWorld.next_pose` / `cell_graph` | Engine step: input -> controller + collision -> camera pose, cell id, state. Pose never comes from the model. |
| `prime_engine.harness.SolidifyStore` | Cell store: pose-keyed keyframes, status (`UNSEEN/FORMING/SOLID/DIRTY`), version, state hash; later the 3D cache and VRAM/RAM/NVMe streaming. |
| `prime_engine.harness.WorldModelRunner.step` | Per-frame loop: retrieve keyframes, run the denoiser (TensorRT), write back to the store. |
| `prime_engine.world.schema` | Capture record format shared with the data/capture harness. |

The architecture plan sets the design; build this once the Python reference behaviour is stable.
