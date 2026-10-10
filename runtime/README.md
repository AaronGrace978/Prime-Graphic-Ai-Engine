# Prime Runtime (Phase 3)

Not built yet. The live frame path is reserved for a native runtime:

- C++20, Vulkan 1.3, Windows first
- TensorRT (FP8, then NVFP4) with CUDA–Vulkan interop
- Gaussian-splat guide render, depth reprojection to 60–120 Hz
- Dear ImGui editor (cell graph, coverage, state hash)

Electron is not part of this path. A Studio shell can sit beside it later, over IPC.

Phase 0, in this repo, is the Python capture format, the Solidify cache, and the toy Runner. Same records, same cell hash, smaller model. See `docs/engine-plan.md`.
