# Capture Studio

Capture Studio is a local photo-to-3D Gaussian Splatting project. Its first
portfolio feature will analyze a set of photographs before training and explain
which images are blurry, redundant, or leave gaps in camera coverage.

## Current status

Step 1 establishes a reproducible Python project and reports which native and
GPU prerequisites are available. 3D reconstruction and training are not
implemented yet.

## Setup

Install [uv](https://docs.astral.sh/uv/) and run:

```powershell
uv sync
uv run capture-studio check-system
uv run pytest
```

The environment report checks Python, uv, the NVIDIA GPU, PyTorch, the CUDA
compiler, and COLMAP. A `[MISSING]` result tells us what a later setup step must
install; it does not mean the diagnostic command failed.
