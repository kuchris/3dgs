# Capture Studio

Capture Studio is a local photo-to-3D Gaussian Splatting project. Its first
portfolio feature will analyze a set of photographs before training and explain
which images are blurry, redundant, or leave gaps in camera coverage.

## Current status

The project has a reproducible Python environment and a verified CUDA-enabled
PyTorch installation. 3D reconstruction and training are not implemented yet.

## Setup

Install [uv](https://docs.astral.sh/uv/) and run:

```powershell
uv sync
uv run capture-studio check-system
uv run capture-studio check-gpu
uv run pytest
```

The environment report checks Python, uv, the NVIDIA GPU, PyTorch, the CUDA
compiler, and COLMAP. A `[MISSING]` result tells us what a later setup step must
install; it does not mean the diagnostic command failed.

`check-gpu` asks PyTorch to perform a known matrix multiplication on the CUDA
device and checks the result. This proves more than detecting the graphics card:
it verifies that this project's PyTorch build can execute work on the GPU.

The project uses Python 3.10, PyTorch 2.10, and the CUDA 13.0 PyTorch runtime to
match a combination covered by gsplat's current Windows build matrix.
