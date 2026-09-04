# Capture Studio

Capture Studio is a local photo-to-3D Gaussian Splatting project. Its first
portfolio feature will analyze a set of photographs before training and explain
which images are blurry, redundant, or leave gaps in camera coverage.

## Current status

The project has a reproducible Python environment, a verified CUDA-enabled
PyTorch installation, and a working gsplat CUDA rasterizer. 3D reconstruction
and training are not implemented yet.

## Setup

The Windows setup currently expects CUDA Toolkit 13.0 and Visual Studio 2022
Build Tools. Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
.\scripts\setup-windows.ps1
uv run capture-studio check-system
uv run capture-studio check-gpu
uv run capture-studio check-gsplat
uv run pytest
```

The setup script loads the C++ compiler and CUDA development headers before
asking `uv` to install the locked dependencies. gsplat is compiled only once;
`uv` caches the resulting package for later installs.

The environment report checks Python, uv, the NVIDIA GPU, PyTorch, the CUDA
compiler, and COLMAP. A `[MISSING]` result tells us what a later setup step must
install; it does not mean the diagnostic command failed.

`check-gpu` asks PyTorch to perform a known matrix multiplication on the CUDA
device and checks the result. This proves more than detecting the graphics card:
it verifies that this project's PyTorch build can execute work on the GPU.

`check-gsplat` goes one step further. It renders one synthetic 3D Gaussian into
a 64 by 64 image, checks that the Gaussian is visible, and runs
backpropagation. A passing result proves that both rendering and the gradient
calculation needed for training work on the GPU.

The project uses Python 3.10, PyTorch 2.10, and the CUDA 13.0 PyTorch runtime to
match a combination covered by gsplat's current Windows build matrix.

gsplat is pinned to commit `2b902ff` and built with its core 3DGS module only.
The current upstream head contains a Windows CUDA 13 compiler regression in a
new spherical-harmonics kernel. Optional 2DGS, 3DGUT, and experimental rendering
modules are outside this project's scope and are disabled.
