# Capture Studio

Capture Studio is a local photo-to-3D Gaussian Splatting project. Its first
portfolio feature analyzes photographs before training and explains which files
may reduce reconstruction quality.

## Current status

The project has a reproducible Python environment, a verified CUDA-enabled
PyTorch installation, a working gsplat CUDA rasterizer, COLMAP GPU feature
extraction, and basic photo-quality analysis. 3D reconstruction and training
are not implemented yet.

## Setup

The Windows setup currently expects CUDA Toolkit 13.0 and Visual Studio 2022
Build Tools. Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
.\scripts\setup-windows.ps1
.\scripts\install-colmap-windows.ps1
uv run capture-studio check-system
uv run capture-studio check-gpu
uv run capture-studio check-gsplat
uv run capture-studio check-colmap
uv run pytest
```

## Analyze a photo folder

Run the first user-facing feature against the included synthetic example:

```powershell
uv run capture-studio analyze .\examples\photo-analysis\photos
```

The visual comparison is saved at
[`examples/photo-analysis/comparison.png`](examples/photo-analysis/comparison.png).
The analyzer accepts any explicitly selected folder containing JPEG, PNG,
WebP, TIFF, or BMP images.

The report identifies unreadable images, photos below 2 megapixels, possible
blur, and byte-for-byte duplicates. The blur score is calculated at a
consistent maximum size so photographs can be compared more fairly. It is a
screening hint: a low-texture wall can score like a blurry photo even when it
is in focus.

This command currently analyzes files directly inside the selected folder. It
does not analyze videos or nested folders. Near-duplicate viewpoints and camera
coverage will be added after COLMAP reconstruction data is available.

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

`check-colmap` generates a temporary textured image and asks COLMAP to extract
SIFT features with GPU 0. It then checks the generated database contains real
features. COLMAP supplies the camera positions that training needs; gsplat uses
those camera positions to optimize and render the Gaussian scene.

The project uses Python 3.10, PyTorch 2.10, and the CUDA 13.0 PyTorch runtime to
match a combination covered by gsplat's current Windows build matrix.

gsplat is pinned to commit `2b902ff` and built with its core 3DGS module only.
The current upstream head contains a Windows CUDA 13 compiler regression in a
new spherical-harmonics kernel. Optional 2DGS, 3DGUT, and experimental rendering
modules are outside this project's scope and are disabled.

The COLMAP installer uses the official CUDA-enabled Windows archive for version
4.2.0 and verifies its published SHA-256 digest before installing it under the
current user's local Programs folder.
