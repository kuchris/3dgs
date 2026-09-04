from __future__ import annotations

from dataclasses import dataclass


class GSplatCheckError(RuntimeError):
    """Raised when gsplat cannot render and differentiate a test scene."""


@dataclass(frozen=True)
class GSplatCheckResult:
    gsplat_version: str
    device_name: str
    render_shape: tuple[int, ...]
    nonzero_alpha_pixels: int


def run_gsplat_check() -> GSplatCheckResult:
    try:
        import gsplat
        import torch
        from gsplat.rendering import rasterization
    except ImportError as error:
        raise GSplatCheckError("gsplat and PyTorch must be installed") from error

    if not torch.cuda.is_available():
        raise GSplatCheckError("PyTorch cannot access a CUDA device")

    device = torch.device("cuda:0")
    means = torch.tensor([[0.0, 0.0, 3.0]], device=device, requires_grad=True)
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]], device=device, requires_grad=True
    )
    scales = torch.tensor([[0.35, 0.35, 0.35]], device=device, requires_grad=True)
    opacities = torch.tensor([0.9], device=device, requires_grad=True)
    colors = torch.tensor([[1.0, 0.2, 0.1]], device=device, requires_grad=True)
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    intrinsics = torch.tensor(
        [[[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]],
        device=device,
    )

    try:
        render, alpha, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            intrinsics,
            width=64,
            height=64,
        )
        (render.sum() + alpha.sum()).backward()
        torch.cuda.synchronize(device)
    except Exception as error:
        raise GSplatCheckError(f"CUDA rasterization failed: {error}") from error

    expected_shape = (1, 64, 64, 3)
    if tuple(render.shape) != expected_shape:
        raise GSplatCheckError(
            f"expected render shape {expected_shape}, got {tuple(render.shape)}"
        )
    if not torch.isfinite(render).all().item():
        raise GSplatCheckError("the rendered image contains non-finite values")

    nonzero_alpha_pixels = int((alpha > 0).sum().item())
    if nonzero_alpha_pixels == 0:
        raise GSplatCheckError("the rendered Gaussian is not visible")
    if colors.grad is None or not torch.isfinite(colors.grad).all().item():
        raise GSplatCheckError("backpropagation produced an invalid color gradient")

    return GSplatCheckResult(
        gsplat_version=gsplat.__version__,
        device_name=torch.cuda.get_device_name(device),
        render_shape=tuple(render.shape),
        nonzero_alpha_pixels=nonzero_alpha_pixels,
    )


def format_gsplat_report(result: GSplatCheckResult) -> str:
    return "\n".join(
        [
            "gsplat CUDA rasterizer smoke test",
            f"[OK] gsplat: {result.gsplat_version}",
            f"[OK] Device: {result.device_name}",
            f"[OK] Render shape: {result.render_shape}",
            f"[OK] Visible alpha pixels: {result.nonzero_alpha_pixels}",
            "[OK] Backpropagation: finite color gradient",
        ]
    )
