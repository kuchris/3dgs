from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from capture_studio.colmap_scene import ColmapSceneError, SceneCamera, load_colmap_scene


class GaussianTrainingError(RuntimeError):
    """Raised when the Gaussian Splatting smoke trainer cannot finish."""


@dataclass(frozen=True)
class GaussianTrainingResult:
    steps: int
    cameras: int
    gaussians: int
    initial_loss: float
    final_loss: float
    peak_memory_gb: float
    device_name: str
    output: Path


ProgressReporter = Callable[[str], None]


def _load_image(camera: SceneCamera, image_scale: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(1, round(camera.width / image_scale))
    height = max(1, round(camera.height / image_scale))
    with Image.open(camera.image_path) as source:
        image = source.convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        pixels = np.asarray(image, dtype=np.float32) / 255.0

    intrinsics = camera.intrinsics.copy()
    intrinsics[0, :] *= width / camera.width
    intrinsics[1, :] *= height / camera.height
    return pixels, intrinsics


def create_render_comparison(
    reference: np.ndarray,
    rendered: np.ndarray,
    output_path: Path,
) -> None:
    if reference.shape != rendered.shape or reference.ndim != 3:
        raise GaussianTrainingError("reference and render must have matching RGB shapes")
    height, width = reference.shape[:2]
    label_height = 34
    canvas = Image.new("RGB", (width * 2, height + label_height), "#111827")
    reference_image = Image.fromarray(
        np.clip(reference * 255.0, 0, 255).astype(np.uint8)
    )
    rendered_image = Image.fromarray(
        np.clip(rendered * 255.0, 0, 255).astype(np.uint8)
    )
    canvas.paste(reference_image, (0, label_height))
    canvas.paste(rendered_image, (width, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Original photograph", fill="white")
    draw.text((width + 12, 10), "Gaussian render after smoke training", fill="white")
    canvas.save(output_path)


def _validate_output(output_path: Path) -> None:
    if output_path.exists() and any(output_path.iterdir()):
        raise GaussianTrainingError(f"output folder is not empty: {output_path}")


def train_gaussian_smoke_test(
    data_path: Path,
    output_path: Path,
    steps: int = 100,
    image_scale: int = 4,
    progress: ProgressReporter = print,
) -> GaussianTrainingResult:
    if steps < 1:
        raise GaussianTrainingError("training steps must be a positive integer")
    if image_scale < 1:
        raise GaussianTrainingError("image scale must be a positive integer")

    output_path = output_path.expanduser().resolve()
    _validate_output(output_path)
    try:
        scene = load_colmap_scene(data_path)
    except ColmapSceneError as error:
        raise GaussianTrainingError(str(error)) from error

    try:
        import torch
        from gsplat.init_utils import knn_scale_init
        from gsplat.rendering import rasterization
    except ImportError as error:
        raise GaussianTrainingError("gsplat and PyTorch must be installed") from error
    if not torch.cuda.is_available():
        raise GaussianTrainingError("PyTorch cannot access a CUDA device")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    output_path.mkdir(parents=True, exist_ok=True)

    progress(
        f"Initializing {len(scene.points):,} Gaussians from the sparse 3D points..."
    )
    means = torch.nn.Parameter(torch.from_numpy(scene.points).to(device))
    with torch.no_grad():
        initial_log_scales = knn_scale_init(means, k=3, chunk_size=1024)
    scales = torch.nn.Parameter(initial_log_scales[:, None].repeat(1, 3))
    quaternions = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(means), 1)
    )
    opacities = torch.nn.Parameter(
        torch.logit(torch.full((len(means),), 0.1, device=device))
    )
    color_logits = torch.nn.Parameter(
        torch.logit(torch.from_numpy(scene.colors).to(device).clamp(0.01, 0.99))
    )

    optimizer = torch.optim.Adam(
        [
            {"params": [means], "lr": 1.6e-4},
            {"params": [scales], "lr": 5e-3},
            {"params": [quaternions], "lr": 1e-3},
            {"params": [opacities], "lr": 5e-2},
            {"params": [color_logits], "lr": 2.5e-3},
        ],
        eps=1e-15,
        fused=True,
    )

    def render(camera: SceneCamera):
        target, intrinsics = _load_image(camera, image_scale)
        target_tensor = torch.from_numpy(target).to(device).unsqueeze(0)
        viewmat = torch.from_numpy(camera.world_to_camera).to(device).unsqueeze(0)
        intrinsic_tensor = torch.from_numpy(intrinsics).to(device).unsqueeze(0)
        height, width = target.shape[:2]
        colors, _, _ = rasterization(
            means=means,
            quats=torch.nn.functional.normalize(quaternions, dim=-1),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=torch.sigmoid(color_logits),
            viewmats=viewmat,
            Ks=intrinsic_tensor,
            width=width,
            height=height,
            packed=True,
            backgrounds=torch.ones((1, 3), device=device),
        )
        return colors, target_tensor

    preview_camera = scene.cameras[0]
    with torch.no_grad():
        initial_render, preview_target = render(preview_camera)
        initial_loss = torch.nn.functional.l1_loss(
            initial_render, preview_target
        ).item()

    progress(f"Training for {steps} steps on GPU 0...")
    random_camera = random.Random(42)
    training_losses: list[float] = []
    report_every = max(1, steps // 10)
    for step in range(steps):
        camera = (
            preview_camera
            if step % 16 == 0
            else scene.cameras[random_camera.randrange(len(scene.cameras))]
        )
        optimizer.zero_grad(set_to_none=True)
        rendered, target = render(camera)
        loss = torch.nn.functional.l1_loss(rendered, target)
        if not torch.isfinite(loss):
            raise GaussianTrainingError(f"training produced a non-finite loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
        if (step + 1) % report_every == 0 or step == 0:
            progress(f"  Step {step + 1}/{steps}: L1 loss {loss.item():.4f}")

    with torch.no_grad():
        final_render, preview_target = render(preview_camera)
        final_loss = torch.nn.functional.l1_loss(final_render, preview_target).item()
    if final_loss >= initial_loss:
        raise GaussianTrainingError(
            "preview loss did not decrease: "
            f"{initial_loss:.6f} -> {final_loss:.6f}"
        )

    reference_array = preview_target[0].detach().cpu().numpy()
    render_array = final_render[0].detach().cpu().clamp(0, 1).numpy()
    initial_render_array = initial_render[0].detach().cpu().clamp(0, 1).numpy()
    Image.fromarray((reference_array * 255).astype(np.uint8)).save(
        output_path / "reference.png"
    )
    Image.fromarray((initial_render_array * 255).astype(np.uint8)).save(
        output_path / "initial-render.png"
    )
    Image.fromarray((render_array * 255).astype(np.uint8)).save(
        output_path / "render.png"
    )
    create_render_comparison(
        reference_array,
        render_array,
        output_path / "comparison.png",
    )

    checkpoint_path = output_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 1,
            "steps": steps,
            "image_scale": image_scale,
            "world_center": torch.from_numpy(scene.world_center),
            "world_scale": scene.world_scale,
            "means": means.detach().cpu(),
            "scales": scales.detach().cpu(),
            "quaternions": quaternions.detach().cpu(),
            "opacities": opacities.detach().cpu(),
            "color_logits": color_logits.detach().cpu(),
        },
        checkpoint_path,
    )
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    device_name = torch.cuda.get_device_name()
    metrics = {
        "steps": steps,
        "cameras": len(scene.cameras),
        "gaussians": len(means),
        "initial_preview_l1": initial_loss,
        "final_preview_l1": final_loss,
        "last_training_l1": training_losses[-1],
        "peak_gpu_memory_gb": peak_memory_gb,
        "device": device_name,
        "preview_camera": preview_camera.name,
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    return GaussianTrainingResult(
        steps=steps,
        cameras=len(scene.cameras),
        gaussians=len(means),
        initial_loss=initial_loss,
        final_loss=final_loss,
        peak_memory_gb=peak_memory_gb,
        device_name=device_name,
        output=output_path,
    )


def format_gaussian_training_report(result: GaussianTrainingResult) -> str:
    return "\n".join(
        [
            "Capture Studio Gaussian Splatting smoke training",
            f"[OK] Device: {result.device_name}",
            f"[OK] Cameras loaded: {result.cameras}",
            f"[OK] Gaussians: {result.gaussians:,}",
            f"[OK] Training steps: {result.steps}",
            f"[OK] Preview L1 loss: {result.initial_loss:.4f} -> {result.final_loss:.4f}",
            f"[OK] Peak GPU memory: {result.peak_memory_gb:.2f} GB",
            f"[OK] Checkpoint: {result.output / 'checkpoint.pt'}",
            f"[OK] Comparison: {result.output / 'comparison.png'}",
            f"[OK] Metrics: {result.output / 'metrics.json'}",
        ]
    )
