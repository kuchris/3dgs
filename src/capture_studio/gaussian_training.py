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
    mode: str
    steps: int
    cameras: int
    initial_gaussians: int
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
    render_label: str = "Gaussian render after smoke training",
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
    draw.text((width + 12, 10), render_label, fill="white")
    canvas.save(output_path)


def _validate_output(output_path: Path) -> None:
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise GaussianTrainingError(f"output folder is not empty: {output_path}")


def _train_gaussians(
    data_path: Path,
    output_path: Path,
    steps: int,
    image_scale: int,
    densify: bool,
    mode: str,
    progress: ProgressReporter,
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
        from gsplat.losses import ssim_loss
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy
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
    initial_gaussians = len(scene.points)
    means = torch.nn.Parameter(torch.from_numpy(scene.points).to(device))
    with torch.no_grad():
        initial_log_scales = knn_scale_init(means, k=3, chunk_size=1024)
    parameters = {
        "means": means,
        "scales": torch.nn.Parameter(initial_log_scales[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(means), 1)
        ),
        "opacities": torch.nn.Parameter(
            torch.logit(torch.full((len(means),), 0.1, device=device))
        ),
        "colors": torch.nn.Parameter(
            torch.logit(torch.from_numpy(scene.colors).to(device).clamp(0.01, 0.99))
        ),
    }
    learning_rates = {
        "means": 1.6e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "colors": 2.5e-3,
    }
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [parameter], "lr": learning_rates[name]}],
            eps=1e-15,
            fused=True,
        )
        for name, parameter in parameters.items()
    }
    means_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / steps)
    )
    strategy = None
    strategy_state = None
    if densify:
        strategy = DefaultStrategy(
            refine_start_iter=100,
            refine_stop_iter=steps,
            refine_every=100,
            verbose=True,
        )
        strategy.check_sanity(parameters, optimizers)
        strategy_state = strategy.initialize_state(scene_scale=1.0)

    def render(camera: SceneCamera):
        target, intrinsics = _load_image(camera, image_scale)
        target_tensor = torch.from_numpy(target).to(device).unsqueeze(0)
        viewmat = torch.from_numpy(camera.world_to_camera).to(device).unsqueeze(0)
        intrinsic_tensor = torch.from_numpy(intrinsics).to(device).unsqueeze(0)
        height, width = target.shape[:2]
        colors, _, info = rasterization(
            means=parameters["means"],
            quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
            scales=torch.exp(parameters["scales"]),
            opacities=torch.sigmoid(parameters["opacities"]),
            colors=torch.sigmoid(parameters["colors"]),
            viewmats=viewmat,
            Ks=intrinsic_tensor,
            width=width,
            height=height,
            packed=True,
            backgrounds=torch.ones((1, 3), device=device),
        )
        return colors, target_tensor, info

    preview_camera = scene.cameras[0]
    with torch.no_grad():
        initial_render, preview_target, _ = render(preview_camera)
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
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        rendered, target, info = render(camera)
        if strategy is not None:
            strategy.step_pre_backward(
                parameters, optimizers, strategy_state, step, info
            )
        l1_loss = torch.nn.functional.l1_loss(rendered, target)
        if densify:
            structural_loss = ssim_loss(
                rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
            )
            loss = torch.lerp(l1_loss, structural_loss, 0.2)
        else:
            loss = l1_loss
        if not torch.isfinite(loss):
            raise GaussianTrainingError(f"training produced a non-finite loss at step {step + 1}")
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        means_scheduler.step()
        if strategy is not None:
            strategy.step_post_backward(
                parameters,
                optimizers,
                strategy_state,
                step,
                info,
                packed=True,
            )
        training_losses.append(l1_loss.item())
        if (step + 1) % report_every == 0 or step == 0:
            progress(
                f"  Step {step + 1}/{steps}: L1 loss {l1_loss.item():.4f}; "
                f"Gaussians {len(parameters['means']):,}"
            )

    with torch.no_grad():
        final_render, preview_target, _ = render(preview_camera)
        final_loss = torch.nn.functional.l1_loss(final_render, preview_target).item()
    if final_loss >= initial_loss:
        raise GaussianTrainingError(
            "preview loss did not decrease: "
            f"{initial_loss:.6f} -> {final_loss:.6f}"
        )
    if densify and len(parameters["means"]) == initial_gaussians:
        raise GaussianTrainingError("densification did not change the Gaussian count")

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
        render_label=f"Gaussian render after {mode} training",
    )

    checkpoint_path = output_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 2,
            "mode": mode,
            "steps": steps,
            "image_scale": image_scale,
            "world_center": torch.from_numpy(scene.world_center),
            "world_scale": scene.world_scale,
            "means": parameters["means"].detach().cpu(),
            "scales": parameters["scales"].detach().cpu(),
            "quaternions": parameters["quats"].detach().cpu(),
            "opacities": parameters["opacities"].detach().cpu(),
            "color_logits": parameters["colors"].detach().cpu(),
        },
        checkpoint_path,
    )
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    device_name = torch.cuda.get_device_name()
    metrics = {
        "steps": steps,
        "cameras": len(scene.cameras),
        "densification": densify,
        "initial_gaussians": initial_gaussians,
        "final_gaussians": len(parameters["means"]),
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
        mode=mode,
        steps=steps,
        cameras=len(scene.cameras),
        initial_gaussians=initial_gaussians,
        gaussians=len(parameters["means"]),
        initial_loss=initial_loss,
        final_loss=final_loss,
        peak_memory_gb=peak_memory_gb,
        device_name=device_name,
        output=output_path,
    )


def train_gaussian_smoke_test(
    data_path: Path,
    output_path: Path,
    steps: int = 100,
    image_scale: int = 4,
    progress: ProgressReporter = print,
) -> GaussianTrainingResult:
    return _train_gaussians(
        data_path,
        output_path,
        steps=steps,
        image_scale=image_scale,
        densify=False,
        mode="smoke",
        progress=progress,
    )


def train_gaussian_quality(
    data_path: Path,
    output_path: Path,
    steps: int = 1000,
    image_scale: int = 2,
    progress: ProgressReporter = print,
) -> GaussianTrainingResult:
    if steps < 201:
        raise GaussianTrainingError(
            "quality training requires at least 201 steps for densification"
        )
    return _train_gaussians(
        data_path,
        output_path,
        steps=steps,
        image_scale=image_scale,
        densify=True,
        mode="quality",
        progress=progress,
    )


def format_gaussian_training_report(result: GaussianTrainingResult) -> str:
    return "\n".join(
        [
            f"Capture Studio Gaussian Splatting {result.mode} training",
            f"[OK] Device: {result.device_name}",
            f"[OK] Cameras loaded: {result.cameras}",
            f"[OK] Gaussians: {result.initial_gaussians:,} -> {result.gaussians:,}",
            f"[OK] Training steps: {result.steps}",
            f"[OK] Preview L1 loss: {result.initial_loss:.4f} -> {result.final_loss:.4f}",
            f"[OK] Peak GPU memory: {result.peak_memory_gb:.2f} GB",
            f"[OK] Checkpoint: {result.output / 'checkpoint.pt'}",
            f"[OK] Comparison: {result.output / 'comparison.png'}",
            f"[OK] Metrics: {result.output / 'metrics.json'}",
        ]
    )
