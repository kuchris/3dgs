from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from capture_studio.colmap_scene import ColmapSceneError, load_colmap_scene
from capture_studio.gaussian_training import (
    GaussianTrainingError,
    GaussianTrainingResult,
    _load_image,
    _validate_output,
    create_render_comparison,
)


def camera_split(count: int) -> tuple[list[int], list[int]]:
    if count < 2:
        raise GaussianTrainingError("quality training needs at least two cameras")
    validation = list(range(0, count, 8))
    training = [index for index in range(count) if index not in validation]
    return training, validation


def camera_for_step(indices: list[int], step: int, seed: int = 42) -> int:
    # Every photograph appears once per shuffled epoch; resume needs only the step.
    epoch, offset = divmod(step, len(indices))
    order = indices.copy()
    random.Random(seed + epoch).shuffle(order)
    return order[offset]


def quality_schedule(steps: int) -> dict[str, int]:
    return {
        "refine_start_iter": min(500, steps // 10),
        "refine_stop_iter": min(15000, steps * 3 // 4),
        "refine_every": min(100, max(1, steps // 10)),
        "reset_every": 3000,
        "sh_interval": min(1000, max(1, steps // 4)),
    }


def image_metrics(rendered, target) -> dict[str, float]:
    import torch
    from gsplat.losses import ssim_loss

    rendered = rendered.clamp(0, 1)
    mse = torch.mean((rendered - target) ** 2).item()
    return {
        "l1": torch.mean(torch.abs(rendered - target)).item(),
        "psnr": -10 * math.log10(max(mse, 1e-12)),
        "ssim": 1
        - ssim_loss(rendered.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)).item(),
    }


def dataset_digest(data_path: Path) -> str:
    digest = hashlib.sha256()
    for folder in ("sparse-text", "images"):
        for path in sorted((data_path / folder).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(data_path).as_posix().encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest()


def _to_device(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def train_quality_v2(
    data_path: Path,
    output_path: Path,
    steps: int = 30000,
    image_scale: int = 1,
    progress=print,
    resume: Path | None = None,
    checkpoint_every: int = 1000,
    stop_after: int | None = None,
) -> GaussianTrainingResult:
    import torch
    from gsplat.init_utils import knn_scale_init
    from gsplat.losses import ssim_loss
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    if steps < 201:
        raise GaussianTrainingError("quality training requires at least 201 steps")
    if image_scale < 1 or checkpoint_every < 1:
        raise GaussianTrainingError(
            "image scale and checkpoint interval must be positive"
        )
    if stop_after is not None and not 1 <= stop_after <= steps:
        raise GaussianTrainingError("stop-after must be between 1 and total steps")
    if not torch.cuda.is_available():
        raise GaussianTrainingError("PyTorch cannot access a CUDA device")
    output_path = output_path.expanduser().resolve()
    _validate_output(output_path)
    data_path = data_path.expanduser().resolve()
    try:
        scene = load_colmap_scene(data_path)
    except ColmapSceneError as error:
        raise GaussianTrainingError(str(error)) from error
    training, validation = camera_split(len(scene.cameras))
    schedule = quality_schedule(steps)
    config = {
        "steps": steps,
        "image_scale": image_scale,
        "schedule": schedule,
        "dataset_sha256": dataset_digest(data_path),
        "training_cameras": [scene.cameras[i].name for i in training],
        "validation_cameras": [scene.cameras[i].name for i in validation],
    }
    checkpoint = None
    if resume is not None:
        try:
            checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        except Exception as error:
            raise GaussianTrainingError(
                f"could not load resume checkpoint: {error}"
            ) from error
        if checkpoint.get("format_version") != 3 or checkpoint.get("config") != config:
            raise GaussianTrainingError(
                "resume requires a v2 trainer checkpoint with matching data and settings"
            )
    start = checkpoint["steps"] if checkpoint else 0
    end = stop_after if stop_after is not None else steps
    if end <= start:
        raise GaussianTrainingError("stop-after/total steps must exceed the saved step")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    if checkpoint:
        parameters = {
            key: torch.nn.Parameter(value.to(device))
            for key, value in checkpoint["parameters"].items()
        }
    else:
        means = torch.from_numpy(scene.points).to(device)
        with torch.no_grad():
            scales = knn_scale_init(means, k=3, chunk_size=1024)
        parameters = {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(means), 1)
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((len(means),), 0.1, device=device))
            ),
            "sh0": torch.nn.Parameter(
                (
                    (torch.from_numpy(scene.colors).to(device) - 0.5)
                    / 0.28209479177387814
                )[:, None]
            ),
            "shN": torch.nn.Parameter(torch.zeros((len(means), 15, 3), device=device)),
        }
    rates = {
        "means": 1.6e-4,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        key: torch.optim.Adam([value], lr=rates[key], eps=1e-15, fused=True)
        for key, value in parameters.items()
    }
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1 / steps)
    )
    strategy = DefaultStrategy(
        **{key: value for key, value in schedule.items() if key != "sh_interval"}
    )
    strategy.check_sanity(parameters, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=1.0)
    if checkpoint:
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][key])
        scheduler.load_state_dict(checkpoint["scheduler"])
        strategy_state = _to_device(checkpoint["strategy_state"], device)
        torch.set_rng_state(checkpoint["cpu_rng"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    def report(message):
        progress(message)
        with (output_path / "training.log").open("a", encoding="utf-8") as log:
            log.write(message + "\n")

    report(
        f"Quality v2: {len(training)} training / {len(validation)} evaluation photos; steps {start} -> {end}/{steps}"
    )
    report("Caching resized photographs in CPU memory...")
    images = [_load_image(camera, image_scale) for camera in scene.cameras]
    active_degree = min(3, max(0, start - 1) // schedule["sh_interval"])

    def render(index):
        target, intrinsics = images[index]
        camera = scene.cameras[index]
        target_tensor = torch.from_numpy(target).to(device)[None]
        colors, _, info = rasterization(
            means=parameters["means"],
            quats=torch.nn.functional.normalize(parameters["quats"], dim=-1),
            scales=parameters["scales"].exp(),
            opacities=parameters["opacities"].sigmoid(),
            colors=torch.cat([parameters["sh0"], parameters["shN"]], dim=1),
            sh_degree=active_degree,
            viewmats=torch.from_numpy(camera.world_to_camera).to(device)[None],
            Ks=torch.from_numpy(intrinsics).to(device)[None],
            width=target.shape[1],
            height=target.shape[0],
            packed=True,
            backgrounds=torch.ones((1, 3), device=device),
        )
        return colors, target_tensor, info

    preview_index = validation[0]
    with torch.no_grad():
        initial_render, target, _ = render(preview_index)
        initial_loss = (
            checkpoint["initial_preview_l1"]
            if checkpoint
            else image_metrics(initial_render, target)["l1"]
        )
    elapsed_before = checkpoint.get("elapsed_seconds", 0.0) if checkpoint else 0.0
    started = time.monotonic()

    def save(completed):
        payload = {
            "format_version": 3,
            "mode": "quality-v2",
            "steps": completed,
            "image_scale": image_scale,
            "sh_degree": active_degree,
            "config": config,
            "world_center": torch.from_numpy(scene.world_center),
            "world_scale": scene.world_scale,
            "parameters": {
                key: value.detach().cpu() for key, value in parameters.items()
            },
            "optimizers": {
                key: value.state_dict() for key, value in optimizers.items()
            },
            "scheduler": scheduler.state_dict(),
            "strategy_state": strategy_state,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "initial_preview_l1": initial_loss,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
        }
        temporary = output_path / "checkpoint.tmp"
        torch.save(payload, temporary)
        temporary.replace(output_path / "checkpoint.pt")
        report(
            f"Saved resumable checkpoint at step {completed}: {output_path / 'checkpoint.pt'}"
        )

    report_every = min(100, max(1, steps // 10))
    for step in range(start, end):
        active_degree = min(3, step // schedule["sh_interval"])
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        rendered, target, info = render(camera_for_step(training, step))
        strategy.step_pre_backward(parameters, optimizers, strategy_state, step, info)
        l1 = torch.nn.functional.l1_loss(rendered, target)
        structural = ssim_loss(
            rendered.clamp(0, 1).permute(0, 3, 1, 2), target.permute(0, 3, 1, 2)
        )
        loss = 0.8 * l1 + 0.2 * structural
        if not torch.isfinite(loss):
            raise GaussianTrainingError(
                f"non-finite loss at step {step + 1}; resume the last checkpoint"
            )
        loss.backward()
        # Densification can replace parameters, so update the current tensors first.
        for optimizer in optimizers.values():
            optimizer.step()
        scheduler.step()
        strategy.step_post_backward(
            parameters, optimizers, strategy_state, step, info, packed=True
        )
        if (step + 1) % report_every == 0 or step == start:
            elapsed = time.monotonic() - started
            eta = elapsed / (step + 1 - start) * (end - step - 1)
            report(
                f"Step {step + 1}/{steps}: L1 {l1.item():.4f}; {len(parameters['means']):,} Gaussians; SH {active_degree}; ETA {eta / 60:.1f} min"
            )
        if (step + 1) % checkpoint_every == 0 or step + 1 == end:
            save(step + 1)

    report("Evaluating photographs excluded from training...")
    evaluations = []
    with torch.no_grad():
        for index in validation:
            rendered, target, _ = render(index)
            scores = image_metrics(rendered, target)
            evaluations.append({"camera": scene.cameras[index].name, **scores})
            create_render_comparison(
                target[0].cpu().numpy(),
                rendered[0].clamp(0, 1).cpu().numpy(),
                output_path / f"evaluation-{index:03d}.png",
                "Held-out view: Quality v2",
            )
        rendered, target, _ = render(preview_index)
        final_loss = image_metrics(rendered, target)["l1"]
        for name, tensor in (("reference", target), ("render", rendered)):
            Image.fromarray(
                (tensor[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            ).save(output_path / f"{name}.png")
        create_render_comparison(
            target[0].cpu().numpy(),
            rendered[0].clamp(0, 1).cpu().numpy(),
            output_path / "comparison.png",
            "Held-out view: Quality v2",
        )
    metrics = {
        "mode": "quality-v2",
        "steps": end,
        "planned_steps": steps,
        "complete": end == steps,
        "cameras": len(scene.cameras),
        "training_cameras": len(training),
        "evaluation_cameras": len(validation),
        "image_scale": image_scale,
        "sh_degree": active_degree,
        "initial_gaussians": len(scene.points),
        "final_gaussians": len(parameters["means"]),
        "initial_preview_l1": initial_loss,
        "final_preview_l1": final_loss,
        "evaluation_mean": {
            key: float(np.mean([item[key] for item in evaluations]))
            for key in ("l1", "psnr", "ssim")
        },
        "evaluation_per_camera": evaluations,
        "evaluation_note": "Images excluded from photometric training; COLMAP poses and sparse initialization use all images.",
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": elapsed_before + time.monotonic() - started,
        "device": torch.cuda.get_device_name(),
    }
    (output_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    report(f"Evaluation mean: {metrics['evaluation_mean']}")
    return GaussianTrainingResult(
        "quality-v2",
        end,
        len(scene.cameras),
        len(scene.points),
        len(parameters["means"]),
        initial_loss,
        final_loss,
        metrics["peak_gpu_memory_gb"],
        metrics["device"],
        output_path,
    )
