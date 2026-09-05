from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class ModelExportError(RuntimeError):
    """Raised when a trained Gaussian checkpoint cannot be exported."""


@dataclass(frozen=True)
class ModelExportResult:
    checkpoint: Path
    output: Path
    gaussians: int
    size_bytes: int


def export_gaussian_ply(checkpoint_path: Path, output_path: Path) -> ModelExportResult:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise ModelExportError(f"checkpoint does not exist: {checkpoint_path}")
    if output_path.suffix.lower() != ".ply":
        raise ModelExportError("output filename must end with .ply")
    if output_path.exists():
        raise ModelExportError(f"output file already exists: {output_path}")

    try:
        import torch
        from gsplat import export_splats
    except ImportError as error:
        raise ModelExportError("gsplat and PyTorch must be installed") from error

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ModelExportError(f"could not load checkpoint: {error}") from error

    if checkpoint.get("format_version") == 3:
        parameters = checkpoint["parameters"]
        checkpoint = {**checkpoint, **parameters, "quaternions": parameters["quats"]}
    has_sh = "sh0" in checkpoint and "shN" in checkpoint
    color_keys = ("sh0", "shN") if has_sh else ("color_logits",)
    required = ("means", "scales", "quaternions", "opacities", *color_keys)
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise ModelExportError(
            f"checkpoint is missing required tensors: {', '.join(missing)}"
        )

    means = checkpoint["means"].float()
    scales = checkpoint["scales"].float()
    quaternions = checkpoint["quaternions"].float()
    opacities = checkpoint["opacities"].float()
    count = len(means)
    expected_shapes = {
        "means": (count, 3),
        "scales": (count, 3),
        "quaternions": (count, 4),
        "opacities": (count,),
    }
    tensors = {
        "means": means,
        "scales": scales,
        "quaternions": quaternions,
        "opacities": opacities,
    }
    if has_sh:
        degree = checkpoint.get("sh_degree", 3)
        if degree not in (0, 1, 2, 3):
            raise ModelExportError("SH degree must be between 0 and 3")
        expected_shapes.update({"sh0": (count, 1, 3), "shN": (count, 15, 3)})
    else:
        expected_shapes["color_logits"] = (count, 3)
    tensors.update({key: checkpoint[key].float() for key in color_keys})
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ModelExportError(
                f"checkpoint tensor {name} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shapes[name]}"
            )
        if not torch.isfinite(tensor).all():
            raise ModelExportError(f"checkpoint tensor {name} contains non-finite values")

    quaternion_norms = torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    if (quaternion_norms == 0).any():
        raise ModelExportError("checkpoint contains a zero-length quaternion")
    quaternions = quaternions / quaternion_norms

    if has_sh:
        spherical_harmonics_dc = tensors["sh0"]
        spherical_harmonics_rest = tensors["shN"][:, :(degree + 1) ** 2 - 1]
    else:
        rgb = torch.sigmoid(tensors["color_logits"])
        spherical_harmonics_dc = ((rgb - 0.5) / 0.28209479177387814).unsqueeze(1)
        spherical_harmonics_rest = torch.empty((count, 0, 3), dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=means,
        scales=scales,
        quats=quaternions,
        opacities=opacities,
        sh0=spherical_harmonics_dc,
        shN=spherical_harmonics_rest,
        format="ply",
        save_to=str(output_path),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ModelExportError("gsplat did not create the PLY file")

    return ModelExportResult(
        checkpoint=checkpoint_path,
        output=output_path,
        gaussians=count,
        size_bytes=output_path.stat().st_size,
    )


def format_model_export_report(result: ModelExportResult) -> str:
    return "\n".join(
        [
            "Capture Studio Gaussian model export",
            f"[OK] Gaussians: {result.gaussians:,}",
            f"[OK] PLY size: {result.size_bytes / 1024**2:.2f} MB",
            f"[OK] Source checkpoint: {result.checkpoint}",
            f"[OK] Standard PLY: {result.output}",
        ]
    )
