from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from capture_studio.colmap_check import find_colmap
from capture_studio.photo_analysis import SUPPORTED_EXTENSIONS


class ReconstructionError(RuntimeError):
    """Raised when COLMAP cannot produce a sparse reconstruction."""


@dataclass(frozen=True)
class ReconstructionResult:
    input_images: int
    registered_images: int
    cameras: int
    points: int
    mean_reprojection_error: float
    output: Path
    model: Path
    preview: Path


ProgressReporter = Callable[[str], None]


def _run_stage(
    executable: Path,
    arguments: list[str],
    name: str,
    log_path: Path,
    progress: ProgressReporter,
) -> str:
    progress(name)
    completed = subprocess.run(
        [str(executable), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    output = completed.stdout + completed.stderr
    log_path.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-20:])
        raise ReconstructionError(f"{name} failed; see {log_path}\n{tail}")
    return output


def _integer_metric(output: str, label: str) -> int:
    match = re.search(rf"{re.escape(label)}:\s+(\d+)", output)
    return int(match.group(1)) if match else 0


def _float_metric(output: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}:\s+([0-9.]+)", output)
    return float(match.group(1)) if match else 0.0


def parse_model_metrics(output: str) -> tuple[int, int, float]:
    return (
        _integer_metric(output, "Registered images"),
        _integer_metric(output, "Points"),
        _float_metric(output, "Mean reprojection error"),
    )


def _header_count(path: Path, item_name: str) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^# Number of {re.escape(item_name)}:\s+(\d+)", text, re.MULTILINE
    )
    if not match:
        raise ReconstructionError(f"could not read the model count from {path}")
    return int(match.group(1))


def read_text_model(model_path: Path) -> tuple[int, int, int]:
    cameras = _header_count(model_path / "cameras.txt", "cameras")
    images = _header_count(model_path / "images.txt", "images")
    points = _header_count(model_path / "points3D.txt", "points")
    return cameras, images, points


def _read_sparse_points(points_path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    colors: list[list[int]] = []
    for line in points_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        positions.append([float(value) for value in tokens[1:4]])
        colors.append([int(value) for value in tokens[4:7]])
    return np.asarray(positions, dtype=np.float64), np.asarray(colors, dtype=np.uint8)


def create_sparse_preview(points_path: Path, preview_path: Path) -> None:
    positions, colors = _read_sparse_points(points_path)
    if len(positions) < 3:
        raise ReconstructionError("not enough sparse points to create a preview")

    if len(positions) > 100_000:
        selected = np.linspace(0, len(positions) - 1, 100_000, dtype=np.int64)
        positions = positions[selected]
        colors = colors[selected]

    centered = positions - np.median(positions, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    lower = np.percentile(projected, 1, axis=0)
    upper = np.percentile(projected, 99, axis=0)
    span = np.maximum(upper - lower, 1e-9)
    normalized = np.clip((projected - lower) / span, 0, 1)

    width, height, margin = 1200, 800, 35
    x = margin + normalized[:, 0] * (width - 2 * margin)
    y = height - margin - normalized[:, 1] * (height - 2 * margin)

    preview = Image.new("RGB", (width, height), "#111827")
    draw = ImageDraw.Draw(preview)
    for point_x, point_y, color in zip(x, y, colors, strict=True):
        red, green, blue = (int(channel) for channel in color)
        draw.point((int(point_x), int(point_y)), fill=(red, green, blue))
    draw.text(
        (15, 12),
        f"COLMAP sparse reconstruction | {len(positions):,} displayed points",
        fill="white",
    )
    preview.save(preview_path)


def run_reconstruction(
    image_path: Path,
    output_path: Path,
    progress: ProgressReporter = print,
) -> ReconstructionResult:
    image_path = image_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not image_path.is_dir():
        raise ReconstructionError(f"image folder does not exist: {image_path}")
    if output_path == image_path or image_path in output_path.parents:
        raise ReconstructionError("output folder must be outside the image folder")
    if output_path.exists() and any(output_path.iterdir()):
        raise ReconstructionError(f"output folder is not empty: {output_path}")

    image_count = sum(
        1
        for path in image_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if image_count < 2:
        raise ReconstructionError("at least two supported images are required")

    executable = find_colmap()
    if executable is None:
        raise ReconstructionError(
            "COLMAP was not found; run scripts/install-colmap-windows.ps1"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    logs_path = output_path / "logs"
    sparse_path = output_path / "sparse"
    logs_path.mkdir()
    sparse_path.mkdir()
    database_path = output_path / "database.db"

    _run_stage(
        executable,
        [
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--ImageReader.single_camera",
            "1",
            "--FeatureExtraction.use_gpu",
            "1",
            "--FeatureExtraction.gpu_index",
            "0",
            "--log_target",
            "stdout",
        ],
        "[1/3] Extracting image features on GPU 0...",
        logs_path / "01-feature-extraction.log",
        progress,
    )
    _run_stage(
        executable,
        [
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--FeatureMatching.use_gpu",
            "1",
            "--FeatureMatching.gpu_index",
            "0",
            "--log_target",
            "stdout",
        ],
        "[2/3] Matching image features on GPU 0...",
        logs_path / "02-feature-matching.log",
        progress,
    )
    _run_stage(
        executable,
        [
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--output_path",
            str(sparse_path),
            "--log_target",
            "stdout",
        ],
        "[3/3] Estimating cameras and sparse 3D points...",
        logs_path / "03-mapping.log",
        progress,
    )

    models = sorted(
        path for path in sparse_path.iterdir() if (path / "cameras.bin").is_file()
    )
    if not models:
        raise ReconstructionError(
            f"COLMAP produced no sparse model; see {logs_path / '03-mapping.log'}"
        )

    analyzed_models: list[tuple[int, int, Path, float]] = []
    for index, model in enumerate(models):
        analysis = _run_stage(
            executable,
            ["model_analyzer", "--path", str(model), "--log_target", "stdout"],
            f"Analyzing sparse model {model.name}...",
            logs_path / f"04-model-{index}-analysis.log",
            progress,
        )
        registered, points, mean_error = parse_model_metrics(analysis)
        analyzed_models.append((registered, points, model, mean_error))
    _, _, best_model, mean_reprojection_error = max(
        analyzed_models, key=lambda candidate: candidate[:2]
    )

    text_model = output_path / "sparse-text"
    text_model.mkdir()
    _run_stage(
        executable,
        [
            "model_converter",
            "--input_path",
            str(best_model),
            "--output_path",
            str(text_model),
            "--output_type",
            "TXT",
        ],
        "Exporting the best model as readable text...",
        logs_path / "05-model-conversion.log",
        progress,
    )

    cameras, registered_images, points = read_text_model(text_model)
    preview_path = output_path / "sparse-preview.png"
    create_sparse_preview(text_model / "points3D.txt", preview_path)

    return ReconstructionResult(
        input_images=image_count,
        registered_images=registered_images,
        cameras=cameras,
        points=points,
        mean_reprojection_error=mean_reprojection_error,
        output=output_path,
        model=best_model,
        preview=preview_path,
    )


def format_reconstruction_report(result: ReconstructionResult) -> str:
    registration_rate = result.registered_images / result.input_images * 100
    return "\n".join(
        [
            "Capture Studio sparse reconstruction",
            f"[OK] Registered images: {result.registered_images}/{result.input_images} ({registration_rate:.1f}%)",
            f"[OK] Cameras: {result.cameras}",
            f"[OK] Sparse 3D points: {result.points:,}",
            f"[OK] Mean reprojection error: {result.mean_reprojection_error:.3f} px",
            f"[OK] Binary model: {result.model}",
            f"[OK] Text model: {result.output / 'sparse-text'}",
            f"[OK] Preview: {result.preview}",
            f"[OK] Logs: {result.output / 'logs'}",
        ]
    )
