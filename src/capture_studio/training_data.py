from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from capture_studio.colmap_check import find_colmap
from capture_studio.photo_analysis import SUPPORTED_EXTENSIONS
from capture_studio.reconstruction import read_text_model


class TrainingDataError(RuntimeError):
    """Raised when COLMAP cannot prepare training data."""


@dataclass(frozen=True)
class TrainingDataResult:
    source_images: int
    prepared_images: int
    registered_images: int
    cameras: int
    points: int
    max_image_dimension: int
    output: Path


ProgressReporter = Callable[[str], None]


def _image_files(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _run_stage(
    executable: Path,
    arguments: list[str],
    name: str,
    log_path: Path,
    progress: ProgressReporter,
) -> None:
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
        raise TrainingDataError(f"{name} failed; see {log_path}\n{tail}")


def prepare_training_data(
    image_path: Path,
    model_path: Path,
    output_path: Path,
    max_image_size: int = 1600,
    progress: ProgressReporter = print,
) -> TrainingDataResult:
    image_path = image_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not image_path.is_dir():
        raise TrainingDataError(f"image folder does not exist: {image_path}")
    if not model_path.is_dir():
        raise TrainingDataError(f"sparse model folder does not exist: {model_path}")
    required_model_files = ["cameras.bin", "images.bin", "points3D.bin"]
    missing = [name for name in required_model_files if not (model_path / name).is_file()]
    if missing:
        raise TrainingDataError(
            f"sparse model is missing required files: {', '.join(missing)}"
        )
    if max_image_size < 1:
        raise TrainingDataError("maximum image size must be a positive integer")
    if output_path == image_path or image_path in output_path.parents:
        raise TrainingDataError("output folder must be outside the image folder")
    if output_path.exists() and any(output_path.iterdir()):
        raise TrainingDataError(f"output folder is not empty: {output_path}")

    source_images = _image_files(image_path)
    if not source_images:
        raise TrainingDataError("image folder contains no supported photos")

    executable = find_colmap()
    if executable is None:
        raise TrainingDataError(
            "COLMAP was not found; run scripts/install-colmap-windows.ps1"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    logs_path = output_path / "logs"
    logs_path.mkdir()

    _run_stage(
        executable,
        [
            "image_undistorter",
            "--image_path",
            str(image_path),
            "--input_path",
            str(model_path),
            "--output_path",
            str(output_path),
            "--output_type",
            "COLMAP",
            "--copy_policy",
            "COPY",
            "--max_image_size",
            str(max_image_size),
            "--log_target",
            "stdout",
        ],
        "[1/2] Undistorting and resizing registered images...",
        logs_path / "01-image-undistortion.log",
        progress,
    )

    binary_model = output_path / "sparse"
    text_model = output_path / "sparse-text"
    text_model.mkdir()
    _run_stage(
        executable,
        [
            "model_converter",
            "--input_path",
            str(binary_model),
            "--output_path",
            str(text_model),
            "--output_type",
            "TXT",
        ],
        "[2/2] Exporting the prepared camera model as readable text...",
        logs_path / "02-model-conversion.log",
        progress,
    )

    cameras, registered_images, points = read_text_model(text_model)
    prepared_files = _image_files(output_path / "images")
    if len(prepared_files) != registered_images:
        raise TrainingDataError(
            "prepared image count does not match the registered camera count: "
            f"{len(prepared_files)} != {registered_images}"
        )

    max_dimension = 0
    for path in prepared_files:
        with Image.open(path) as image:
            max_dimension = max(max_dimension, *image.size)
    if max_dimension > max_image_size:
        raise TrainingDataError(
            f"prepared image dimension {max_dimension} exceeds {max_image_size}"
        )

    return TrainingDataResult(
        source_images=len(source_images),
        prepared_images=len(prepared_files),
        registered_images=registered_images,
        cameras=cameras,
        points=points,
        max_image_dimension=max_dimension,
        output=output_path,
    )


def format_training_data_report(result: TrainingDataResult) -> str:
    return "\n".join(
        [
            "Capture Studio training-data preparation",
            f"[OK] Prepared images: {result.prepared_images}/{result.source_images}",
            f"[OK] Registered images: {result.registered_images}",
            f"[OK] Cameras: {result.cameras}",
            f"[OK] Sparse 3D points: {result.points:,}",
            f"[OK] Largest prepared dimension: {result.max_image_dimension} px",
            f"[OK] Training images: {result.output / 'images'}",
            f"[OK] Binary model: {result.output / 'sparse'}",
            f"[OK] Text model: {result.output / 'sparse-text'}",
            f"[OK] Logs: {result.output / 'logs'}",
        ]
    )
