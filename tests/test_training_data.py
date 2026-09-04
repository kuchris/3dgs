from pathlib import Path

import pytest

from capture_studio.training_data import (
    TrainingDataError,
    TrainingDataResult,
    format_training_data_report,
    prepare_training_data,
)


def test_report_explains_prepared_outputs(tmp_path: Path) -> None:
    result = TrainingDataResult(
        source_images=128,
        prepared_images=128,
        registered_images=128,
        cameras=1,
        points=84_004,
        max_image_dimension=1600,
        output=tmp_path,
    )

    report = format_training_data_report(result)

    assert "[OK] Prepared images: 128/128" in report
    assert "[OK] Sparse 3D points: 84,004" in report
    assert "[OK] Largest prepared dimension: 1600 px" in report
    assert f"[OK] Training images: {tmp_path / 'images'}" in report


def test_refuses_incomplete_sparse_model(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "photo.jpg").write_bytes(b"photo")
    model = tmp_path / "model"
    model.mkdir()

    with pytest.raises(TrainingDataError, match="missing required files"):
        prepare_training_data(images, model, tmp_path / "output")


def test_refuses_nonempty_output_folder(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "photo.jpg").write_bytes(b"photo")
    model = tmp_path / "model"
    model.mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (model / name).write_bytes(b"model")
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(TrainingDataError, match="output folder is not empty"):
        prepare_training_data(images, model, output)
