from pathlib import Path

import pytest
from PIL import Image

from capture_studio.reconstruction import (
    ReconstructionError,
    ReconstructionResult,
    create_sparse_preview,
    format_reconstruction_report,
    parse_model_metrics,
    read_text_model,
    run_reconstruction,
)


def test_parses_timestamped_model_analyzer_metrics() -> None:
    output = (
        "I20260904 22:59:23 model.cc:446] Registered images: 128\n"
        "I20260904 22:59:23 model.cc:448] Points: 84004\n"
        "I20260904 22:59:23 model.cc:456] Mean reprojection error: 0.611949px\n"
    )

    assert parse_model_metrics(output) == (128, 84004, 0.611949)


def test_reads_colmap_text_model_counts(tmp_path: Path) -> None:
    (tmp_path / "cameras.txt").write_text(
        "# Number of cameras: 1\n1 PINHOLE 1920 1080 1000 1000 960 540\n",
        encoding="utf-8",
    )
    (tmp_path / "images.txt").write_text(
        "# Number of images: 2, mean observations per image: 1\n"
        "1 1 0 0 0 0 0 0 1 first.jpg\n0.5 0.5 1\n"
        "2 1 0 0 0 1 0 0 1 second.jpg\n0.5 0.5 1\n",
        encoding="utf-8",
    )
    (tmp_path / "points3D.txt").write_text(
        "# Number of points: 2, mean track length: 1\n"
        "1 0 0 0 255 0 0 0.1 1 0\n2 1 1 1 0 255 0 0.1 2 0\n",
        encoding="utf-8",
    )

    assert read_text_model(tmp_path) == (1, 2, 2)


def test_creates_sparse_point_preview(tmp_path: Path) -> None:
    points_path = tmp_path / "points3D.txt"
    points_path.write_text(
        "1 0 0 0 255 0 0 0.1 1 0\n2 1 0 0 0 255 0 0.1 1 0\n3 0 1 1 0 0 255 0.1 1 0\n",
        encoding="utf-8",
    )
    preview_path = tmp_path / "preview.png"

    create_sparse_preview(points_path, preview_path)

    with Image.open(preview_path) as preview:
        assert preview.size == (1200, 800)


def test_report_explains_reconstruction_outputs(tmp_path: Path) -> None:
    result = ReconstructionResult(
        input_images=128,
        registered_images=128,
        cameras=1,
        points=108_000,
        mean_reprojection_error=0.612,
        output=tmp_path,
        model=tmp_path / "sparse" / "0",
        preview=tmp_path / "sparse-preview.png",
    )

    report = format_reconstruction_report(result)

    assert "[OK] Registered images: 128/128 (100.0%)" in report
    assert "[OK] Sparse 3D points: 108,000" in report
    assert "[OK] Mean reprojection error: 0.612 px" in report
    assert f"[OK] Preview: {tmp_path / 'sparse-preview.png'}" in report


def test_refuses_nonempty_output_folder(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"one")
    (images / "two.jpg").write_bytes(b"two")
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ReconstructionError, match="output folder is not empty"):
        run_reconstruction(images, output)
