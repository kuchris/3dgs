from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import pytest

from capture_studio.photo_analysis import (
    PhotoAnalysisError,
    analyze_folder,
    format_photo_report,
)


def _checkerboard() -> Image.Image:
    y, x = np.indices((256, 256))
    pixels = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def test_analysis_flags_blur_and_exact_duplicates(tmp_path: Path) -> None:
    sharp_path = tmp_path / "a-sharp.png"
    duplicate_path = tmp_path / "b-sharp-copy.png"
    blurred_path = tmp_path / "blurred.png"

    sharp = _checkerboard()
    sharp.save(sharp_path)
    duplicate_path.write_bytes(sharp_path.read_bytes())
    sharp.filter(ImageFilter.GaussianBlur(radius=8)).save(blurred_path)

    result = analyze_folder(tmp_path)
    photos = {photo.path.name: photo for photo in result.photos}

    assert photos["a-sharp.png"].blur_score > photos["blurred.png"].blur_score
    assert not any("possible blur" in warning for warning in photos["a-sharp.png"].warnings)
    assert any("possible blur" in warning for warning in photos["blurred.png"].warnings)
    assert "exact duplicate of a-sharp.png" in photos["b-sharp-copy.png"].warnings
    assert "Photos needing attention: 3" in format_photo_report(result)


def test_analysis_rejects_folder_without_supported_images(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    with pytest.raises(PhotoAnalysisError, match="no supported images found"):
        analyze_folder(tmp_path)


def test_analysis_reports_corrupted_image(tmp_path: Path) -> None:
    (tmp_path / "broken.jpg").write_bytes(b"not really a JPEG")

    result = analyze_folder(tmp_path)

    assert not result.photos
    assert result.unreadable[0].path.name == "broken.jpg"
