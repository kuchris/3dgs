from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from capture_studio.gaussian_training import (
    GaussianTrainingError,
    GaussianTrainingResult,
    create_render_comparison,
    format_gaussian_training_report,
    train_gaussian_smoke_test,
)


def test_creates_side_by_side_render_comparison(tmp_path: Path) -> None:
    reference = np.zeros((6, 8, 3), dtype=np.float32)
    rendered = np.ones((6, 8, 3), dtype=np.float32)
    output = tmp_path / "comparison.png"

    create_render_comparison(reference, rendered, output)

    with Image.open(output) as comparison:
        assert comparison.size == (16, 40)


def test_report_explains_smoke_training_outputs(tmp_path: Path) -> None:
    result = GaussianTrainingResult(
        steps=100,
        cameras=128,
        gaussians=84_004,
        initial_loss=0.42,
        final_loss=0.31,
        peak_memory_gb=2.5,
        device_name="Test GPU",
        output=tmp_path,
    )

    report = format_gaussian_training_report(result)

    assert "[OK] Cameras loaded: 128" in report
    assert "[OK] Gaussians: 84,004" in report
    assert "[OK] Preview L1 loss: 0.4200 -> 0.3100" in report
    assert f"[OK] Checkpoint: {tmp_path / 'checkpoint.pt'}" in report


def test_rejects_invalid_training_settings(tmp_path: Path) -> None:
    with pytest.raises(GaussianTrainingError, match="steps must be"):
        train_gaussian_smoke_test(tmp_path, tmp_path / "output", steps=0)
    with pytest.raises(GaussianTrainingError, match="image scale must be"):
        train_gaussian_smoke_test(tmp_path, tmp_path / "output", image_scale=0)
