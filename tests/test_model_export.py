from pathlib import Path

import pytest
import torch
from gsplat.exporter import load_ply_to_splats

from capture_studio.model_export import ModelExportError, export_gaussian_ply


def _write_checkpoint(path: Path) -> None:
    torch.save(
        {
            "means": torch.tensor(
                [[0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [0.0, 0.5, 2.0]]
            ),
            "scales": torch.full((3, 3), -2.0),
            "quaternions": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
            "opacities": torch.zeros(3),
            "color_logits": torch.zeros((3, 3)),
        },
        path,
    )


def test_exports_checkpoint_as_reloadable_ply(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "model.ply"
    _write_checkpoint(checkpoint)

    result = export_gaussian_ply(checkpoint, output)
    loaded = load_ply_to_splats(str(output))

    assert result.gaussians == 3
    assert result.size_bytes > 0
    assert loaded["means"].shape == (3, 3)
    assert loaded["sh0"].shape == (3, 1, 3)


def test_refuses_to_overwrite_existing_ply(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "model.ply"
    _write_checkpoint(checkpoint)
    output.write_bytes(b"keep")

    with pytest.raises(ModelExportError, match="already exists"):
        export_gaussian_ply(checkpoint, output)
