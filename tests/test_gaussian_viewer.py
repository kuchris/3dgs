from pathlib import Path

import numpy as np
import pytest
import torch

from capture_studio.colmap_scene import SceneCamera
from capture_studio.gaussian_viewer import (
    _rotation_matrices,
    load_gaussian_viewer_data,
    viewer_pose_from_colmap,
)
from capture_studio.model_export import export_gaussian_ply


def test_identity_quaternion_produces_identity_rotation() -> None:
    rotations = _rotation_matrices(np.array([[1.0, 0.0, 0.0, 0.0]]))

    np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-6)


def test_loads_exported_gaussians_for_browser_viewer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    model = tmp_path / "model.ply"
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
        checkpoint,
    )
    export_gaussian_ply(checkpoint, model)

    data = load_gaussian_viewer_data(model)

    assert data.centers.shape == (3, 3)
    assert data.covariances.shape == (3, 3, 3)
    assert data.colors.shape == (3, 3)
    assert data.opacities.shape == (3, 1)
    np.testing.assert_allclose(data.opacities, 0.5, atol=1e-6)
    assert np.linalg.eigvalsh(data.covariances).min() > 0


def test_builds_browser_pose_from_colmap_camera(tmp_path: Path) -> None:
    camera = SceneCamera(
        name="photo.jpg",
        image_path=tmp_path / "photo.jpg",
        width=8,
        height=6,
        intrinsics=np.array([[5, 0, 4], [0, 5, 3], [0, 0, 1]], dtype=np.float32),
        world_to_camera=np.eye(4, dtype=np.float32),
    )

    pose = viewer_pose_from_colmap(camera)

    np.testing.assert_allclose(pose.position, [0, 0, 0])
    np.testing.assert_allclose(pose.look_at, [0, 0, 1])
    np.testing.assert_allclose(pose.up, [0, -1, 0])
    assert pose.field_of_view == pytest.approx(2 * np.arctan(0.6))
