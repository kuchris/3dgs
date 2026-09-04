from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from capture_studio.colmap_scene import (
    ColmapSceneError,
    load_colmap_scene,
    quaternion_to_rotation,
)


def _write_scene(root: Path) -> None:
    images = root / "images"
    model = root / "sparse-text"
    images.mkdir(parents=True)
    model.mkdir()
    Image.new("RGB", (8, 6), "red").save(images / "first.jpg")
    Image.new("RGB", (8, 6), "blue").save(images / "second.jpg")
    (model / "cameras.txt").write_text(
        "# Number of cameras: 1\n1 PINHOLE 8 6 5 5 4 3\n",
        encoding="utf-8",
    )
    (model / "images.txt").write_text(
        "# Number of images: 2\n"
        "1 1 0 0 0 0 0 0 1 first.jpg\n\n"
        "2 1 0 0 0 -2 0 0 1 second.jpg\n\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text(
        "# Number of points: 2\n"
        "1 0 0 2 255 0 0 0.1 1 0\n"
        "2 2 0 2 0 0 255 0.1 2 0\n",
        encoding="utf-8",
    )


def test_identity_quaternion_produces_identity_rotation() -> None:
    rotation = quaternion_to_rotation(np.array([1.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(rotation, np.eye(3))


def test_loads_and_normalizes_prepared_colmap_scene(tmp_path: Path) -> None:
    _write_scene(tmp_path)

    scene = load_colmap_scene(tmp_path)

    assert len(scene.cameras) == 2
    assert scene.points.shape == (2, 3)
    assert scene.colors.shape == (2, 3)
    np.testing.assert_allclose(scene.world_center, [1.0, 0.0, 0.0])
    assert scene.world_scale == pytest.approx(1.0)
    np.testing.assert_allclose(scene.points[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(scene.cameras[0].world_to_camera[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(scene.cameras[1].world_to_camera[:3, 3], [-1, 0, 0])


def test_rejects_unprepared_camera_model(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    (tmp_path / "sparse-text" / "cameras.txt").write_text(
        "1 SIMPLE_RADIAL 8 6 5 4 3 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ColmapSceneError, match="only prepared PINHOLE"):
        load_colmap_scene(tmp_path)
