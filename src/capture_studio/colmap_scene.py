from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class ColmapSceneError(RuntimeError):
    """Raised when prepared COLMAP training data is invalid."""


@dataclass(frozen=True)
class SceneCamera:
    name: str
    image_path: Path
    width: int
    height: int
    intrinsics: np.ndarray
    world_to_camera: np.ndarray


@dataclass(frozen=True)
class ColmapScene:
    cameras: tuple[SceneCamera, ...]
    points: np.ndarray
    colors: np.ndarray
    world_center: np.ndarray
    world_scale: float


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ColmapSceneError("camera quaternion cannot be zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _data_lines(path: Path, keep_empty: bool = False) -> list[str]:
    if not path.is_file():
        raise ColmapSceneError(f"COLMAP text model file does not exist: {path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    return lines if keep_empty else [line for line in lines if line]


def _read_cameras(path: Path) -> dict[int, tuple[int, int, np.ndarray]]:
    cameras: dict[int, tuple[int, int, np.ndarray]] = {}
    for line in _data_lines(path):
        tokens = line.split()
        if len(tokens) != 8 or tokens[1] != "PINHOLE":
            model = tokens[1] if len(tokens) > 1 else "unknown"
            raise ColmapSceneError(
                f"only prepared PINHOLE cameras are supported, got {model}"
            )
        camera_id = int(tokens[0])
        width, height = int(tokens[2]), int(tokens[3])
        fx, fy, cx, cy = (float(value) for value in tokens[4:8])
        intrinsics = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        cameras[camera_id] = (width, height, intrinsics)
    if not cameras:
        raise ColmapSceneError("COLMAP model contains no cameras")
    return cameras


def _read_image_records(path: Path) -> list[tuple[str, int, np.ndarray]]:
    lines = _data_lines(path, keep_empty=True)
    if len(lines) % 2 != 0:
        raise ColmapSceneError("COLMAP images.txt has an incomplete image record")

    records: list[tuple[str, int, np.ndarray]] = []
    for index in range(0, len(lines), 2):
        tokens = lines[index].split(maxsplit=9)
        if len(tokens) != 10:
            raise ColmapSceneError("COLMAP images.txt has an invalid camera line")
        quaternion = np.array([float(value) for value in tokens[1:5]])
        translation = np.array([float(value) for value in tokens[5:8]])
        camera_id = int(tokens[8])
        rotation = quaternion_to_rotation(quaternion)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = translation
        records.append((tokens[9], camera_id, world_to_camera))
    if not records:
        raise ColmapSceneError("COLMAP model contains no registered images")
    return sorted(records, key=lambda record: record[0])


def _read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    colors: list[list[float]] = []
    for line in _data_lines(path):
        tokens = line.split()
        if len(tokens) < 8:
            raise ColmapSceneError("COLMAP points3D.txt has an invalid point line")
        positions.append([float(value) for value in tokens[1:4]])
        colors.append([int(value) / 255.0 for value in tokens[4:7]])
    if not positions:
        raise ColmapSceneError("COLMAP model contains no sparse points")
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
    )


def load_colmap_scene(data_path: Path) -> ColmapScene:
    data_path = data_path.expanduser().resolve()
    images_path = data_path / "images"
    model_path = data_path / "sparse-text"
    if not images_path.is_dir():
        raise ColmapSceneError(f"training image folder does not exist: {images_path}")
    if not model_path.is_dir():
        raise ColmapSceneError(f"COLMAP text model folder does not exist: {model_path}")

    camera_models = _read_cameras(model_path / "cameras.txt")
    image_records = _read_image_records(model_path / "images.txt")
    points, colors = _read_points(model_path / "points3D.txt")

    camera_centers = []
    for _, _, world_to_camera in image_records:
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        camera_centers.append(-rotation.T @ translation)
    world_center = np.mean(camera_centers, axis=0)
    world_scale = float(
        np.max(np.linalg.norm(np.asarray(camera_centers) - world_center, axis=1))
    )
    if not np.isfinite(world_scale) or world_scale <= 0:
        raise ColmapSceneError("camera positions do not define a valid scene scale")

    normalized_points = (points - world_center) / world_scale
    scene_cameras: list[SceneCamera] = []
    for name, camera_id, world_to_camera in image_records:
        if camera_id not in camera_models:
            raise ColmapSceneError(f"image {name} uses unknown camera {camera_id}")
        width, height, intrinsics = camera_models[camera_id]
        image_path = images_path / name
        if not image_path.is_file():
            raise ColmapSceneError(f"registered image does not exist: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ColmapSceneError(
                    f"image size does not match camera for {name}: "
                    f"{image.size} != {(width, height)}"
                )

        normalized_world_to_camera = world_to_camera.copy()
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        normalized_world_to_camera[:3, 3] = (
            rotation @ world_center + translation
        ) / world_scale
        scene_cameras.append(
            SceneCamera(
                name=name,
                image_path=image_path,
                width=width,
                height=height,
                intrinsics=intrinsics.astype(np.float32),
                world_to_camera=normalized_world_to_camera.astype(np.float32),
            )
        )

    return ColmapScene(
        cameras=tuple(scene_cameras),
        points=normalized_points.astype(np.float32),
        colors=colors,
        world_center=world_center.astype(np.float64),
        world_scale=world_scale,
    )
