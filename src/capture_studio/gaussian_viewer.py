from __future__ import annotations

import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from capture_studio.colmap_scene import ColmapSceneError, SceneCamera, load_colmap_scene


class GaussianViewerError(RuntimeError):
    """Raised when a Gaussian PLY cannot be shown in the local viewer."""


@dataclass(frozen=True)
class GaussianViewerData:
    centers: np.ndarray
    covariances: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray


@dataclass(frozen=True)
class ViewerCameraPose:
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    field_of_view: float


ProgressReporter = Callable[[str], None]


def _rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float32)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise GaussianViewerError("model contains a zero-length quaternion")
    w, x, y, z = (quaternions / norms).T
    rotations = np.empty((len(quaternions), 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    rotations[:, 0, 1] = 2 * x * y - 2 * w * z
    rotations[:, 0, 2] = 2 * x * z + 2 * w * y
    rotations[:, 1, 0] = 2 * x * y + 2 * w * z
    rotations[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    rotations[:, 1, 2] = 2 * y * z - 2 * w * x
    rotations[:, 2, 0] = 2 * x * z - 2 * w * y
    rotations[:, 2, 1] = 2 * y * z + 2 * w * x
    rotations[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return rotations


def load_gaussian_viewer_data(model_path: Path) -> GaussianViewerData:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise GaussianViewerError(f"Gaussian PLY does not exist: {model_path}")
    try:
        import torch
        from gsplat.exporter import load_ply_to_splats, sh2rgb
    except ImportError as error:
        raise GaussianViewerError(
            "gsplat, PyTorch, and plyfile must be installed"
        ) from error

    try:
        splats = load_ply_to_splats(str(model_path))
    except Exception as error:
        raise GaussianViewerError(f"could not load Gaussian PLY: {error}") from error

    centers = splats["means"].numpy(force=True).astype(np.float32)
    scales = splats["scales"].exp().numpy(force=True).astype(np.float32)
    quaternions = splats["quats"].numpy(force=True).astype(np.float32)
    rotations = _rotation_matrices(quaternions)
    covariances = np.einsum(
        "nij,nj,nkj->nik",
        rotations,
        scales**2,
        rotations,
        optimize=True,
    ).astype(np.float32)
    rgb = sh2rgb(splats["sh0"][:, 0]).clamp(0, 1)
    colors = (rgb * 255).to(torch.uint8).numpy(force=True)
    opacities = splats["opacities"].sigmoid().numpy(force=True)[:, None]

    arrays = (centers, covariances, colors, opacities)
    if not all(np.isfinite(array).all() for array in arrays):
        raise GaussianViewerError("model contains non-finite viewer data")
    return GaussianViewerData(
        centers=centers,
        covariances=covariances,
        colors=colors,
        opacities=opacities.astype(np.float32),
    )


def viewer_pose_from_colmap(camera: SceneCamera) -> ViewerCameraPose:
    rotation = camera.world_to_camera[:3, :3]
    translation = camera.world_to_camera[:3, 3]
    position = -rotation.T @ translation
    forward = rotation.T @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = rotation.T @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
    focal_y = float(camera.intrinsics[1, 1])
    field_of_view = 2 * np.arctan(camera.height / (2 * focal_y))
    return ViewerCameraPose(
        position=position,
        look_at=position + forward,
        up=up,
        field_of_view=float(field_of_view),
    )


def serve_gaussian_model(
    model_path: Path,
    data_path: Path | None = None,
    port: int = 8080,
    open_browser: bool = True,
    progress: ProgressReporter = print,
) -> None:
    if not 1 <= port <= 65535:
        raise GaussianViewerError("port must be between 1 and 65535")
    data = load_gaussian_viewer_data(model_path)
    try:
        import viser
    except ImportError as error:
        raise GaussianViewerError("viser must be installed") from error

    progress(f"Loading {len(data.centers):,} Gaussians into the browser viewer...")
    server = viser.ViserServer(
        host="127.0.0.1",
        port=port,
        label="Capture Studio 3DGS",
        verbose=False,
    )
    server.scene.set_up_direction("+z")
    server.scene.add_gaussian_splats(
        "/trained-gaussians",
        centers=data.centers,
        covariances=data.covariances,
        rgbs=data.colors,
        opacities=data.opacities,
    )
    server.gui.add_markdown(
        "# Capture Studio\n"
        "Drag to rotate, scroll to zoom, and right-drag to move the view."
    )
    server.gui.add_number(
        "Gaussians",
        initial_value=len(data.centers),
        disabled=True,
    )

    if data_path is not None:
        try:
            scene = load_colmap_scene(data_path)
        except ColmapSceneError as error:
            server.stop()
            raise GaussianViewerError(str(error)) from error
        camera_pose = viewer_pose_from_colmap(scene.cameras[0])
        server.scene.set_up_direction(camera_pose.up)
        server.initial_camera.position = camera_pose.position
        server.initial_camera.look_at = camera_pose.look_at
        server.initial_camera.up = camera_pose.up
        server.initial_camera.fov = camera_pose.field_of_view
        progress(f"Initial camera: {scene.cameras[0].name}")
    else:
        lower, upper = np.percentile(data.centers, [5, 95], axis=0)
        center = (lower + upper) / 2
        radius = max(float(np.linalg.norm(upper - lower)), 0.1)
        server.initial_camera.look_at = center
        server.initial_camera.position = center + radius * np.array([0.8, -0.8, 0.5])
        server.initial_camera.up = np.array([0.0, 0.0, 1.0])

    url = f"http://127.0.0.1:{port}"
    progress(f"Viewer: {url}")
    progress("Keep this command running; press Ctrl+C to stop the viewer.")
    if open_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        progress("Stopping viewer...")
    finally:
        server.stop()
