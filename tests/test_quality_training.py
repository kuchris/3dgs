import numpy as np
import pytest
import torch

from capture_studio.quality_training import (
    camera_for_step,
    camera_split,
    image_metrics,
    quality_schedule,
)


def test_each_epoch_uses_every_training_camera_once_and_excludes_validation():
    training, validation = camera_split(128)
    assert len(training) == 112
    assert len(validation) == 16
    for epoch in range(3):
        sampled = [
            camera_for_step(training, step)
            for step in range(epoch * 112, (epoch + 1) * 112)
        ]
        assert sorted(sampled) == training
        assert set(sampled).isdisjoint(validation)


def test_schedule_leaves_time_to_refine_after_last_densification():
    schedule = quality_schedule(30000)
    assert schedule["refine_start_iter"] == 500
    assert schedule["refine_stop_iter"] == 15000
    assert schedule["sh_interval"] == 1000
    short = quality_schedule(201)
    assert (
        short["refine_start_iter"] + short["refine_every"]
        < short["refine_stop_iter"]
        < 201
    )


def test_image_metrics_reward_matching_images_and_psnr_has_known_value():
    target = torch.full((1, 16, 16, 3), 0.5)
    same = image_metrics(target, target)
    changed = image_metrics(target + 0.1, target)
    assert same["l1"] == 0
    assert same["ssim"] == pytest.approx(1)
    assert np.isfinite(same["psnr"])
    assert changed["psnr"] == pytest.approx(20, abs=1e-4)
    assert changed["ssim"] < same["ssim"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA gsplat")
def test_resume_restores_optimizer_and_rejects_changed_input(tmp_path, monkeypatch):
    from PIL import Image

    from capture_studio.colmap_scene import ColmapScene, SceneCamera
    from capture_studio.quality_training import GaussianTrainingError, train_quality_v2

    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    image = data / "images" / "photo.png"
    Image.new("RGB", (16, 16), (100, 140, 180)).save(image)
    intrinsics = np.array([[16, 0, 8], [0, 16, 8], [0, 0, 1]], dtype=np.float32)
    cameras = tuple(
        SceneCamera(str(i), image, 16, 16, intrinsics, np.eye(4, dtype=np.float32))
        for i in range(9)
    )
    scene = ColmapScene(
        cameras,
        np.array(
            [[-0.2, -0.2, 2], [0.2, -0.2, 2], [-0.2, 0.2, 2], [0.2, 0.2, 2]],
            dtype=np.float32,
        ),
        np.full((4, 3), 0.5, dtype=np.float32),
        np.zeros(3),
        1.0,
    )
    monkeypatch.setattr(
        "capture_studio.quality_training.load_colmap_scene", lambda _: scene
    )
    settings = {"steps": 201, "progress": lambda _: None}
    train_quality_v2(data, tmp_path / "full", stop_after=6, **settings)
    train_quality_v2(data, tmp_path / "paused", stop_after=5, **settings)
    saved = tmp_path / "paused" / "checkpoint.pt"
    train_quality_v2(data, tmp_path / "resumed", resume=saved, stop_after=6, **settings)
    full = torch.load(
        tmp_path / "full" / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    resumed = torch.load(
        tmp_path / "resumed" / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    for key in full["parameters"]:
        if key != "quats":
            torch.testing.assert_close(
                full["parameters"][key],
                resumed["parameters"][key],
                atol=1e-5,
                rtol=1e-4,
            )
        for index, state in full["optimizers"][key]["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(
                    value,
                    resumed["optimizers"][key]["state"][index][name],
                    # CUDA gradient accumulation is not bitwise deterministic.
                    atol=1e-5,
                    rtol=1e-4,
                )
    assert full["scheduler"] == resumed["scheduler"]
    # Rotation is ambiguous for nearly isotropic splats; compare actual covariances.
    from capture_studio.gaussian_viewer import _rotation_matrices

    covariances = []
    for saved_state in (full, resumed):
        parameters = saved_state["parameters"]
        rotations = _rotation_matrices(parameters["quats"].numpy())
        covariances.append(
            np.einsum(
                "nij,nj,nkj->nik",
                rotations,
                parameters["scales"].exp().numpy() ** 2,
                rotations,
            )
        )
    np.testing.assert_allclose(*covariances, atol=1e-5, rtol=1e-4)
    with pytest.raises(GaussianTrainingError, match="matching data and settings"):
        train_quality_v2(data, tmp_path / "bad-settings", resume=saved, steps=202)
    Image.new("RGB", (16, 16), "red").save(image)
    with pytest.raises(GaussianTrainingError, match="matching data and settings"):
        train_quality_v2(data, tmp_path / "bad-input", resume=saved, **settings)
