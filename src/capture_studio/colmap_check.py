from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile

import numpy as np


COLMAP_VERSION = "4.2.0"


class ColmapCheckError(RuntimeError):
    """Raised when COLMAP cannot complete the GPU feature extraction test."""


@dataclass(frozen=True)
class ColmapCheckResult:
    version: str
    executable: Path
    feature_count: int


def find_colmap() -> Path | None:
    explicit_path = os.environ.get("COLMAP_PATH")
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.is_dir():
            candidate /= "COLMAP.bat"
        if candidate.is_file():
            return candidate

    command_path = shutil.which("colmap") or shutil.which("COLMAP.bat")
    if command_path:
        return Path(command_path)

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidate = (
            Path(local_app_data)
            / "Programs"
            / f"COLMAP-{COLMAP_VERSION}"
            / "COLMAP.bat"
        )
        if candidate.is_file():
            return candidate

    return None


def _create_test_image(path: Path) -> None:
    y, x = np.indices((512, 512))
    checker = ((x // 16 + y // 16) % 2) * 170
    waves = 40 * np.sin(x / 7.0) + 40 * np.cos(y / 11.0)
    image = np.clip(40 + checker + waves, 0, 255).astype(np.uint8)
    path.write_bytes(b"P5\n512 512\n255\n" + image.tobytes())


def _run_colmap(executable: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *arguments],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )


def run_colmap_check() -> ColmapCheckResult:
    executable = find_colmap()
    if executable is None:
        raise ColmapCheckError(
            "COLMAP was not found; run scripts/install-colmap-windows.ps1"
        )

    version_result = _run_colmap(executable, ["version"])
    version = (version_result.stdout or version_result.stderr).strip()
    if version_result.returncode != 0:
        raise ColmapCheckError(f"COLMAP version check failed: {version}")
    if "with CUDA" not in version:
        raise ColmapCheckError(f"COLMAP does not report CUDA support: {version}")

    with tempfile.TemporaryDirectory(prefix="capture-studio-colmap-") as temporary:
        test_root = Path(temporary)
        image_directory = test_root / "images"
        image_directory.mkdir()
        _create_test_image(image_directory / "texture.pgm")
        database_path = test_root / "database.db"

        extraction_result = _run_colmap(
            executable,
            [
                "feature_extractor",
                "--database_path",
                str(database_path),
                "--image_path",
                str(image_directory),
                "--FeatureExtraction.use_gpu",
                "1",
                "--FeatureExtraction.gpu_index",
                "0",
                "--log_target",
                "stdout",
            ],
        )
        output = extraction_result.stdout + extraction_result.stderr
        if extraction_result.returncode != 0:
            raise ColmapCheckError(f"GPU feature extraction failed:\n{output.strip()}")
        if "Creating SIFT GPU feature extractor" not in output:
            raise ColmapCheckError("COLMAP did not confirm use of its GPU extractor")

        with closing(sqlite3.connect(database_path)) as database:
            row = database.execute(
                "SELECT COALESCE(SUM(rows), 0) FROM keypoints"
            ).fetchone()
        feature_count = int(row[0]) if row else 0
        if feature_count == 0:
            raise ColmapCheckError("COLMAP did not extract any image features")

    return ColmapCheckResult(
        version=version,
        executable=executable,
        feature_count=feature_count,
    )


def format_colmap_report(result: ColmapCheckResult) -> str:
    return "\n".join(
        [
            "COLMAP GPU smoke test",
            f"[OK] Version: {result.version}",
            f"[OK] Executable: {result.executable}",
            "[OK] SIFT feature extraction: verified on GPU 0",
            f"[OK] Features extracted: {result.feature_count}",
        ]
    )
