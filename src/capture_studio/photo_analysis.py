from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MIN_MEGAPIXELS = 2.0
POSSIBLE_BLUR_THRESHOLD = 100.0
BLUR_ANALYSIS_MAX_SIZE = 1024


class PhotoAnalysisError(RuntimeError):
    """Raised when a photo folder cannot be analyzed."""


@dataclass(frozen=True)
class PhotoResult:
    path: Path
    width: int
    height: int
    blur_score: float
    warnings: tuple[str, ...]

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


@dataclass(frozen=True)
class UnreadablePhoto:
    path: Path
    reason: str


@dataclass(frozen=True)
class PhotoAnalysisResult:
    folder: Path
    photos: tuple[PhotoResult, ...]
    unreadable: tuple[UnreadablePhoto, ...]
    ignored: tuple[Path, ...]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as photo_file:
        for block in iter(lambda: photo_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blur_score(image: Image.Image) -> float:
    grayscale = ImageOps.grayscale(image)
    if max(grayscale.size) > BLUR_ANALYSIS_MAX_SIZE:
        grayscale.thumbnail(
            (BLUR_ANALYSIS_MAX_SIZE, BLUR_ANALYSIS_MAX_SIZE),
            Image.Resampling.LANCZOS,
        )

    pixels = np.asarray(grayscale, dtype=np.float32)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return 0.0

    center = pixels[1:-1, 1:-1]
    laplacian = (
        pixels[:-2, 1:-1]
        + pixels[2:, 1:-1]
        + pixels[1:-1, :-2]
        + pixels[1:-1, 2:]
        - 4 * center
    )
    return float(laplacian.var())


def analyze_folder(folder: Path) -> PhotoAnalysisResult:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise PhotoAnalysisError(f"photo folder does not exist: {folder}")

    files = sorted(
        (path for path in folder.iterdir() if path.is_file()),
        key=lambda path: path.name.casefold(),
    )
    photo_paths = [path for path in files if path.suffix.lower() in SUPPORTED_EXTENSIONS]
    ignored = tuple(path for path in files if path.suffix.lower() not in SUPPORTED_EXTENSIONS)
    if not photo_paths:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise PhotoAnalysisError(f"no supported images found; expected: {extensions}")

    photos: list[PhotoResult] = []
    unreadable: list[UnreadablePhoto] = []
    first_path_by_digest: dict[str, Path] = {}

    for path in photo_paths:
        try:
            with Image.open(path) as opened_image:
                image = ImageOps.exif_transpose(opened_image)
                image.load()
                width, height = image.size
                blur_score = _blur_score(image)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            unreadable.append(UnreadablePhoto(path=path, reason=str(error)))
            continue

        warnings: list[str] = []
        megapixels = width * height / 1_000_000
        if megapixels < MIN_MEGAPIXELS:
            warnings.append(
                f"low resolution ({megapixels:.2f} MP; target at least {MIN_MEGAPIXELS:.1f} MP)"
            )
        if blur_score < POSSIBLE_BLUR_THRESHOLD:
            warnings.append(
                f"possible blur (score {blur_score:.1f}; target at least {POSSIBLE_BLUR_THRESHOLD:.0f})"
            )

        digest = _file_digest(path)
        duplicate_of = first_path_by_digest.get(digest)
        if duplicate_of is None:
            first_path_by_digest[digest] = path
        else:
            warnings.append(f"exact duplicate of {duplicate_of.name}")

        photos.append(
            PhotoResult(
                path=path,
                width=width,
                height=height,
                blur_score=blur_score,
                warnings=tuple(warnings),
            )
        )

    return PhotoAnalysisResult(
        folder=folder,
        photos=tuple(photos),
        unreadable=tuple(unreadable),
        ignored=ignored,
    )


def format_photo_report(result: PhotoAnalysisResult) -> str:
    lines = [
        "Capture Studio photo analysis",
        f"Folder: {result.folder}",
        f"Readable photos: {len(result.photos)}",
    ]

    for photo in result.photos:
        status = "WARN" if photo.warnings else "OK"
        lines.append(
            f"[{status}] {photo.path.name}: {photo.width}x{photo.height}, "
            f"{photo.megapixels:.2f} MP, blur score {photo.blur_score:.1f}"
        )
        for warning in photo.warnings:
            lines.append(f"       - {warning}")

    for photo in result.unreadable:
        lines.append(f"[ERROR] {photo.path.name}: unreadable ({photo.reason})")

    warning_count = sum(bool(photo.warnings) for photo in result.photos)
    lines.extend(
        [
            "",
            "Summary",
            f"Photos needing attention: {warning_count}",
            f"Unreadable photos: {len(result.unreadable)}",
            f"Ignored non-image files: {len(result.ignored)}",
            "Note: blur is a screening hint, not a final judgment of photo quality.",
        ]
    )
    return "\n".join(lines)
