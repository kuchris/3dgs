import argparse
from pathlib import Path

from capture_studio.colmap_check import (
    ColmapCheckError,
    format_colmap_report,
    run_colmap_check,
)
from capture_studio.gaussian_training import (
    GaussianTrainingError,
    format_gaussian_training_report,
    train_gaussian_quality,
    train_gaussian_smoke_test,
)
from capture_studio.gpu_check import GPUCheckError, format_gpu_report, run_gpu_check
from capture_studio.gsplat_check import (
    GSplatCheckError,
    format_gsplat_report,
    run_gsplat_check,
)
from capture_studio.model_export import (
    ModelExportError,
    export_gaussian_ply,
    format_model_export_report,
)
from capture_studio.photo_analysis import (
    PhotoAnalysisError,
    analyze_folder,
    format_photo_report,
)
from capture_studio.reconstruction import (
    ReconstructionError,
    format_reconstruction_report,
    run_reconstruction,
)
from capture_studio.system_check import build_report, format_report
from capture_studio.training_data import (
    TrainingDataError,
    format_training_data_report,
    prepare_training_data,
)


def _analyze(args: argparse.Namespace) -> None:
    try:
        result = analyze_folder(args.folder)
    except PhotoAnalysisError as error:
        print(f"Photo analysis failed: {error}")
        raise SystemExit(1) from error
    print(format_photo_report(result))


def _check_system(_args: argparse.Namespace) -> None:
    print(format_report(build_report()))


def _check_colmap(_args: argparse.Namespace) -> None:
    try:
        result = run_colmap_check()
    except ColmapCheckError as error:
        print(f"COLMAP smoke test failed: {error}")
        raise SystemExit(1) from error
    print(format_colmap_report(result))


def _check_gpu(_args: argparse.Namespace) -> None:
    try:
        result = run_gpu_check()
    except GPUCheckError as error:
        print(f"GPU smoke test failed: {error}")
        raise SystemExit(1) from error
    print(format_gpu_report(result))


def _check_gsplat(_args: argparse.Namespace) -> None:
    try:
        result = run_gsplat_check()
    except GSplatCheckError as error:
        print(f"gsplat smoke test failed: {error}")
        raise SystemExit(1) from error
    print(format_gsplat_report(result))


def _reconstruct(args: argparse.Namespace) -> None:
    try:
        result = run_reconstruction(args.folder, args.output)
    except ReconstructionError as error:
        print(f"Reconstruction failed: {error}")
        raise SystemExit(1) from error
    print(format_reconstruction_report(result))


def _prepare_training(args: argparse.Namespace) -> None:
    try:
        result = prepare_training_data(
            args.folder,
            args.model,
            args.output,
            max_image_size=args.max_image_size,
        )
    except TrainingDataError as error:
        print(f"Training-data preparation failed: {error}")
        raise SystemExit(1) from error
    print(format_training_data_report(result))


def _train_smoke(args: argparse.Namespace) -> None:
    try:
        result = train_gaussian_smoke_test(
            args.data,
            args.output,
            steps=args.steps,
            image_scale=args.image_scale,
        )
    except GaussianTrainingError as error:
        print(f"Gaussian training failed: {error}")
        raise SystemExit(1) from error
    print(format_gaussian_training_report(result))


def _train_quality(args: argparse.Namespace) -> None:
    try:
        result = train_gaussian_quality(
            args.data,
            args.output,
            steps=args.steps,
            image_scale=args.image_scale,
        )
    except GaussianTrainingError as error:
        print(f"Gaussian training failed: {error}")
        raise SystemExit(1) from error
    print(format_gaussian_training_report(result))


def _export_ply(args: argparse.Namespace) -> None:
    try:
        result = export_gaussian_ply(args.checkpoint, args.output)
    except ModelExportError as error:
        print(f"Gaussian export failed: {error}")
        raise SystemExit(1) from error
    print(format_model_export_report(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="capture-studio",
        description="Build and inspect local 3D Gaussian Splatting captures.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subcommands.add_parser(
        "analyze", help="Check the basic quality of photos in a folder."
    )
    analyze_parser.add_argument("folder", type=Path, help="Folder containing photos.")
    analyze_parser.set_defaults(handler=_analyze)

    reconstruct_parser = subcommands.add_parser(
        "reconstruct", help="Build a sparse COLMAP model from a photo folder."
    )
    reconstruct_parser.add_argument(
        "folder", type=Path, help="Folder containing photos."
    )
    reconstruct_parser.add_argument(
        "--output", type=Path, required=True, help="Empty output folder for the model."
    )
    reconstruct_parser.set_defaults(handler=_reconstruct)

    prepare_parser = subcommands.add_parser(
        "prepare-training",
        help="Undistort photos and cameras for Gaussian Splatting training.",
    )
    prepare_parser.add_argument("folder", type=Path, help="Folder containing photos.")
    prepare_parser.add_argument(
        "--model", type=Path, required=True, help="Sparse COLMAP model folder."
    )
    prepare_parser.add_argument(
        "--output", type=Path, required=True, help="Empty training-data folder."
    )
    prepare_parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Maximum width or height in pixels (default: 1600).",
    )
    prepare_parser.set_defaults(handler=_prepare_training)

    train_parser = subcommands.add_parser(
        "train-smoke", help="Run a short Gaussian Splatting training test."
    )
    train_parser.add_argument(
        "data", type=Path, help="Prepared training-data folder."
    )
    train_parser.add_argument(
        "--output", type=Path, required=True, help="Empty training output folder."
    )
    train_parser.add_argument(
        "--steps", type=int, default=100, help="Optimization steps (default: 100)."
    )
    train_parser.add_argument(
        "--image-scale",
        type=int,
        default=4,
        help="Training image downscale factor (default: 4).",
    )
    train_parser.set_defaults(handler=_train_smoke)

    quality_parser = subcommands.add_parser(
        "train-quality", help="Train Gaussians with splitting and pruning."
    )
    quality_parser.add_argument(
        "data", type=Path, help="Prepared training-data folder."
    )
    quality_parser.add_argument(
        "--output", type=Path, required=True, help="Empty training output folder."
    )
    quality_parser.add_argument(
        "--steps", type=int, default=1000, help="Optimization steps (default: 1000)."
    )
    quality_parser.add_argument(
        "--image-scale",
        type=int,
        default=2,
        help="Training image downscale factor (default: 2).",
    )
    quality_parser.set_defaults(handler=_train_quality)

    export_parser = subcommands.add_parser(
        "export-ply", help="Export a trained checkpoint as a standard 3DGS PLY."
    )
    export_parser.add_argument("checkpoint", type=Path, help="Training checkpoint.")
    export_parser.add_argument(
        "--output", type=Path, required=True, help="New .ply output file."
    )
    export_parser.set_defaults(handler=_export_ply)

    check_parser = subcommands.add_parser(
        "check-system", help="Report local Python, GPU, and native-tool support."
    )
    check_parser.set_defaults(handler=_check_system)

    colmap_parser = subcommands.add_parser(
        "check-colmap", help="Extract test-image features with COLMAP on CUDA."
    )
    colmap_parser.set_defaults(handler=_check_colmap)

    gpu_parser = subcommands.add_parser(
        "check-gpu", help="Run and verify a small PyTorch calculation on CUDA."
    )
    gpu_parser.set_defaults(handler=_check_gpu)

    gsplat_parser = subcommands.add_parser(
        "check-gsplat", help="Render and differentiate a small Gaussian on CUDA."
    )
    gsplat_parser.set_defaults(handler=_check_gsplat)

    args = parser.parse_args()
    args.handler(args)
