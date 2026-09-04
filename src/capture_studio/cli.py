import argparse
from pathlib import Path

from capture_studio.colmap_check import (
    ColmapCheckError,
    format_colmap_report,
    run_colmap_check,
)
from capture_studio.gpu_check import GPUCheckError, format_gpu_report, run_gpu_check
from capture_studio.gsplat_check import (
    GSplatCheckError,
    format_gsplat_report,
    run_gsplat_check,
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
