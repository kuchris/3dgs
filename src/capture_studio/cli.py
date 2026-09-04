import argparse

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
from capture_studio.system_check import build_report, format_report


def _check_system() -> None:
    print(format_report(build_report()))


def _check_colmap() -> None:
    try:
        result = run_colmap_check()
    except ColmapCheckError as error:
        print(f"COLMAP smoke test failed: {error}")
        raise SystemExit(1) from error
    print(format_colmap_report(result))


def _check_gpu() -> None:
    try:
        result = run_gpu_check()
    except GPUCheckError as error:
        print(f"GPU smoke test failed: {error}")
        raise SystemExit(1) from error
    print(format_gpu_report(result))


def _check_gsplat() -> None:
    try:
        result = run_gsplat_check()
    except GSplatCheckError as error:
        print(f"gsplat smoke test failed: {error}")
        raise SystemExit(1) from error
    print(format_gsplat_report(result))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="capture-studio",
        description="Build and inspect local 3D Gaussian Splatting captures.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

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
    args.handler()
