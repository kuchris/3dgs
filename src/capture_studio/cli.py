import argparse

from capture_studio.gpu_check import GPUCheckError, format_gpu_report, run_gpu_check
from capture_studio.system_check import build_report, format_report


def _check_system() -> None:
    print(format_report(build_report()))


def _check_gpu() -> None:
    try:
        result = run_gpu_check()
    except GPUCheckError as error:
        print(f"GPU smoke test failed: {error}")
        raise SystemExit(1) from error
    print(format_gpu_report(result))


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

    gpu_parser = subcommands.add_parser(
        "check-gpu", help="Run and verify a small PyTorch calculation on CUDA."
    )
    gpu_parser.set_defaults(handler=_check_gpu)

    args = parser.parse_args()
    args.handler()

