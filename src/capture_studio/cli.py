import argparse

from capture_studio.system_check import build_report, format_report


def _check_system() -> None:
    print(format_report(build_report()))


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

    args = parser.parse_args()
    args.handler()

