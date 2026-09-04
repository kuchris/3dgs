from importlib import metadata

from capture_studio.system_check import build_report, format_report


def test_report_distinguishes_available_and_missing_dependencies() -> None:
    outputs = {
        "uv": "uv 0.12.9",
        "nvidia-smi": "NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 610.47",
        "nvcc": None,
        "colmap": None,
    }

    def fake_runner(command: str, arguments: object) -> str | None:
        return outputs[command]

    def fake_package_version(package: str) -> str:
        if package == "torch":
            raise metadata.PackageNotFoundError(package)
        raise AssertionError(f"Unexpected package: {package}")

    report = format_report(
        build_report(fake_runner, fake_package_version, colmap_finder=lambda: None)
    )

    assert "[OK     ] NVIDIA GPU: NVIDIA GeForce RTX 5070 Ti" in report
    assert "[MISSING] PyTorch: not installed" in report
    assert "[MISSING] CUDA compiler: nvcc not found in PATH" in report
    assert "[MISSING] COLMAP: not installed" in report
