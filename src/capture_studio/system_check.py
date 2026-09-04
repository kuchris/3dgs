from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence

from capture_studio.colmap_check import find_colmap


@dataclass(frozen=True)
class Check:
    name: str
    available: bool
    detail: str


CommandRunner = Callable[[str, Sequence[str]], str | None]
PackageVersion = Callable[[str], str]
ColmapFinder = Callable[[], Path | None]


def _command_output(command: str, arguments: Sequence[str]) -> str | None:
    executable = command if Path(command).is_file() else shutil.which(command)
    if executable is None:
        return None

    completed = subprocess.run(
        [executable, *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    output = completed.stdout.strip() or completed.stderr.strip()
    return output if output else None


def _first_line(output: str | None, missing: str) -> tuple[bool, str]:
    if output is None:
        return False, missing
    return True, output.splitlines()[0].strip()


def build_report(
    runner: CommandRunner = _command_output,
    package_version: PackageVersion = metadata.version,
    colmap_finder: ColmapFinder = find_colmap,
) -> list[Check]:
    colmap_path = colmap_finder()
    checks = [
        Check("Python", True, sys.version.split()[0]),
        Check("uv", *_first_line(runner("uv", ["--version"]), "not found in PATH")),
        Check(
            "NVIDIA GPU",
            *_first_line(
                runner(
                    "nvidia-smi",
                    [
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader",
                    ],
                ),
                "nvidia-smi not found",
            ),
        ),
        Check(
            "CUDA compiler",
            *_first_line(runner("nvcc", ["--version"]), "nvcc not found in PATH"),
        ),
        Check(
            "COLMAP",
            *_first_line(
                runner(str(colmap_path) if colmap_path else "colmap", ["-h"]),
                "not installed",
            ),
        ),
    ]

    try:
        torch_version = package_version("torch")
    except metadata.PackageNotFoundError:
        checks.insert(3, Check("PyTorch", False, "not installed"))
    else:
        checks.insert(3, Check("PyTorch", True, torch_version))

    return checks


def format_report(checks: Sequence[Check]) -> str:
    lines = ["Capture Studio system check"]
    for check in checks:
        status = "OK" if check.available else "MISSING"
        lines.append(f"[{status:<7}] {check.name}: {check.detail}")
    return "\n".join(lines)
