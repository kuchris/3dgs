from __future__ import annotations

from dataclasses import dataclass


class GPUCheckError(RuntimeError):
    """Raised when PyTorch cannot complete the CUDA smoke test."""


@dataclass(frozen=True)
class GPUCheckResult:
    device_name: str
    total_memory_mib: int
    compute_capability: tuple[int, int]
    torch_version: str
    cuda_runtime: str


def run_gpu_check() -> GPUCheckResult:
    try:
        import torch
    except ImportError as error:
        raise GPUCheckError("PyTorch is not installed") from error

    if not torch.cuda.is_available():
        raise GPUCheckError("PyTorch cannot access a CUDA device")

    device = torch.device("cuda:0")
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device=device)
    actual = left @ right
    torch.cuda.synchronize(device)

    if not torch.equal(actual, expected):
        raise GPUCheckError("CUDA matrix multiplication returned an unexpected result")

    properties = torch.cuda.get_device_properties(device)
    return GPUCheckResult(
        device_name=properties.name,
        total_memory_mib=properties.total_memory // (1024 * 1024),
        compute_capability=torch.cuda.get_device_capability(device),
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda or "unknown",
    )


def format_gpu_report(result: GPUCheckResult) -> str:
    major, minor = result.compute_capability
    return "\n".join(
        [
            "PyTorch CUDA smoke test",
            f"[OK] Device: {result.device_name}",
            f"[OK] VRAM: {result.total_memory_mib} MiB",
            f"[OK] Compute capability: {major}.{minor}",
            f"[OK] PyTorch: {result.torch_version}",
            f"[OK] CUDA runtime: {result.cuda_runtime}",
            "[OK] Matrix multiplication: verified on cuda:0",
        ]
    )

