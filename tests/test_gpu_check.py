from capture_studio.gpu_check import GPUCheckResult, format_gpu_report


def test_gpu_report_contains_device_and_verified_calculation() -> None:
    result = GPUCheckResult(
        device_name="NVIDIA GeForce RTX 5070 Ti",
        total_memory_mib=16303,
        compute_capability=(12, 0),
        torch_version="2.10.0+cu130",
        cuda_runtime="13.0",
    )

    report = format_gpu_report(result)

    assert "[OK] Device: NVIDIA GeForce RTX 5070 Ti" in report
    assert "[OK] Compute capability: 12.0" in report
    assert "[OK] Matrix multiplication: verified on cuda:0" in report
