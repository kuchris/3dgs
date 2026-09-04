from capture_studio.gsplat_check import GSplatCheckResult, format_gsplat_report


def test_gsplat_report_contains_render_and_gradient_results() -> None:
    result = GSplatCheckResult(
        gsplat_version="1.5.3",
        device_name="NVIDIA GeForce RTX 5070 Ti",
        render_shape=(1, 64, 64, 3),
        nonzero_alpha_pixels=1176,
    )

    report = format_gsplat_report(result)

    assert "[OK] Device: NVIDIA GeForce RTX 5070 Ti" in report
    assert "[OK] Render shape: (1, 64, 64, 3)" in report
    assert "[OK] Visible alpha pixels: 1176" in report
    assert "[OK] Backpropagation: finite color gradient" in report
