from pathlib import Path

from capture_studio.colmap_check import ColmapCheckResult, format_colmap_report


def test_colmap_report_contains_gpu_extraction_result() -> None:
    result = ColmapCheckResult(
        version="COLMAP 4.2.0 (Commit be5e291 on 2026-08-31 with CUDA)",
        executable=Path("C:/Tools/COLMAP/COLMAP.bat"),
        feature_count=1717,
    )

    report = format_colmap_report(result)

    assert "[OK] Version: COLMAP 4.2.0" in report
    assert "[OK] SIFT feature extraction: verified on GPU 0" in report
    assert "[OK] Features extracted: 1717" in report
