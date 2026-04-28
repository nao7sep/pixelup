from __future__ import annotations

from io import StringIO

from pixelup.reporting import Reporter, ReportMode


def test_human_verbose_reports_start_progress_and_elapsed_time() -> None:
    stderr = StringIO()
    reporter = Reporter(ReportMode.HUMAN, verbose=True, stderr=stderr)

    reporter.start(
        input_path="/tmp/in.png",
        output_path="/tmp/out.png",
        model="RealESRGAN_x4plus",
        scale=4,
        tiles=2,
    )
    reporter.progress(phase="load_model")
    reporter.success({"ok": True, "ms": 12})

    output = stderr.getvalue()
    assert "Input: /tmp/in.png" in output
    assert "tiles: 2" in output
    assert "load model..." in output
    assert "Completed in 12 ms" in output
