from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_active_baseline_runner_chains_validation_then_snapshot() -> None:
    runner = (ROOT / "scripts" / "run_active_baseline_demo_windows.bat").read_text(encoding="utf-8")

    validation_index = runner.index("run_active_validation_windows.bat")
    snapshot_index = runner.index("run_active_demo_snapshot_windows.bat")

    assert validation_index < snapshot_index
    assert "if errorlevel 1" in runner


def test_active_baseline_handoff_documents_current_policy() -> None:
    doc = (ROOT / "docs" / "active_baseline_handoff.md").read_text(encoding="utf-8")

    for text in (
        "target_mode = visibility",
        "proposal = visibility_geometric",
        "temporal_filtered_ris",
        "scripts\\run_active_baseline_demo_windows.bat",
        "outputs/active_demo/active_renderer_snapshot_summary.json",
        "initial_ris_gpu_ms",
    ):
        assert text in doc


def test_readme_promotes_active_baseline_demo_runner() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "scripts\\run_active_baseline_demo_windows.bat" in readme
    assert "outputs/active_demo/active_renderer_snapshot_contact.png" in readme
