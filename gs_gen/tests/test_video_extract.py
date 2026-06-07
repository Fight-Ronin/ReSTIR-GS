from __future__ import annotations

import pytest

from gs_gen.video_extract import compute_extract_indices


def test_compute_extract_indices_downsamples_by_fps_ratio() -> None:
    assert compute_extract_indices(frame_count=10, source_fps=30.0, target_fps=5.0) == [0, 6]


def test_compute_extract_indices_caps_max_frames() -> None:
    assert compute_extract_indices(frame_count=100, source_fps=20.0, target_fps=10.0, max_frames=3) == [0, 2, 4]


def test_compute_extract_indices_handles_unknown_source_fps() -> None:
    assert compute_extract_indices(frame_count=3, source_fps=0.0, target_fps=5.0) == [0, 1, 2]


def test_compute_extract_indices_rejects_invalid_target_fps() -> None:
    with pytest.raises(ValueError, match="target_fps"):
        compute_extract_indices(frame_count=10, source_fps=30.0, target_fps=0.0)
