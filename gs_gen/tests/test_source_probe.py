from __future__ import annotations

from gs_gen.source_probe import SourceInput, probe_source


def test_probe_images_counts_supported_files(tmp_path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "a.jpg").write_bytes(b"x")
    (root / "b.png").write_bytes(b"x")
    (root / "notes.txt").write_text("ignore", encoding="utf-8")

    result = probe_source(SourceInput(kind="images", path=root))

    assert result["valid"] is True
    assert result["image_count"] == 2


def test_probe_video_accepts_common_suffix(tmp_path) -> None:
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"x")

    result = probe_source(SourceInput(kind="video", path=video))

    assert result["valid"] is True
    assert result["suffix"] == ".mp4"


def test_probe_video_rejects_unknown_suffix(tmp_path) -> None:
    video = tmp_path / "walkthrough.txt"
    video.write_bytes(b"x")

    result = probe_source(SourceInput(kind="video", path=video))

    assert result["valid"] is False
    assert "unsupported video suffix" in result["errors"][0]
