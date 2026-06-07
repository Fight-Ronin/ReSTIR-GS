from __future__ import annotations

from pathlib import Path

from gs_gen.nerfstudio import build_nerfstudio_plan
from gs_gen.source_probe import SourceInput


def test_plan_uses_images_subcommand() -> None:
    plan = build_nerfstudio_plan(
        "my_room",
        SourceInput(kind="images", path=Path("data/my room/images")),
        workspace=Path("outputs/gsgen"),
    )

    assert "ns-process-data images" in plan.commands[0]
    assert '"data' in plan.commands[0]
    assert "my room" in plan.commands[0]
    assert "ns-train splatfacto" in plan.commands[1]
    assert "ns-export gaussian-splat" in plan.commands[2]
    assert "python -m gs_gen validate" in plan.commands[3]


def test_plan_uses_video_subcommand() -> None:
    plan = build_nerfstudio_plan(
        "my_room",
        SourceInput(kind="video", path=Path("data/my_room/walkthrough.mp4")),
        workspace=Path("outputs/gsgen"),
    )

    assert "ns-process-data video" in plan.commands[0]
    assert "walkthrough.mp4" in plan.commands[0]
