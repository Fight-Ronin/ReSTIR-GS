from __future__ import annotations

import json
from pathlib import Path

from gs_gen.stage import stage_asset
from gs_gen.tests.fixtures import graphdeco_properties, write_dataset, write_ply


def test_stage_asset_writes_stable_folder(tmp_path: Path) -> None:
    dataset_root = write_dataset(tmp_path)
    splat = tmp_path / "splat.ply"
    write_ply(splat, graphdeco_properties())

    result = stage_asset(
        "my_room",
        dataset_root=dataset_root,
        splat_path=splat,
        workspace=tmp_path / "workspace",
        copy_images=True,
    )

    staged = tmp_path / "workspace" / "my_room" / "staged"
    assert result["validation"]["valid"] is True
    assert (staged / "transforms.json").exists()
    assert (staged / "splat.ply").exists()
    assert (staged / "images" / "frame_000.png").exists()
    info = json.loads((staged / "asset_info.json").read_text(encoding="utf-8"))
    assert info["asset_id"] == "my_room"


def test_stage_asset_dry_run_writes_no_files(tmp_path: Path) -> None:
    dataset_root = write_dataset(tmp_path)
    splat = tmp_path / "splat.ply"
    write_ply(splat, graphdeco_properties())

    result = stage_asset(
        "my_room",
        dataset_root=dataset_root,
        splat_path=splat,
        workspace=tmp_path / "workspace",
        dry_run=True,
    )

    assert result["validation"]["valid"] is True
    assert not (tmp_path / "workspace").exists()
