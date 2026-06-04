from __future__ import annotations

import json
import sys

from restir_gs.eval.real_asset_benchmark import load_benchmark_manifest
from scripts.download_voxel51_assets import (
    DEFAULT_SCENES,
    plan_voxel51_assets,
    voxel51_asset_path,
    voxel51_asset_url,
    main as download_main,
)


def test_voxel51_url_builder_matches_current_huggingface_tree() -> None:
    for scene in DEFAULT_SCENES:
        assert voxel51_asset_url(scene) == (
            "https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/main/"
            f"FO_dataset/{scene}/point_cloud/iteration_7000/point_cloud.ply"
        )


def test_voxel51_asset_paths_match_manifest_filenames(tmp_path) -> None:
    path = voxel51_asset_path("playroom", output_dir=tmp_path)

    assert path == tmp_path / "voxel51_playroom_iteration_7000_point_cloud.ply"


def test_voxel51_dry_run_does_not_write_files(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_voxel51_assets.py", "--dry-run"])

    assert download_main() == 0
    captured = capsys.readouterr()

    assert "dry-run: no files written" in captured.out
    assert len(plan_voxel51_assets(output_dir=tmp_path / "outputs/assets")) == 4
    assert not (tmp_path / "outputs").exists()


def test_voxel51_existing_non_empty_files_are_skipped(tmp_path) -> None:
    output_dir = tmp_path / "assets"
    existing = voxel51_asset_path("train", output_dir=output_dir)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"ply\n")

    assets = plan_voxel51_assets(["train"], output_dir=output_dir)

    assert assets[0].exists is True
    assert assets[0].size_bytes == 4


def test_real_asset_benchmark_manifest_includes_four_voxel51_scenes() -> None:
    manifest = load_benchmark_manifest("configs/real_asset_benchmark.json")
    scene_ids = [scene.scene_id for scene in manifest.scenes]

    assert scene_ids == [
        "voxel51_drjohnson",
        "voxel51_playroom",
        "voxel51_train",
        "voxel51_truck",
    ]
    assert len(manifest.scenes) == 4
    assert all(str(scene.ply).replace("\\", "/").startswith("outputs/assets/") for scene in manifest.scenes)


def test_real_asset_benchmark_manifest_json_has_four_scenes() -> None:
    data = json.loads(open("configs/real_asset_benchmark.json", encoding="utf-8").read())

    assert len(data["scenes"]) == 4
