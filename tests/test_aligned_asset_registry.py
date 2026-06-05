from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from restir_gs.render.aligned_asset_registry import (
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
)
from restir_gs.render.ply_loader import GaussianAssetStats, LoadedGaussianAsset
from restir_gs.render.synthetic_scene import SyntheticGaussians
from restir_gs.render.transforms_loader import ImportedTransforms
from scripts.demo_24_aligned_asset_smoke_matrix import make_smoke_row, parse_asset_ids
from scripts.download_aligned_asset import main as download_asset_main
from scripts.download_aligned_splat import main as download_splat_main


def test_default_manifest_contains_dxgl_apple() -> None:
    manifest = load_aligned_asset_manifest("configs/aligned_assets.json")
    spec = get_aligned_asset_spec(manifest, "dxgl_apple")

    assert manifest.version == 1
    assert spec.dataset_type == "dxgl"
    assert spec.gaussian_schema == "auto"
    assert spec.default_frames == [0, 49, 98, 147]
    assert spec.temporal_window == [45, 46, 47, 48, 49, 50, 51, 52, 53]


def test_manifest_missing_required_fields_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = _asset_payload()
    payload.pop("dataset_root")
    path.write_text(json.dumps({"version": 1, "assets": [payload]}), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_root"):
        load_aligned_asset_manifest(path)


def test_manifest_unsupported_dataset_type_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "assets": [
                    {
                        "asset_id": "bad",
                        "dataset_type": "unknown",
                        "dataset_url": "https://example.com/data.zip",
                        "dataset_root": "outputs/bad",
                        "splat_url": "https://example.com/bad.ply",
                        "splat_path": "outputs/bad.ply",
                        "default_frames": [0],
                        "temporal_window": [0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported aligned dataset_type"):
        load_aligned_asset_manifest(path)


def test_path_resolution_uses_manifest_repo_root(tmp_path: Path) -> None:
    manifest_path = tmp_path / "configs" / "aligned_assets.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"version": 1, "assets": [_asset_payload()]}), encoding="utf-8")
    manifest = load_aligned_asset_manifest(manifest_path)
    spec = get_aligned_asset_spec(manifest, "fixture")

    resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)

    assert resolved.dataset_root == tmp_path / "outputs" / "fixture"
    assert resolved.splat_path == tmp_path / "outputs" / "fixture_splat" / "fixture.ply"


def test_registered_loader_routes_splat_through_generic_loader(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "configs" / "aligned_assets.json"
    manifest_path.parent.mkdir()
    payload = _asset_payload()
    payload["normalization"] = {"camera_normalization": "none", "normalization_rotation": "raw_y_to_z_up", "bbox_percentile": 0.98}
    manifest_path.write_text(json.dumps({"version": 1, "assets": [payload]}), encoding="utf-8")
    manifest = load_aligned_asset_manifest(manifest_path)
    resolved = resolve_aligned_asset_paths(get_aligned_asset_spec(manifest, "fixture"), repo_root=manifest.repo_root)
    calls: dict[str, object] = {}

    def fake_load_gaussian_asset(path, device="cuda", max_gaussians=None, schema="auto"):
        calls["path"] = Path(path)
        calls["schema"] = schema
        return LoadedGaussianAsset(
            scene=SyntheticGaussians(
                means=torch.zeros((1, 3)),
                quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                scales=torch.ones((1, 3)),
                opacities=torch.ones((1,)),
                colors=torch.ones((1, 3)),
            ),
            stats=GaussianAssetStats(
                path=str(path),
                source_format="ply",
                schema="graphdeco_3dgs_ply",
                original_count=1,
                loaded_count=1,
                color_source="f_dc",
                has_sh_rest=False,
            ),
        )

    def fake_load_transforms(path, dataset_root=None, device="cpu", camera_normalization=None):
        return ImportedTransforms(path=Path(path), dataset_root=Path(dataset_root), width=1, height=1, frame_count=0, frames=[])

    monkeypatch.setattr("restir_gs.render.dxgl_asset.load_gaussian_asset", fake_load_gaussian_asset)
    monkeypatch.setattr("restir_gs.render.dxgl_asset.load_nerfstudio_transforms", fake_load_transforms)

    loaded = load_registered_aligned_asset(resolved, device="cpu")

    assert calls["path"] == resolved.splat_path
    assert calls["schema"] == "auto"
    assert loaded.loaded.stats.loaded_count == 1


def test_generic_download_dry_runs_write_no_files(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest_path = tmp_path / "configs" / "aligned_assets.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({"version": 1, "assets": [_asset_payload()]}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["download_aligned_asset.py", "--manifest", str(manifest_path), "--asset-id", "fixture", "--dry-run"])
    assert download_asset_main() == 0
    asset_out = capsys.readouterr().out
    assert "https://example.com/data.zip" in asset_out
    assert "dry-run: no files written" in asset_out

    monkeypatch.setattr(sys, "argv", ["download_aligned_splat.py", "--manifest", str(manifest_path), "--asset-id", "fixture", "--dry-run"])
    assert download_splat_main() == 0
    splat_out = capsys.readouterr().out
    assert "https://example.com/fixture.ply" in splat_out
    assert "dry-run: no files written" in splat_out
    assert not (tmp_path / "outputs").exists()


def test_smoke_row_normalizer_records_required_fields() -> None:
    row = make_smoke_row(
        "fixture",
        "dxgl",
        "gbuffer",
        "rgb_mae",
        0.25,
        frame_index=7,
        loaded_count=11,
        original_count=13,
        valid_pixels=17,
    )

    assert row["asset_id"] == "fixture"
    assert row["dataset_type"] == "dxgl"
    assert row["stage"] == "gbuffer"
    assert row["frame_index"] == 7
    assert row["loaded_count"] == 11
    assert row["original_count"] == 13
    assert row["valid_pixels"] == 17
    assert row["metric_name"] == "rgb_mae"
    assert row["metric_value"] == 0.25
    assert row["finite"] == "true"


def test_smoke_matrix_asset_id_parser_accepts_csv_and_rejects_empty() -> None:
    assert parse_asset_ids("dxgl_apple, fixture_asset") == ["dxgl_apple", "fixture_asset"]

    with pytest.raises(ValueError, match="Expected at least one asset id"):
        parse_asset_ids(" , ")


def test_smoke_matrix_windows_runner_documents_env_command_surface() -> None:
    runner = Path("scripts/run_aligned_asset_smoke_matrix_windows.bat")
    text = runner.read_text(encoding="utf-8")

    assert "RESTIRGS_ALIGNED_MANIFEST" in text
    assert "RESTIRGS_ALIGNED_ASSET_IDS" in text
    assert "RESTIRGS_ALIGNED_SMOKE_EXTRA_ARGS" in text
    assert "scripts\\demo_24_aligned_asset_smoke_matrix.py" in text
    assert "--manifest" in text
    assert "--asset-ids" in text
    assert "--device cuda" in text


def _asset_payload() -> dict[str, object]:
    return {
        "asset_id": "fixture",
        "dataset_type": "dxgl",
        "dataset_url": "https://example.com/data.zip",
        "dataset_root": "outputs/fixture",
        "splat_url": "https://example.com/fixture.ply",
        "splat_path": "outputs/fixture_splat/fixture.ply",
        "gaussian_schema": "auto",
        "max_gaussians": 0,
        "default_frames": [0],
        "temporal_window": [0],
        "normalization": {
            "camera_normalization": "inferred_from_points3d",
            "normalization_rotation": "raw_y_to_z_up",
            "bbox_percentile": 0.98,
        },
    }
