from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from restir_gs.render.aligned_asset_registry import (
    get_aligned_asset_set,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
    resolve_requested_asset_ids,
)
from restir_gs.render.ply_loader import GaussianAssetStats, LoadedGaussianAsset
from restir_gs.render.synthetic_scene import SyntheticGaussians
from restir_gs.render.transforms_loader import ImportedTransforms
from scripts.demo_24_aligned_asset_smoke_matrix import make_smoke_row, parse_asset_ids
from scripts.download_aligned_asset import main as download_asset_main
from scripts.download_aligned_splat import main as download_splat_main


def test_default_manifest_contains_dxgl_apple() -> None:
    manifest = load_aligned_asset_manifest("configs/aligned_assets.json")
    asset_ids = [asset.asset_id for asset in manifest.assets]

    assert manifest.version == 1
    assert asset_ids == [
        "dxgl_apple",
        "dxgl_cash_register",
        "dxgl_drill",
        "dxgl_fire_extinguisher",
        "dxgl_led_lightbulb",
        "dxgl_measuring_tape",
        "dxgl_modern_arm_chair",
        "dxgl_multi_cleaner_5l",
        "dxgl_potted_plant",
        "dxgl_wet_floor_sign",
    ]
    assert get_aligned_asset_set(manifest, "smoke") == ["dxgl_apple"]
    assert get_aligned_asset_set(manifest, "testing") == asset_ids
    for asset_id in asset_ids:
        spec = get_aligned_asset_spec(manifest, asset_id)
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


def test_manifest_asset_set_validation_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    path.write_text(json.dumps({"version": 1, "assets": [_asset_payload()], "asset_sets": {"empty": []}}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty list"):
        load_aligned_asset_manifest(path)

    path.write_text(json.dumps({"version": 1, "assets": [_asset_payload()], "asset_sets": {"bad": ["missing"]}}), encoding="utf-8")
    with pytest.raises(KeyError, match="Unknown asset ids"):
        load_aligned_asset_manifest(path)

    path.write_text(json.dumps({"version": 1, "assets": [_asset_payload()], "asset_sets": {"dupe": ["fixture", "fixture"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate asset ids"):
        load_aligned_asset_manifest(path)


def test_resolve_requested_asset_ids_precedence_and_defaults(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    second = _asset_payload()
    second["asset_id"] = "second"
    second["dataset_root"] = "outputs/second"
    second["splat_path"] = "outputs/second_splat/second.ply"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "asset_sets": {"smoke": ["fixture"], "testing": ["fixture", "second"]},
                "assets": [_asset_payload(), second],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_aligned_asset_manifest(manifest_path)

    assert resolve_requested_asset_ids(manifest) == ["fixture", "second"]
    assert resolve_requested_asset_ids(manifest, asset_set="smoke") == ["fixture"]
    assert resolve_requested_asset_ids(manifest, asset_ids=["second"], asset_set="smoke") == ["second"]
    with pytest.raises(KeyError, match="Unknown aligned asset_set"):
        resolve_requested_asset_ids(manifest, asset_set="missing")
    with pytest.raises(ValueError, match="Duplicate asset ids"):
        resolve_requested_asset_ids(manifest, asset_ids=["fixture", "fixture"])


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


def test_generic_download_asset_set_dry_runs_write_no_files(tmp_path: Path, monkeypatch, capsys) -> None:
    second = _asset_payload()
    second["asset_id"] = "second"
    second["dataset_url"] = "https://example.com/second.zip"
    second["dataset_root"] = "outputs/second"
    second["splat_url"] = "https://example.com/second.ply"
    second["splat_path"] = "outputs/second_splat/second.ply"
    manifest_path = tmp_path / "configs" / "aligned_assets.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps({"version": 1, "asset_sets": {"testing": ["fixture", "second"]}, "assets": [_asset_payload(), second]}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["download_aligned_asset.py", "--manifest", str(manifest_path), "--asset-set", "testing", "--dry-run"])
    assert download_asset_main() == 0
    asset_out = capsys.readouterr().out
    assert "https://example.com/data.zip" in asset_out
    assert "https://example.com/second.zip" in asset_out

    monkeypatch.setattr(sys, "argv", ["download_aligned_splat.py", "--manifest", str(manifest_path), "--asset-set", "testing", "--dry-run"])
    assert download_splat_main() == 0
    splat_out = capsys.readouterr().out
    assert "https://example.com/fixture.ply" in splat_out
    assert "https://example.com/second.ply" in splat_out
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

    assert "scripts\\_setup_windows_cuda_env.bat" in text
    assert "RESTIRGS_ALIGNED_MANIFEST" in text
    assert "RESTIRGS_ALIGNED_ASSET_SET" in text
    assert "RESTIRGS_ALIGNED_ASSET_IDS" in text
    assert "RESTIRGS_ALIGNED_SMOKE_EXTRA_ARGS" in text
    assert "scripts\\demo_24_aligned_asset_smoke_matrix.py" in text
    assert "--manifest" in text
    assert "--asset-set" in text
    assert "--asset-ids" in text
    assert "--device cuda" in text


def test_active_validation_runner_chains_current_active_runners() -> None:
    runner = Path("scripts/run_active_validation_windows.bat")
    text = runner.read_text(encoding="utf-8")

    assert "scripts\\_setup_windows_cuda_env.bat" in text
    assert "RESTIRGS_ALIGNED_MANIFEST" in text
    assert "RESTIRGS_ALIGNED_ASSET_SET" in text
    assert "RESTIRGS_ALIGNED_ASSET_IDS" in text
    assert "RESTIRGS_ALIGNED_SMOKE_EXTRA_ARGS" in text
    assert "scripts\\demo_24_aligned_asset_smoke_matrix.py" in text
    assert "scripts\\run_aligned_restir_renderer_windows.bat" in text


def test_restir_renderer_windows_runner_documents_active_visibility_defaults() -> None:
    runner = Path("scripts/run_aligned_restir_renderer_windows.bat")
    text = runner.read_text(encoding="utf-8")

    assert "RESTIRGS_RESTIR_TARGET_MODE=visibility" in text
    assert "RESTIRGS_RESTIR_NUM_LIGHTS=16" in text
    assert "RESTIRGS_RESTIR_WIDTH=128" in text
    assert "RESTIRGS_RESTIR_HEIGHT=128" in text
    assert "RESTIRGS_RESTIR_FRAME_INDICES=45,46,47" in text
    assert "RESTIRGS_RESTIR_FRAME_INDICES%\"==\"manifest" in text
    assert "RESTIRGS_RESTIR_OUTPUT_DIR=outputs\\aligned_restir" in text
    assert "RESTIRGS_RESTIR_VISIBILITY_SHADOW_PCF_RADIUS=1" in text
    assert "RESTIRGS_RESTIR_TEMPORAL_HISTORY_M_CAP" in text
    assert "RESTIRGS_RESTIR_EXTRA_ARGS" in text
    assert "scripts\\demo_26_aligned_restir_renderer.py" in text
    assert "--target-mode" in text
    assert "--num-lights" in text
    assert "--temporal-history-m-cap" in text
    assert "--visibility-shadow-pcf-radius" in text


def test_active_windows_runners_share_cuda_preflight() -> None:
    for path in (
        Path("scripts/run_active_validation_windows.bat"),
        Path("scripts/run_aligned_asset_smoke_matrix_windows.bat"),
        Path("scripts/run_aligned_restir_renderer_windows.bat"),
        Path("scripts/run_interactive_viewer_windows.bat"),
    ):
        assert "scripts\\_setup_windows_cuda_env.bat" in path.read_text(encoding="utf-8")


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
