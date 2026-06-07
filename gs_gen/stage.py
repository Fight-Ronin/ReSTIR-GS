from __future__ import annotations

import json
from pathlib import Path
import shutil

from gs_gen.paths import GsGenPaths, validate_asset_id
from gs_gen.validate import validate_exported_asset


def stage_asset(
    asset_id: str,
    dataset_root: Path,
    splat_path: Path,
    workspace: Path,
    copy_images: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    asset_id = validate_asset_id(asset_id)
    validation = validate_exported_asset(dataset_root, splat_path)
    paths = GsGenPaths(asset_id=asset_id, workspace=workspace)
    payload = {
        "version": 1,
        "asset_id": asset_id,
        "source": "nerfstudio_splatfacto",
        "dataset_root": str(dataset_root),
        "splat_path": str(splat_path),
        "staged_dir": str(paths.staged_dir),
        "images_copied": copy_images,
        "validation": validation,
    }
    if dry_run or not validation["valid"]:
        return payload

    paths.staged_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_root / "transforms.json", paths.staged_dir / "transforms.json")
    shutil.copy2(splat_path, paths.staged_dir / "splat.ply")
    if copy_images:
        _copytree_replace(dataset_root / "images", paths.staged_dir / "images")
    (paths.staged_dir / "asset_info.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _copytree_replace(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
