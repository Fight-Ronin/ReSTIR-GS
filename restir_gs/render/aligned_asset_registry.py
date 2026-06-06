from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from restir_gs.render.dxgl_asset import DxglAlignedAsset, load_dxgl_aligned_asset


DEFAULT_MANIFEST_PATH = Path("configs/aligned_assets.json")


@dataclass(frozen=True)
class AlignedAssetNormalization:
    camera_normalization: str
    normalization_rotation: str
    bbox_percentile: float


@dataclass(frozen=True)
class AlignedAssetSpec:
    asset_id: str
    dataset_type: str
    dataset_url: str
    dataset_root: Path
    splat_url: str
    splat_path: Path
    gaussian_schema: str
    max_gaussians: int
    default_frames: list[int]
    temporal_window: list[int]
    normalization: AlignedAssetNormalization


@dataclass(frozen=True)
class ResolvedAlignedAssetSpec:
    spec: AlignedAssetSpec
    repo_root: Path
    dataset_root: Path
    splat_path: Path


@dataclass(frozen=True)
class AlignedAssetManifest:
    path: Path
    repo_root: Path
    version: int
    asset_sets: dict[str, list[str]]
    assets: list[AlignedAssetSpec]


def load_aligned_asset_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> AlignedAssetManifest:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets_data = data.get("assets")
    if not isinstance(assets_data, list) or not assets_data:
        raise ValueError(f"Aligned asset manifest has no assets: {manifest_path}")
    repo_root = manifest_path.resolve().parent.parent
    assets = [_parse_asset_spec(item, manifest_path) for item in assets_data]
    asset_ids = [asset.asset_id for asset in assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError(f"Aligned asset manifest has duplicate asset_id values: {asset_ids}")
    asset_sets = _parse_asset_sets(data.get("asset_sets", {}), asset_ids, manifest_path)
    return AlignedAssetManifest(
        path=manifest_path,
        repo_root=repo_root,
        version=int(data.get("version", 1)),
        asset_sets=asset_sets,
        assets=assets,
    )


def get_aligned_asset_spec(manifest: AlignedAssetManifest, asset_id: str) -> AlignedAssetSpec:
    for spec in manifest.assets:
        if spec.asset_id == asset_id:
            return spec
    known = ", ".join(spec.asset_id for spec in manifest.assets)
    raise KeyError(f"Unknown aligned asset_id '{asset_id}'. Known assets: {known}")


def get_aligned_asset_set(manifest: AlignedAssetManifest, set_name: str) -> list[str]:
    if set_name not in manifest.asset_sets:
        known = ", ".join(sorted(manifest.asset_sets))
        raise KeyError(f"Unknown aligned asset_set '{set_name}'. Known sets: {known}")
    return list(manifest.asset_sets[set_name])


def resolve_requested_asset_ids(
    manifest: AlignedAssetManifest,
    asset_ids: list[str] | None = None,
    asset_set: str | None = None,
) -> list[str]:
    if asset_ids is not None:
        resolved = list(asset_ids)
    elif asset_set is not None:
        resolved = get_aligned_asset_set(manifest, asset_set)
    elif "testing" in manifest.asset_sets:
        resolved = get_aligned_asset_set(manifest, "testing")
    else:
        resolved = [asset.asset_id for asset in manifest.assets]
    _validate_asset_id_list(resolved, [asset.asset_id for asset in manifest.assets], "requested asset ids")
    return resolved


def resolve_aligned_asset_paths(
    spec: AlignedAssetSpec,
    repo_root: str | Path | None = None,
) -> ResolvedAlignedAssetSpec:
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    return ResolvedAlignedAssetSpec(
        spec=spec,
        repo_root=root,
        dataset_root=_resolve_path(root, spec.dataset_root),
        splat_path=_resolve_path(root, spec.splat_path),
    )


def load_registered_aligned_asset(
    spec: AlignedAssetSpec | ResolvedAlignedAssetSpec,
    device: torch.device | str = "cuda",
    max_gaussians_override: int | None = None,
) -> DxglAlignedAsset:
    resolved = spec if isinstance(spec, ResolvedAlignedAssetSpec) else resolve_aligned_asset_paths(spec)
    item = resolved.spec
    if item.dataset_type != "dxgl":
        raise ValueError(f"Unsupported aligned dataset_type '{item.dataset_type}'. Phase 28 supports 'dxgl' only.")
    max_gaussians = item.max_gaussians if max_gaussians_override is None else max_gaussians_override
    max_gaussians_value = None if max_gaussians <= 0 else max_gaussians
    return load_dxgl_aligned_asset(
        resolved.dataset_root,
        resolved.splat_path,
        device=device,
        max_gaussians=max_gaussians_value,
        gaussian_schema=item.gaussian_schema,
        camera_normalization=item.normalization.camera_normalization,
        normalization_rotation=item.normalization.normalization_rotation,
        normalization_bbox_percentile=item.normalization.bbox_percentile,
    )


def _parse_asset_spec(item: Any, manifest_path: Path) -> AlignedAssetSpec:
    if not isinstance(item, dict):
        raise ValueError(f"Aligned asset entry is not an object in {manifest_path}")
    normalization_data = item.get("normalization", {})
    if not isinstance(normalization_data, dict):
        raise ValueError(f"Aligned asset normalization is not an object: {item.get('asset_id')}")
    spec = AlignedAssetSpec(
        asset_id=_required_str(item, "asset_id", manifest_path),
        dataset_type=_required_str(item, "dataset_type", manifest_path),
        dataset_url=_required_str(item, "dataset_url", manifest_path),
        dataset_root=Path(_required_str(item, "dataset_root", manifest_path)),
        splat_url=_required_str(item, "splat_url", manifest_path),
        splat_path=Path(_required_str(item, "splat_path", manifest_path)),
        gaussian_schema=str(item.get("gaussian_schema", "auto")),
        max_gaussians=int(item.get("max_gaussians", 0)),
        default_frames=_required_int_list(item, "default_frames", manifest_path),
        temporal_window=_required_int_list(item, "temporal_window", manifest_path),
        normalization=AlignedAssetNormalization(
            camera_normalization=str(normalization_data.get("camera_normalization", "inferred_from_points3d")),
            normalization_rotation=str(normalization_data.get("normalization_rotation", "raw_y_to_z_up")),
            bbox_percentile=float(normalization_data.get("bbox_percentile", 0.98)),
        ),
    )
    if spec.dataset_type != "dxgl":
        raise ValueError(f"Unsupported aligned dataset_type '{spec.dataset_type}' for asset {spec.asset_id}")
    if spec.max_gaussians < 0:
        raise ValueError(f"Expected non-negative max_gaussians for asset {spec.asset_id}")
    if not 0.0 < spec.normalization.bbox_percentile <= 1.0:
        raise ValueError(f"Expected bbox_percentile in (0,1] for asset {spec.asset_id}")
    return spec


def _parse_asset_sets(asset_sets_data: Any, asset_ids: list[str], manifest_path: Path) -> dict[str, list[str]]:
    if asset_sets_data in (None, {}):
        return {}
    if not isinstance(asset_sets_data, dict):
        raise ValueError(f"Aligned asset manifest asset_sets must be an object: {manifest_path}")
    parsed: dict[str, list[str]] = {}
    for set_name, values in asset_sets_data.items():
        if not isinstance(set_name, str) or not set_name:
            raise ValueError(f"Aligned asset manifest has an invalid asset_set name in {manifest_path}")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Aligned asset_set '{set_name}' must be a non-empty list in {manifest_path}")
        set_asset_ids: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value:
                raise ValueError(f"Aligned asset_set '{set_name}' contains an invalid asset id in {manifest_path}")
            set_asset_ids.append(value)
        _validate_asset_id_list(set_asset_ids, asset_ids, f"asset_set '{set_name}'")
        parsed[set_name] = set_asset_ids
    return parsed


def _validate_asset_id_list(asset_ids: list[str], known_asset_ids: list[str], context: str) -> None:
    if not asset_ids:
        raise ValueError(f"Expected at least one asset id for {context}")
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError(f"Duplicate asset ids in {context}: {asset_ids}")
    unknown = [asset_id for asset_id in asset_ids if asset_id not in known_asset_ids]
    if unknown:
        known = ", ".join(known_asset_ids)
        raise KeyError(f"Unknown asset ids in {context}: {unknown}. Known assets: {known}")


def _required_str(item: dict[str, Any], key: str, manifest_path: Path) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Aligned asset entry missing required string '{key}' in {manifest_path}")
    return value


def _required_int_list(item: dict[str, Any], key: str, manifest_path: Path) -> list[int]:
    value = item.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Aligned asset entry missing required integer list '{key}' in {manifest_path}")
    values = [int(part) for part in value]
    if any(part < 0 for part in values):
        raise ValueError(f"Expected non-negative frame indices for '{key}' in {manifest_path}")
    return values


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path
