from __future__ import annotations

from pathlib import Path

from gs_gen.ply_probe import validate_3dgs_ply
from gs_gen.transforms_probe import validate_transforms


def validate_exported_asset(dataset_root: Path, splat_path: Path) -> dict[str, object]:
    path_checks = {
        "dataset_root_exists": dataset_root.exists(),
        "images_dir_exists": (dataset_root / "images").is_dir(),
        "transforms_exists": (dataset_root / "transforms.json").exists(),
        "splat_exists": splat_path.exists(),
    }
    transforms = validate_transforms(dataset_root / "transforms.json", dataset_root=dataset_root)
    ply = validate_3dgs_ply(splat_path)
    return {
        "valid": all(path_checks.values()) and bool(transforms["valid"]) and bool(ply["valid"]),
        "dataset_root": str(dataset_root),
        "splat_path": str(splat_path),
        "path_checks": path_checks,
        "transforms": transforms,
        "ply": ply,
    }


def format_validation_summary(result: dict[str, object]) -> str:
    transforms = result["transforms"]
    ply = result["ply"]
    lines = [
        f"valid: {result['valid']}",
        f"dataset_root: {result['dataset_root']}",
        f"splat_path: {result['splat_path']}",
        f"path_checks: {result['path_checks']}",
    ]
    if isinstance(transforms, dict):
        lines.extend(
            [
                f"frame_count: {transforms.get('frame_count', 0)}",
                f"transform_errors: {transforms.get('errors', [])}",
                f"missing_image_count: {transforms.get('missing_image_count', 0)}",
            ]
        )
    if isinstance(ply, dict):
        lines.extend([f"vertex_count: {ply.get('vertex_count', 0)}", f"ply_errors: {ply.get('errors', [])}"])
    return "\n".join(lines)
