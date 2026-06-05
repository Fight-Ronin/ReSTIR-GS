from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
CAMERA_METADATA_MARKERS = (
    "transforms.json",
    "cameras.json",
    "cameras.txt",
    "images.txt",
    "sparse/",
    "colmap/",
)


@dataclass(frozen=True)
class ReferenceInventory:
    scene_id: str
    image_paths: list[str]
    camera_metadata_paths: list[str]
    camera_alignment: str


def classify_tree_paths(scene_id: str, paths: list[str]) -> ReferenceInventory:
    """Classify Hugging Face repo tree paths for one Voxel51 scene."""
    image_paths = sorted(path for path in paths if _is_scene_root_image(path))
    camera_metadata_paths = sorted(path for path in paths if _is_camera_metadata_path(path))
    camera_alignment = "available" if camera_metadata_paths else "unavailable"
    return ReferenceInventory(
        scene_id=scene_id,
        image_paths=image_paths,
        camera_metadata_paths=camera_metadata_paths,
        camera_alignment=camera_alignment,
    )


def tree_entries_to_paths(entries: list[dict[str, Any]]) -> list[str]:
    """Extract path strings from Hugging Face tree API entries."""
    paths: list[str] = []
    for entry in entries:
        path = entry.get("path")
        if isinstance(path, str) and path:
            paths.append(path)
    return paths


def reference_image_local_path(remote_path: str, scene: str, output_dir: str | Path) -> Path:
    """Map a remote Voxel51 reference path to its local output path."""
    return Path(output_dir) / f"voxel51_{scene}" / Path(remote_path).name


def _is_scene_root_image(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return False
    # Scene-root references look like FO_dataset/playroom/DSC05572.jpg.
    return len(Path(path).parts) == 3


def _is_camera_metadata_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(marker in normalized for marker in CAMERA_METADATA_MARKERS)
