from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.eval.reference_inventory import (
    classify_tree_paths,
    reference_image_local_path,
    tree_entries_to_paths,
)


DATASET_BASE_URL = "https://huggingface.co/datasets/Voxel51/gaussian_splatting"
API_BASE_URL = "https://huggingface.co/api/datasets/Voxel51/gaussian_splatting/tree/main"
DEFAULT_SCENES = ("drjohnson", "playroom", "train", "truck")
DEFAULT_OUTPUT_DIR = Path("outputs/references")
DEFAULT_INVENTORY_PATH = DEFAULT_OUTPUT_DIR / "voxel51_reference_inventory.json"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ReferenceDownload:
    scene: str
    remote_path: str
    url: str
    path: Path
    exists: bool
    size_bytes: int


def voxel51_tree_api_url(scene: str) -> str:
    _validate_scene(scene)
    return f"{API_BASE_URL}/FO_dataset/{scene}?recursive=true"


def voxel51_resolve_url(remote_path: str) -> str:
    quoted_path = urllib.parse.quote(remote_path, safe="/")
    return f"{DATASET_BASE_URL}/resolve/main/{quoted_path}"


def fetch_scene_tree_entries(scene: str) -> list[dict[str, Any]]:
    url = voxel51_tree_api_url(scene)
    try:
        with urllib.request.urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to fetch Voxel51 tree for {scene}: {url}: {exc}") from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected Hugging Face tree API list for {scene}, got {type(payload).__name__}.")
    return [entry for entry in payload if isinstance(entry, dict)]


def make_reference_downloads(
    scene: str,
    remote_paths: list[str],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[ReferenceDownload]:
    _validate_scene(scene)
    downloads: list[ReferenceDownload] = []
    for remote_path in sorted(remote_paths):
        local_path = reference_image_local_path(remote_path, scene, output_dir)
        size = local_path.stat().st_size if local_path.exists() else 0
        downloads.append(
            ReferenceDownload(
                scene=scene,
                remote_path=remote_path,
                url=voxel51_resolve_url(remote_path),
                path=local_path,
                exists=size > 0,
                size_bytes=size,
            )
        )
    return downloads


def download_reference_image(download: ReferenceDownload) -> None:
    if download.exists:
        return

    download.path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = download.path.with_name(download.path.name + ".part")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with urllib.request.urlopen(download.url) as response:
            expected_size = response.headers.get("Content-Length")
            expected_size_int = int(expected_size) if expected_size and expected_size.isdigit() else None
            with partial_path.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Failed to download {download.remote_path}: {exc}") from exc

    size = partial_path.stat().st_size if partial_path.exists() else 0
    if size <= 0:
        partial_path.unlink()
        raise RuntimeError(f"Downloaded empty reference image: {download.url}")
    if expected_size_int is not None and size != expected_size_int:
        partial_path.unlink()
        raise RuntimeError(
            f"Partial reference download for {download.remote_path}: expected {expected_size_int} bytes, got {size}."
        )
    partial_path.replace(download.path)


def _parse_scene_list(value: str) -> tuple[str, ...]:
    scenes = tuple(scene.strip() for scene in value.split(",") if scene.strip())
    if not scenes:
        raise argparse.ArgumentTypeError("Expected at least one scene name.")
    for scene in scenes:
        _validate_scene(scene)
    return scenes


def _validate_scene(scene: str) -> None:
    if scene not in DEFAULT_SCENES:
        raise ValueError(f"Unsupported Voxel51 scene '{scene}'. Expected one of {DEFAULT_SCENES}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Voxel51 scene-root reference images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inventory-path", type=Path, default=DEFAULT_INVENTORY_PATH)
    parser.add_argument("--scenes", type=_parse_scene_list, default=DEFAULT_SCENES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    scene_records: list[dict[str, Any]] = []
    for scene in args.scenes:
        entries = fetch_scene_tree_entries(scene)
        inventory = classify_tree_paths(scene, tree_entries_to_paths(entries))
        downloads = make_reference_downloads(scene, inventory.image_paths, args.output_dir)

        print(f"scene    {scene}")
        print(f"         reference_images={len(downloads)} camera_alignment={inventory.camera_alignment}")
        for download in downloads:
            status = "skip" if download.exists else "download"
            print(f"{status:8} {download.path}")
            print(f"         {download.url}")

        if not args.dry_run:
            for download in downloads:
                if download.exists:
                    continue
                download_reference_image(download)
                print(f"done     {download.path} ({download.path.stat().st_size} bytes)")

        scene_records.append(
            {
                **asdict(inventory),
                "tree_api_url": voxel51_tree_api_url(scene),
                "downloads": [
                    {
                        "remote_path": download.remote_path,
                        "url": download.url,
                        "local_path": str(download.path),
                        "exists": download.exists or download.path.exists(),
                        "size_bytes": download.path.stat().st_size if download.path.exists() else 0,
                    }
                    for download in downloads
                ],
            }
        )

    payload = {
        "version": 1,
        "dataset": "Voxel51/gaussian_splatting",
        "scenes": scene_records,
        "dry_run": bool(args.dry_run),
        "notes": "camera_alignment is unavailable unless camera-like metadata is present in the dataset tree.",
    }
    if args.dry_run:
        print("dry-run: no files written")
        print(json.dumps(payload, indent=2))
        return 0

    args.inventory_path.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote    {args.inventory_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
