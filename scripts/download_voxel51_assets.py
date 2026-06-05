from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DATASET_BASE_URL = "https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/main"
DEFAULT_SCENES = ("drjohnson", "playroom", "train", "truck")
DEFAULT_ITERATION = 7000
DEFAULT_OUTPUT_DIR = Path("outputs/assets")
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Voxel51Asset:
    scene: str
    iteration: int
    url: str
    path: Path
    exists: bool
    size_bytes: int


def voxel51_asset_url(scene: str, iteration: int = DEFAULT_ITERATION) -> str:
    """Return the direct Hugging Face resolve URL for a Voxel51 3DGS PLY."""
    _validate_scene(scene)
    if iteration != DEFAULT_ITERATION:
        raise ValueError("Phase 17 only supports Voxel51 iteration 7000 assets.")
    return (
        f"{DATASET_BASE_URL}/FO_dataset/{scene}/"
        f"point_cloud/iteration_{iteration}/point_cloud.ply"
    )


def voxel51_asset_path(
    scene: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    iteration: int = DEFAULT_ITERATION,
) -> Path:
    """Return the local canonical benchmark asset path for a Voxel51 scene."""
    _validate_scene(scene)
    return Path(output_dir) / f"voxel51_{scene}_iteration_{iteration}_point_cloud.ply"


def plan_voxel51_assets(
    scenes: tuple[str, ...] | list[str] = DEFAULT_SCENES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    iteration: int = DEFAULT_ITERATION,
) -> list[Voxel51Asset]:
    """Build the asset download plan without touching the filesystem."""
    assets: list[Voxel51Asset] = []
    for scene in scenes:
        path = voxel51_asset_path(scene, output_dir=output_dir, iteration=iteration)
        size = path.stat().st_size if path.exists() else 0
        assets.append(
            Voxel51Asset(
                scene=scene,
                iteration=iteration,
                url=voxel51_asset_url(scene, iteration=iteration),
                path=path,
                exists=size > 0,
                size_bytes=size,
            )
        )
    return assets


def download_voxel51_asset(asset: Voxel51Asset) -> None:
    """Download one asset atomically, failing on empty or partial files."""
    if asset.exists:
        return

    asset.path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = asset.path.with_name(asset.path.name + ".part")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with urllib.request.urlopen(asset.url) as response:
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
        raise RuntimeError(f"Failed to download {asset.scene} from {asset.url}: {exc}") from exc

    size = partial_path.stat().st_size if partial_path.exists() else 0
    if size <= 0:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Downloaded empty asset for {asset.scene}: {asset.url}")
    if expected_size_int is not None and size != expected_size_int:
        partial_path.unlink()
        raise RuntimeError(
            f"Partial download for {asset.scene}: expected {expected_size_int} bytes, got {size} bytes."
        )

    partial_path.replace(asset.path)


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
    parser = argparse.ArgumentParser(description="Download public Voxel51 7000-iteration 3DGS PLY assets.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenes", type=_parse_scene_list, default=DEFAULT_SCENES)
    parser.add_argument("--iteration", type=int, default=DEFAULT_ITERATION, choices=[DEFAULT_ITERATION])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assets = plan_voxel51_assets(args.scenes, output_dir=args.output_dir, iteration=args.iteration)
    for asset in assets:
        status = "skip" if asset.exists else "download"
        print(f"{status:8} {asset.scene:10} -> {asset.path}")
        print(f"         {asset.url}")

    if args.dry_run:
        print("dry-run: no files written")
        return 0

    for asset in assets:
        if asset.exists:
            continue
        download_voxel51_asset(asset)
        size = asset.path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Downloaded asset is missing or empty: {asset.path}")
        print(f"done     {asset.scene:10} -> {asset.path} ({size} bytes)")

    missing = [asset.path for asset in plan_voxel51_assets(args.scenes, args.output_dir, args.iteration) if not asset.exists]
    if missing:
        raise RuntimeError(f"Missing or empty downloaded assets: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
