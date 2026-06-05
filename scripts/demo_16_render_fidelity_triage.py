from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BENCHMARK_SUMMARY = Path("outputs/benchmark/real_asset_benchmark_summary.json")
DEFAULT_REFERENCE_INVENTORY = Path("outputs/references/voxel51_reference_inventory.json")
DEFAULT_OUTPUT_DIR = Path("outputs/fidelity")


def load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def build_fidelity_summary(
    benchmark_summary: dict[str, Any],
    reference_inventory: dict[str, Any],
    benchmark_root: str | Path,
) -> dict[str, Any]:
    inventory_by_scene = {
        str(scene["scene_id"]): scene for scene in reference_inventory.get("scenes", []) if isinstance(scene, dict)
    }
    scene_records: list[dict[str, Any]] = []
    selected_view_count = 0
    reference_found_count = 0
    camera_metadata_found_count = 0

    for scene in benchmark_summary.get("scenes", []):
        scene_id = str(scene["scene_id"])
        inventory = inventory_by_scene.get(_voxel51_short_scene_id(scene_id), {})
        downloads = [item for item in inventory.get("downloads", []) if isinstance(item, dict)]
        local_reference_paths = [
            str(item["local_path"])
            for item in downloads
            if item.get("exists") and item.get("local_path")
        ]
        camera_metadata_paths = list(inventory.get("camera_metadata_paths", []))
        camera_alignment = str(inventory.get("camera_alignment", "unavailable"))
        if local_reference_paths:
            reference_found_count += 1
        if camera_metadata_paths:
            camera_metadata_found_count += 1

        selected_views: list[dict[str, Any]] = []
        for view in scene.get("selected_views", []):
            selected_view_count += 1
            preview_path = Path(str(view["preview_path"]))
            if not preview_path.is_absolute():
                preview_path = Path(benchmark_root) / scene_id / str(view["view_id"]) / "preview_rgb.png"
            selected_views.append(
                {
                    "view_id": str(view["view_id"]),
                    "preview_path": str(preview_path),
                    "camera_score": float(view["camera_score"]),
                    "valid_pixels": int(view["valid_pixels"]),
                    "comparison_status": "camera-aligned" if camera_alignment == "available" else "not camera-aligned",
                }
            )

        scene_records.append(
            {
                "scene_id": scene_id,
                "reference_images_found": len(local_reference_paths),
                "reference_image_paths": local_reference_paths,
                "camera_metadata_found": len(camera_metadata_paths),
                "camera_metadata_paths": camera_metadata_paths,
                "camera_alignment": camera_alignment,
                "loaded_count": int(scene["loaded_count"]),
                "original_count": int(scene["original_count"]),
                "selected_views": selected_views,
            }
        )

    return {
        "version": 1,
        "reference_images": "found" if reference_found_count > 0 else "missing",
        "camera_metadata": "found" if camera_metadata_found_count > 0 else "missing",
        "phase17_benchmark_role": (
            "photometric benchmark" if camera_metadata_found_count > 0 else "algorithm smoke benchmark"
        ),
        "selected_view_count": selected_view_count,
        "scene_count": len(scene_records),
        "scenes": scene_records,
    }


def make_contact_sheet(summary: dict[str, Any], output_path: str | Path) -> None:
    scenes = summary["scenes"]
    cell_w = 256
    cell_h = 192
    header_h = 56
    cols = 5
    rows = max(len(scenes), 1)
    sheet = Image.new("RGB", (cell_w * cols, header_h + cell_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "Voxel51 reference images vs Phase 17 auto-camera previews", fill=(0, 0, 0))
    draw.text((8, 28), f"role: {summary['phase17_benchmark_role']}", fill=(90, 20, 20))

    for row_index, scene in enumerate(scenes):
        y = header_h + row_index * cell_h
        scene_label = (
            f"{scene['scene_id']}\n"
            f"refs={scene['reference_images_found']} camera={scene['camera_alignment']}\n"
            f"loaded={scene['loaded_count']}/{scene['original_count']}"
        )
        draw.text((8, y + 8), scene_label, fill=(0, 0, 0))

        ref_image = _load_first_image(scene["reference_image_paths"])
        _paste_image_or_placeholder(sheet, draw, ref_image, (cell_w + 12, y + 28), "reference")

        for view_index, view in enumerate(scene["selected_views"][:3]):
            image = _load_image(view["preview_path"])
            x = (view_index + 2) * cell_w + 12
            label = (
                f"{view['view_id']} {view['comparison_status']}\n"
                f"score={view['camera_score']:.3f} valid={view['valid_pixels']}"
            )
            _paste_image_or_placeholder(sheet, draw, image, (x, y + 28), label)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _voxel51_short_scene_id(scene_id: str) -> str:
    return scene_id.removeprefix("voxel51_")


def _load_first_image(paths: list[str]) -> Image.Image | None:
    for path in paths:
        image = _load_image(path)
        if image is not None:
            return image
    return None


def _load_image(path: str | Path) -> Image.Image | None:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return None
    return Image.open(path).convert("RGB")


def _paste_image_or_placeholder(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image | None,
    xy: tuple[int, int],
    label: str,
) -> None:
    x, y = xy
    draw.text((x, y - 30), label, fill=(0, 0, 0))
    box = (x, y, x + 176, y + 132)
    if image is None:
        draw.rectangle(box, outline=(180, 180, 180), fill=(245, 245, 245))
        draw.text((x + 18, y + 56), "missing", fill=(100, 100, 100))
        return
    thumb = image.copy()
    thumb.thumbnail((176, 132))
    paste_x = x + (176 - thumb.width) // 2
    paste_y = y + (132 - thumb.height) // 2
    sheet.paste(thumb, (paste_x, paste_y))
    draw.rectangle(box, outline=(180, 180, 180))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Voxel51 reference images against Phase 17 auto-camera previews.")
    parser.add_argument("--benchmark-summary", type=Path, default=DEFAULT_BENCHMARK_SUMMARY)
    parser.add_argument("--reference-inventory", type=Path, default=DEFAULT_REFERENCE_INVENTORY)
    parser.add_argument("--benchmark-root", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    benchmark_summary = load_json(args.benchmark_summary)
    reference_inventory = load_json(args.reference_inventory)
    summary = build_fidelity_summary(benchmark_summary, reference_inventory, args.benchmark_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_path = args.output_dir / "voxel51_reference_vs_probe_contact.png"
    summary_path = args.output_dir / "fidelity_triage_summary.json"
    make_contact_sheet(summary, contact_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"reference_images: {summary['reference_images']}")
    print(f"camera_metadata:   {summary['camera_metadata']}")
    print(f"benchmark_role:    {summary['phase17_benchmark_role']}")
    print(f"selected_views:    {summary['selected_view_count']}")
    print(f"wrote:             {contact_path.resolve()}")
    print(f"wrote:             {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
