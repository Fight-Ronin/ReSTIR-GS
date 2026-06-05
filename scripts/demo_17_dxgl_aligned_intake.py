from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.render.ply_loader import load_gaussian_ply_with_stats
from restir_gs.render.transforms_loader import (
    ImportedTransformFrame,
    camera_to_config_payload,
    load_nerfstudio_transforms,
)
from scripts.download_dxgl_apple import DEFAULT_EXTRACT_DIR, find_dxgl_dataset_root, validate_dxgl_dataset_root


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_fidelity")
DEFAULT_FRAME_INDICES = (0, 49, 98, 147)


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer value, got {text!r}")
    return values


def probe_points3d_compatibility(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return {"path": str(path), "exists": False, "splat_compatible": False, "error": "missing points3D.ply"}
    try:
        loaded = load_gaussian_ply_with_stats(path, device="cpu", max_gaussians=16)
    except Exception as exc:
        return {"path": str(path), "exists": True, "splat_compatible": False, "error": str(exc)}
    return {
        "path": str(path),
        "exists": True,
        "splat_compatible": True,
        "original_count": loaded.stats.original_count,
        "loaded_count": loaded.stats.loaded_count,
        "color_source": loaded.stats.color_source,
    }


def make_dxgl_contact_sheet(frames: list[ImportedTransformFrame], output_path: str | Path) -> None:
    cell_w = 192
    cell_h = 156
    label_w = 152
    header_h = 36
    cols = 4
    rows = max(len(frames), 1)
    sheet = Image.new("RGB", (label_w + cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), "DXGL Apple aligned intake: RGB / mask / depth / normal", fill=(0, 0, 0))
    for row, frame in enumerate(frames):
        y = header_h + row * cell_h
        draw.text((8, y + 12), f"frame {frame.index}\n{Path(frame.file_path).name}", fill=(0, 0, 0))
        images = [
            ("RGB", _load_display_image(frame.image_path, "rgb")),
            ("Mask", _load_display_image(frame.mask_path, "mask")),
            ("Depth", _load_display_image(frame.depth_path or frame.depth_16bit_path, "depth")),
            ("Normal", _load_display_image(frame.normal_path, "normal")),
        ]
        for col, (label, image) in enumerate(images):
            x = label_w + col * cell_w
            draw.text((x + 8, y + 8), label, fill=(0, 0, 0))
            _paste_thumbnail(sheet, draw, image, (x + 8, y + 28), (176, 116))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def build_intake_summary(
    dataset_root: Path,
    selected_frames: list[ImportedTransformFrame],
    camera_config_paths: list[Path],
    points3d_probe: dict[str, Any],
    contact_path: Path,
    frame_count: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "dataset": "dxgl_polyhaven_10_apple",
        "dataset_root": str(dataset_root),
        "frame_count": frame_count,
        "selected_frame_indices": [frame.index for frame in selected_frames],
        "contact_sheet": str(contact_path),
        "selected_frames": [
            {
                "index": frame.index,
                "file_path": frame.file_path,
                "image_path": str(frame.image_path),
                "mask_path": str(frame.mask_path) if frame.mask_path is not None else None,
                "depth_path": str(frame.depth_path) if frame.depth_path is not None else None,
                "depth_16bit_path": str(frame.depth_16bit_path) if frame.depth_16bit_path is not None else None,
                "normal_path": str(frame.normal_path) if frame.normal_path is not None else None,
                "camera_config_path": str(camera_config_paths[index]),
                "camera_width": frame.camera.width,
                "camera_height": frame.camera.height,
            }
            for index, frame in enumerate(selected_frames)
        ],
        "points3d_probe": points3d_probe,
        "photometric_metrics": "not_computed_without_compatible_3dgs_render",
    }


def _load_display_image(path: Path | None, kind: str) -> Image.Image | None:
    if path is None or not path.exists() or path.stat().st_size <= 0:
        return None
    image = Image.open(path)
    if kind == "depth":
        return ImageOps.autocontrast(image.convert("L")).convert("RGB")
    if kind == "mask":
        return image.convert("L").convert("RGB")
    return image.convert("RGB")


def _paste_thumbnail(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image | None,
    xy: tuple[int, int],
    size: tuple[int, int],
) -> None:
    x, y = xy
    width, height = size
    box = (x, y, x + width, y + height)
    if image is None:
        draw.rectangle(box, outline=(190, 190, 190), fill=(245, 245, 245))
        draw.text((x + 48, y + 50), "missing", fill=(100, 100, 100))
        return
    thumb = image.copy()
    thumb.thumbnail(size)
    paste_x = x + (width - thumb.width) // 2
    paste_y = y + (height - thumb.height) // 2
    sheet.paste(thumb, (paste_x, paste_y))
    draw.rectangle(box, outline=(180, 180, 180))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect DXGL Apple aligned RGB/depth/normal/mask/camera intake.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-indices", type=parse_int_list, default=list(DEFAULT_FRAME_INDICES))
    args = parser.parse_args()

    dataset_root = find_dxgl_dataset_root(args.dataset_root, required=True)
    validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    imported = load_nerfstudio_transforms(dataset_root / "transforms.json", dataset_root=dataset_root, device="cpu")
    selected_indices = list(args.frame_indices)
    if max(selected_indices) >= imported.frame_count or min(selected_indices) < 0:
        raise ValueError(f"Selected frame indices {selected_indices} exceed frame count {imported.frame_count}")
    selected_frames = [imported.frames[index] for index in selected_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera_config_paths: list[Path] = []
    for frame in selected_frames:
        path = args.output_dir / f"dxgl_apple_frame_{frame.index:03d}_camera.json"
        payload = camera_to_config_payload(
            frame.camera,
            metadata={
                "dataset": "dxgl_polyhaven_10_apple",
                "frame_index": frame.index,
                "file_path": frame.file_path,
                "source_convention": "nerfstudio_opengl_c2w",
                "project_convention": "world_to_camera_plus_z_forward_y_down",
            },
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        camera_config_paths.append(path)

    contact_path = args.output_dir / "dxgl_apple_contact.png"
    make_dxgl_contact_sheet(selected_frames, contact_path)
    points3d_probe = probe_points3d_compatibility(dataset_root / "points3D.ply")
    summary = build_intake_summary(
        dataset_root=dataset_root,
        selected_frames=selected_frames,
        camera_config_paths=camera_config_paths,
        points3d_probe=points3d_probe,
        contact_path=contact_path,
        frame_count=imported.frame_count,
    )
    summary_path = args.output_dir / "dxgl_apple_intake_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"dataset_root:       {dataset_root}")
    print(f"frame_count:        {imported.frame_count}")
    print(f"selected_frames:    {selected_indices}")
    print(f"points3D splat ok:  {points3d_probe['splat_compatible']}")
    print(f"wrote:              {contact_path.resolve()}")
    print(f"wrote:              {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
