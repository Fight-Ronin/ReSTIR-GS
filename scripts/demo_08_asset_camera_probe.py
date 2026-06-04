from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.render.camera_probe import (
    camera_config_payload,
    make_probe_camera_candidates,
    parse_float_list,
    score_render_buffers,
    score_to_dict,
    select_best_candidate,
)
from restir_gs.render.gsplat_renderer import RenderBuffers, render_rgbd
from restir_gs.render.ply_loader import load_gaussian_ply_with_stats


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def to_u8_scalar(values: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    data = values.detach().cpu()
    mask = valid.detach().cpu().to(torch.bool)
    out = torch.zeros_like(data, dtype=torch.float32)
    if bool(mask.any()):
        selected = data[mask]
        lo = selected.min()
        hi = selected.max()
        denom = hi - lo if float(hi - lo) > 1e-8 else torch.tensor(1.0)
        out[mask] = (selected - lo) / denom
    return (out.numpy() * 255.0).astype(np.uint8)


def make_contact_sheet(images: list[np.ndarray], columns: int) -> np.ndarray:
    if not images:
        raise ValueError("Expected at least one image for contact sheet.")
    height, width, channels = images[0].shape
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * height, columns * width, channels), dtype=np.uint8)
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        sheet[row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe stable camera views for a real 3DGS PLY asset.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    parser.add_argument("--camera-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--yaw-values", default="-30,-15,0,15,30")
    parser.add_argument("--pitch-values", default="-10,0,10")
    parser.add_argument("--radius-scales", default="0.9,1.1,1.3")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    yaw_values = parse_float_list(args.yaw_values)
    pitch_values = parse_float_list(args.pitch_values)
    radius_scales = parse_float_list(args.radius_scales)
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians

    loaded = load_gaussian_ply_with_stats(args.ply, device=device, max_gaussians=max_gaussians)
    candidates = make_probe_camera_candidates(
        loaded.scene.means,
        yaw_values=yaw_values,
        pitch_values=pitch_values,
        radius_scales=radius_scales,
        width=args.width,
        height=args.height,
        bbox_percentile=args.camera_bbox_percentile,
    )

    preview_images: list[np.ndarray] = []
    render_results: list[RenderBuffers] = []
    scores = []
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        buffers = render_rgbd(loaded.scene, candidate.camera)
        score = score_render_buffers(buffers)
        preview_images.append(to_u8_rgb(buffers.rgb))
        render_results.append(buffers)
        scores.append(score)
        rows.append(
            {
                "index": candidate.index,
                "yaw_degrees": candidate.info.yaw_degrees,
                "pitch_degrees": candidate.info.pitch_degrees,
                "radius_scale": candidate.info.radius_scale,
                "camera": {
                    "target": candidate.info.target,
                    "eye": candidate.info.eye,
                    "radius": candidate.info.radius,
                    "bbox_percentile": candidate.info.bbox_percentile,
                },
                "score": score_to_dict(score),
            }
        )

    selected_list_index = select_best_candidate(scores)
    selected_candidate = candidates[selected_list_index]
    selected_score = scores[selected_list_index]
    selected_buffers = render_results[selected_list_index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output_dir / "camera_probe_contact.png", make_contact_sheet(preview_images, columns=len(yaw_values)))
    imageio.imwrite(args.output_dir / "camera_probe_selected_rgb.png", to_u8_rgb(selected_buffers.rgb))
    imageio.imwrite(
        args.output_dir / "camera_probe_selected_depth.png",
        to_u8_scalar(selected_buffers.depth, selected_buffers.alpha > 1e-4),
    )
    imageio.imwrite(args.output_dir / "camera_probe_selected_alpha.png", to_u8_scalar(selected_buffers.alpha, selected_buffers.alpha > 0.0))

    selected_payload = camera_config_payload(
        selected_candidate.camera,
        selected_candidate.info,
        score=selected_score,
        candidate_index=selected_candidate.index,
    )
    camera_config_path = args.output_dir / "camera_probe_selected_camera.json"
    camera_config_path.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    summary = {
        "scene": {
            "path": loaded.stats.path,
            "original_count": loaded.stats.original_count,
            "loaded_count": loaded.stats.loaded_count,
            "color_source": loaded.stats.color_source,
            "max_gaussians": max_gaussians,
        },
        "probe": {
            "candidate_count": len(candidates),
            "yaw_values": yaw_values,
            "pitch_values": pitch_values,
            "radius_scales": radius_scales,
            "bbox_percentile": args.camera_bbox_percentile,
            "width": args.width,
            "height": args.height,
            "selected_index": selected_candidate.index,
            "selected_score": score_to_dict(selected_score),
        },
        "candidates": rows,
    }
    summary_path = args.output_dir / "camera_probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"ply:             {args.ply}")
    print(f"gaussians:       {loaded.stats.loaded_count} / {loaded.stats.original_count}")
    print(f"candidates:      {len(candidates)}")
    print(f"selected index:  {selected_candidate.index}")
    print(f"selected score:  {selected_score.score:.6f}")
    print(f"valid pixels:    {selected_score.valid_pixels}")
    print(f"yaw/pitch/r:     {selected_candidate.info.yaw_degrees}, {selected_candidate.info.pitch_degrees}, {selected_candidate.info.radius_scale}")
    print(f"camera config:   {camera_config_path.resolve()}")
    print(f"summary:         {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
