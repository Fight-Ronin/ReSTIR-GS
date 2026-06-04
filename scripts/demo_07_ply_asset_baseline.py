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

from restir_gs.eval.ris_ablation import compute_error_metrics
from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.lighting.deferred import shade_deferred_lambertian
from restir_gs.render.camera_probe import load_camera_config
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.ply_loader import load_gaussian_ply_with_stats, make_asset_camera
from restir_gs.restir.initial import estimate_proposal_diffuse, estimate_ris_initial_diffuse
from restir_gs.restir.proposal import compute_geometric_proposal_distribution, sample_light_candidates_from_distribution


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


def to_u8_vector(values: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    data = values.detach().cpu()
    mask = valid.detach().cpu().to(torch.bool)
    out = torch.zeros_like(data, dtype=torch.float32)
    if bool(mask.any()):
        selected = data[mask]
        lo = selected.min(dim=0).values
        hi = selected.max(dim=0).values
        denom = torch.where((hi - lo) > 1e-8, hi - lo, torch.ones_like(hi))
        out[mask] = (selected - lo) / denom
    return (out.numpy() * 255.0).astype(np.uint8)


def to_u8_normal(normal: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    data = (normal.detach().cpu() * 0.5) + 0.5
    mask = valid.detach().cpu().to(torch.bool)
    out = torch.zeros_like(data, dtype=torch.float32)
    out[mask] = data[mask].clamp(0.0, 1.0)
    return (out.numpy() * 255.0).astype(np.uint8)


def metric_payload(name: str, estimate: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor) -> dict[str, float | str]:
    payload: dict[str, float | str] = {"name": name}
    payload.update(compute_error_metrics(estimate, reference, valid))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real 3DGS PLY single-frame baseline.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    parser.add_argument("--camera-radius-scale", type=float, default=1.4)
    parser.add_argument("--camera-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians
    loaded = load_gaussian_ply_with_stats(args.ply, device=device, max_gaussians=max_gaussians)
    if args.camera_config is None:
        camera, camera_info = make_asset_camera(
            loaded.scene.means,
            width=args.width,
            height=args.height,
            radius_scale=args.camera_radius_scale,
            bbox_percentile=args.camera_bbox_percentile,
        )
        camera_source = "auto_robust_bbox"
        camera_payload = {
            "target": camera_info.target,
            "eye": camera_info.eye,
            "bbox_min": camera_info.bbox_min,
            "bbox_max": camera_info.bbox_max,
            "bbox_diagonal": camera_info.bbox_diagonal,
            "radius": camera_info.radius,
            "focal": camera_info.focal,
            "bbox_percentile": camera_info.bbox_percentile,
            "radius_scale": camera_info.radius_scale,
            "yaw_degrees": camera_info.yaw_degrees,
            "pitch_degrees": camera_info.pitch_degrees,
            "width": camera.width,
            "height": camera.height,
        }
    else:
        camera = load_camera_config(args.camera_config, device=device)
        camera_config_data = json.loads(args.camera_config.read_text(encoding="utf-8"))
        camera_source = "probe_config"
        camera_payload = {
            "config_path": str(args.camera_config),
            "selected_candidate_index": camera_config_data.get("candidate_index"),
            "score": camera_config_data.get("score"),
            "camera_info": camera_config_data.get("camera_info"),
            "width": camera.width,
            "height": camera.height,
        }
    render_buffers = render_rgbd(loaded.scene, camera)
    gbuffer = make_pseudo_gbuffer(render_buffers, camera)
    valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
    if valid_pixels <= 0:
        raise RuntimeError("PLY asset produced zero valid G-buffer pixels with the auto camera.")

    lights, light_info = make_asset_scaled_point_lights(gbuffer, count=args.num_lights, seed=2027, device=device)
    reference = shade_deferred_lambertian(gbuffer, lights)
    proposal = compute_geometric_proposal_distribution(gbuffer, lights)
    samples = sample_light_candidates_from_distribution(proposal, args.candidate_count, seed=8100, device=device)
    geometric_mc = estimate_proposal_diffuse(gbuffer, lights, samples)
    geometric_ris, _ = estimate_ris_initial_diffuse(
        gbuffer,
        lights,
        samples.light_indices,
        selection_seed=8200,
        proposal_probs=samples.proposal_probs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output_dir / "ply_rgb.png", to_u8_rgb(gbuffer.rgb))
    imageio.imwrite(args.output_dir / "ply_depth.png", to_u8_scalar(gbuffer.depth, gbuffer.valid_mask))
    imageio.imwrite(args.output_dir / "ply_alpha.png", to_u8_scalar(gbuffer.alpha, gbuffer.alpha > 0.0))
    imageio.imwrite(args.output_dir / "ply_normal.png", to_u8_normal(gbuffer.normal_cam, reference.valid_mask))
    imageio.imwrite(args.output_dir / "ply_deferred_composite.png", to_u8_rgb(reference.composite_rgb))
    imageio.imwrite(args.output_dir / "ply_geometric_mc_composite.png", to_u8_rgb(geometric_mc.composite_rgb))
    imageio.imwrite(args.output_dir / "ply_geometric_ris_composite.png", to_u8_rgb(geometric_ris.composite_rgb))

    metrics = [
        metric_payload("geometric_mc_diffuse", geometric_mc.diffuse_rgb, reference.diffuse_rgb, reference.valid_mask),
        metric_payload("geometric_mc_composite", geometric_mc.composite_rgb, reference.composite_rgb, reference.valid_mask),
        metric_payload("geometric_ris_diffuse", geometric_ris.diffuse_rgb, reference.diffuse_rgb, reference.valid_mask),
        metric_payload("geometric_ris_composite", geometric_ris.composite_rgb, reference.composite_rgb, reference.valid_mask),
    ]
    payload = {
        "scene": {
            "path": loaded.stats.path,
            "original_count": loaded.stats.original_count,
            "loaded_count": loaded.stats.loaded_count,
            "color_source": loaded.stats.color_source,
            "max_gaussians": max_gaussians,
        },
        "camera_source": camera_source,
        "camera": camera_payload,
        "lights": light_info,
        "render": {
            "valid_pixels": valid_pixels,
            "light_count": args.num_lights,
            "candidate_count": args.candidate_count,
        },
        "metrics": metrics,
    }
    summary_path = args.output_dir / "ply_asset_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"ply:            {args.ply}")
    print(f"gaussians:      {loaded.stats.loaded_count} / {loaded.stats.original_count}")
    print(f"color source:   {loaded.stats.color_source}")
    print(f"valid pixels:   {valid_pixels}")
    for metric in metrics:
        print(f"{metric['name']}: MAE={float(metric['mae']):.8f}, RMSE={float(metric['rmse']):.8f}")
    print(f"wrote: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
