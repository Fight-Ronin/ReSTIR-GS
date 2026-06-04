from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.eval.ris_ablation import compute_error_metrics
from restir_gs.eval.spatial_mis_ablation import default_spatial_mis_variants, run_spatial_mis_ablation
from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.lighting.deferred import shade_deferred_lambertian
from restir_gs.render.camera_probe import load_camera_config
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.ply_loader import load_gaussian_ply_with_stats
from restir_gs.restir.initial import estimate_ris_initial_diffuse
from restir_gs.restir.proposal import compute_geometric_proposal_distribution, sample_light_candidates_from_distribution


CSV_FIELDS = [
    "variant",
    "center_floor",
    "normal_threshold",
    "depth_tolerance",
    "rgb_threshold",
    "normal_penalty",
    "depth_penalty",
    "rgb_penalty",
    "reuse_fraction",
    "accepted_neighbor_count_mean",
    "center_weight_mean",
    "neighbor_weight_mean",
    "error_delta_mean",
    "improve_fraction",
    "harm_fraction",
    "mae",
    "rmse",
    "bias_r",
    "bias_g",
    "bias_b",
    "mean_abs_bias",
]


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


def mean_abs_rgb_error(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return (estimate - reference).abs().mean(dim=-1)


def write_csv(path: Path, rows: list[dict[str, int | float | str | None]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def finite_rows(rows: list[dict[str, int | float | str | None]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run defensive spatial MIS reuse variants on a real 3DGS PLY.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--candidate-seed", type=int, default=14100)
    parser.add_argument("--initial-selection-seed", type=int, default=14200)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians
    loaded = load_gaussian_ply_with_stats(args.ply, device=device, max_gaussians=max_gaussians)
    camera = load_camera_config(args.camera_config, device=device)
    camera_config = json.loads(args.camera_config.read_text(encoding="utf-8"))

    render_buffers = render_rgbd(loaded.scene, camera)
    gbuffer = make_pseudo_gbuffer(render_buffers, camera)
    valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
    if valid_pixels <= 0:
        raise RuntimeError("Real asset selected camera produced zero valid lighting pixels.")

    lights, light_info = make_asset_scaled_point_lights(gbuffer, count=args.num_lights, seed=2027, device=device)
    reference = shade_deferred_lambertian(gbuffer, lights)
    proposal = compute_geometric_proposal_distribution(gbuffer, lights)
    samples = sample_light_candidates_from_distribution(
        proposal,
        args.candidate_count,
        seed=args.candidate_seed,
        device=device,
    )
    initial_ris, _ = estimate_ris_initial_diffuse(
        gbuffer,
        lights,
        samples.light_indices,
        selection_seed=args.initial_selection_seed,
        proposal_probs=samples.proposal_probs,
    )

    result = run_spatial_mis_ablation(
        gbuffer,
        lights,
        reference,
        initial_ris,
        proposal,
        samples,
        variants=default_spatial_mis_variants(),
    )
    if not finite_rows(result.rows):
        raise RuntimeError(f"Spatial MIS ablation produced non-finite rows: {result.rows}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "spatial_mis_ablation.csv"
    json_path = args.output_dir / "spatial_mis_ablation_summary.json"
    write_csv(csv_path, result.rows)

    valid = reference.valid_mask
    best_error = mean_abs_rgb_error(result.best_buffers.diffuse_rgb, reference.diffuse_rgb)
    initial_error = mean_abs_rgb_error(initial_ris.diffuse_rgb, reference.diffuse_rgb)
    imageio.imwrite(args.output_dir / "spatial_mis_best_composite.png", to_u8_rgb(result.best_buffers.composite_rgb))
    imageio.imwrite(args.output_dir / "spatial_mis_best_abs_error.png", to_u8_scalar(best_error, valid))
    imageio.imwrite(args.output_dir / "spatial_mis_initial_abs_error.png", to_u8_scalar(initial_error, valid))

    initial_diffuse_metrics = compute_error_metrics(initial_ris.diffuse_rgb, reference.diffuse_rgb, valid)
    initial_composite_metrics = compute_error_metrics(initial_ris.composite_rgb, reference.composite_rgb, valid)
    payload = {
        "metadata": {
            "scene": {
                "path": loaded.stats.path,
                "original_count": loaded.stats.original_count,
                "loaded_count": loaded.stats.loaded_count,
                "color_source": loaded.stats.color_source,
                "max_gaussians": max_gaussians,
            },
            "camera_config": {
                "path": str(args.camera_config),
                "candidate_index": camera_config.get("candidate_index"),
                "score": camera_config.get("score"),
                "camera_info": camera_config.get("camera_info"),
                "width": camera.width,
                "height": camera.height,
            },
            "lighting": {
                "valid_pixels": valid_pixels,
                "light_count": args.num_lights,
                "light_info": light_info,
            },
            "candidate_count": args.candidate_count,
            "candidate_seed": args.candidate_seed,
            "initial_selection_seed": args.initial_selection_seed,
            "row_count": len(result.rows),
        },
        "initial_metrics": {
            "diffuse_rgb": initial_diffuse_metrics,
            "composite_rgb": initial_composite_metrics,
        },
        "variants": result.rows,
        "best_variant": result.best_row,
        "any_variant_improved_over_initial": any(float(row["mae"]) < float(initial_diffuse_metrics["mae"]) for row in result.rows),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("variant, MAE, RMSE, reuse, center_w, neighbor_w, improve, harm")
    for row in result.rows:
        print(
            f"{row['variant']}, "
            f"MAE={float(row['mae']):.8f}, RMSE={float(row['rmse']):.8f}, "
            f"reuse={float(row['reuse_fraction']):.4f}, "
            f"center_w={float(row['center_weight_mean']):.4f}, "
            f"neighbor_w={float(row['neighbor_weight_mean']):.4f}, "
            f"improve={float(row['improve_fraction']):.4f}, "
            f"harm={float(row['harm_fraction']):.4f}"
        )
    print(f"initial diffuse MAE: {float(initial_diffuse_metrics['mae']):.8f}")
    print(f"best MIS variant:    {result.best_row['variant']} ({float(result.best_row['mae']):.8f})")
    print(f"wrote:               {csv_path.resolve()}")
    print(f"wrote:               {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
