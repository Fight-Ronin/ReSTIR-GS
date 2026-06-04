from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.eval.proposal_ablation import run_proposal_ablation
from restir_gs.eval.real_asset_benchmark import (
    load_benchmark_manifest,
    normalize_benchmark_row,
    normalize_spatial_mis_row,
    select_top_candidate_indices,
    summarize_benchmark_rows,
)
from restir_gs.eval.spatial_mis_ablation import run_spatial_mis_ablation
from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.lighting.deferred import shade_deferred_lambertian
from restir_gs.render.camera_probe import (
    camera_config_payload,
    make_probe_camera_candidates,
    parse_float_list,
    score_render_buffers,
)
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.ply_loader import load_gaussian_ply_with_stats
from restir_gs.restir.initial import estimate_ris_initial_diffuse
from restir_gs.restir.proposal import compute_geometric_proposal_distribution, sample_light_candidates_from_distribution


COMMON_FIELDS = [
    "scene_id",
    "view_id",
    "method_family",
    "camera_score",
    "valid_pixels",
    "reference_quantity",
    "proposal",
    "estimator",
    "k",
    "variant",
]


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(COMMON_FIELDS)
    extra_fields = sorted({key for row in rows for key in row if key not in fields})
    fields.extend(extra_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rows_are_finite(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        for value in row.values():
            if isinstance(value, int | float) and not math.isfinite(float(value)):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a multi-scene real-asset benchmark over selected camera views.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/real_asset_benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--yaw-values", default="-30,-15,0,15,30")
    parser.add_argument("--pitch-values", default="-10,0,10")
    parser.add_argument("--radius-scales", default="0.9,1.1,1.3")
    parser.add_argument("--camera-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    manifest = load_benchmark_manifest(args.manifest)
    defaults = manifest.defaults
    device = torch.device(args.device)
    yaw_values = parse_float_list(args.yaw_values)
    pitch_values = parse_float_list(args.pitch_values)
    radius_scales = parse_float_list(args.radius_scales)

    all_rows: list[dict[str, Any]] = []
    scene_summaries: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for scene in manifest.scenes:
        loaded = load_gaussian_ply_with_stats(scene.ply, device=device, max_gaussians=scene.max_gaussians)
        candidates = make_probe_camera_candidates(
            loaded.scene.means,
            yaw_values=yaw_values,
            pitch_values=pitch_values,
            radius_scales=radius_scales,
            width=defaults.width,
            height=defaults.height,
            bbox_percentile=args.camera_bbox_percentile,
        )
        render_results = []
        scores = []
        for candidate in candidates:
            buffers = render_rgbd(loaded.scene, candidate.camera)
            render_results.append(buffers)
            scores.append(score_render_buffers(buffers))

        selected_indices = select_top_candidate_indices(scores, defaults.view_count)
        selected_views: list[dict[str, Any]] = []
        for view_rank, candidate_list_index in enumerate(selected_indices):
            candidate = candidates[candidate_list_index]
            score = scores[candidate_list_index]
            buffers = render_results[candidate_list_index]
            view_id = f"view_{view_rank:02d}"
            view_dir = args.output_dir / scene.scene_id / view_id
            view_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(view_dir / "preview_rgb.png", to_u8_rgb(buffers.rgb))

            camera_payload = camera_config_payload(
                candidate.camera,
                candidate.info,
                score=score,
                candidate_index=candidate.index,
            )
            camera_path = view_dir / "camera.json"
            camera_path.write_text(json.dumps(camera_payload, indent=2), encoding="utf-8")

            gbuffer = make_pseudo_gbuffer(buffers, candidate.camera)
            valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
            if valid_pixels <= 0:
                raise RuntimeError(f"Selected view {scene.scene_id}/{view_id} produced zero valid lighting pixels.")

            lights, _ = make_asset_scaled_point_lights(gbuffer, count=defaults.num_lights, seed=2027, device=device)
            proposal_rows = run_proposal_ablation(
                gbuffer,
                lights,
                defaults.k_values,
                defaults.seed_count,
                candidate_seed_base=defaults.candidate_seed_base,
                selection_seed_base=defaults.selection_seed_base,
            )
            all_rows.extend(
                normalize_benchmark_row(
                    row,
                    scene_id=scene.scene_id,
                    view_id=view_id,
                    camera_score=score.score,
                    valid_pixels=valid_pixels,
                    method_family="proposal_ablation",
                )
                for row in proposal_rows
            )

            reference = shade_deferred_lambertian(gbuffer, lights)
            proposal_distribution = compute_geometric_proposal_distribution(gbuffer, lights)
            samples = sample_light_candidates_from_distribution(
                proposal_distribution,
                defaults.spatial_candidate_count,
                seed=defaults.spatial_candidate_seed,
                device=device,
            )
            initial_ris, _ = estimate_ris_initial_diffuse(
                gbuffer,
                lights,
                samples.light_indices,
                selection_seed=defaults.spatial_initial_selection_seed,
                proposal_probs=samples.proposal_probs,
            )
            spatial_result = run_spatial_mis_ablation(
                gbuffer,
                lights,
                reference,
                initial_ris,
                proposal_distribution,
                samples,
            )
            all_rows.extend(
                normalize_spatial_mis_row(
                    row,
                    scene_id=scene.scene_id,
                    view_id=view_id,
                    camera_score=score.score,
                    valid_pixels=valid_pixels,
                    k=defaults.spatial_candidate_count,
                    candidate_seed=defaults.spatial_candidate_seed,
                    selection_seed=defaults.spatial_initial_selection_seed,
                )
                for row in spatial_result.rows
            )

            selected_views.append(
                {
                    "view_id": view_id,
                    "candidate_index": candidate.index,
                    "camera_score": score.score,
                    "valid_pixels": valid_pixels,
                    "camera_path": str(camera_path),
                    "preview_path": str(view_dir / "preview_rgb.png"),
                }
            )

        scene_summaries.append(
            {
                "scene_id": scene.scene_id,
                "ply": str(scene.ply),
                "loaded_count": loaded.stats.loaded_count,
                "original_count": loaded.stats.original_count,
                "selected_views": selected_views,
            }
        )

    if not all_rows:
        raise RuntimeError("Benchmark produced no rows.")
    if not rows_are_finite(all_rows):
        raise RuntimeError("Benchmark produced non-finite numeric metrics.")

    csv_path = args.output_dir / "real_asset_benchmark_rows.csv"
    json_path = args.output_dir / "real_asset_benchmark_summary.json"
    write_csv(csv_path, all_rows)
    payload = {
        "metadata": {
            "manifest": str(args.manifest),
            "scene_count": len(manifest.scenes),
            "row_count": len(all_rows),
            "defaults": {
                "view_count": defaults.view_count,
                "width": defaults.width,
                "height": defaults.height,
                "num_lights": defaults.num_lights,
                "k_values": defaults.k_values,
                "seed_count": defaults.seed_count,
                "spatial_candidate_count": defaults.spatial_candidate_count,
            },
        },
        "scenes": scene_summaries,
        "summary": summarize_benchmark_rows(all_rows),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"manifest: {args.manifest}")
    print(f"scenes:   {len(manifest.scenes)}")
    print(f"rows:     {len(all_rows)}")
    print(f"wrote:    {csv_path.resolve()}")
    print(f"wrote:    {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
