from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.lighting.asset_lights import make_asset_scaled_world_lights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import shade_deferred_lambertian
from restir_gs.lighting.visibility import make_shadow_map_bundle, shade_deferred_lambertian_visible
from restir_gs.metrics import compute_rgb_error_metrics
from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
    resolve_requested_asset_ids,
)
from restir_gs.render.dxgl_asset import scale_camera
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.restir.initial import sample_uniform_light_candidates
from restir_gs.restir.proposal import CandidateSamples, compute_geometric_proposal_distribution, sample_light_candidates_from_distribution
from restir_gs.restir.visibility import estimate_visibility_proposal_lighting, estimate_visibility_ris_initial_lighting


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_visibility_matrix")


def parse_asset_ids(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"Expected at least one asset id, got {text!r}")
    return values


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def to_u8_scalar(values: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
    data = values.detach().cpu().float()
    valid = torch.isfinite(data)
    if mask is not None:
        valid = valid & mask.detach().cpu().to(torch.bool)
    out = torch.zeros_like(data)
    if bool(valid.any()):
        selected = data[valid]
        lo = selected.min()
        hi = selected.max()
        denom = hi - lo if float(hi - lo) > 1e-8 else torch.tensor(1.0)
        out[valid] = (selected - lo) / denom
    return (out.numpy() * 255.0).astype(np.uint8)


def make_contact_sheet(records: list[dict[str, Any]], output_path: Path) -> None:
    cell_w = 160
    cell_h = 132
    label_w = 190
    header_h = 34
    labels = ["Shadowed Ref", "Geom MC", "Geom RIS", "Geom RIS Error"]
    sheet = Image.new("RGB", (label_w + len(labels) * cell_w, header_h + max(len(records), 1) * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), "Visibility target smoke matrix", fill=(0, 0, 0))
    for row, record in enumerate(records):
        y = header_h + row * cell_h
        draw.text(
            (8, y + 8),
            f"{record['asset_id']}\nframe {record['frame_index']}\n"
            f"delta={record['shadow_delta_mae']:.4f}\n"
            f"geom={record['geom_ris_mae']:.4f}",
            fill=(0, 0, 0),
        )
        for col, label in enumerate(labels):
            x = label_w + col * cell_w
            draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
            image = record["images"][label].copy().convert("RGB")
            image.thumbnail((146, 96))
            sheet.paste(image, (x + 6 + (146 - image.width) // 2, y + 24 + (96 - image.height) // 2))
            draw.rectangle((x + 6, y + 24, x + 152, y + 120), outline=(180, 180, 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def make_row(
    asset_id: str,
    proposal: str,
    estimator: str,
    k: int,
    frame_index: int,
    candidate_seed: int,
    selection_seed: int,
    quantity: str,
    estimate: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, int | float | str]:
    row: dict[str, int | float | str] = {
        "asset_id": asset_id,
        "frame_index": frame_index,
        "proposal": proposal,
        "estimator": estimator,
        "k": k,
        "candidate_seed": candidate_seed,
        "selection_seed": selection_seed,
        "reference_quantity": quantity,
        "target_mode": "visibility",
    }
    row.update(compute_rgb_error_metrics(estimate, reference, valid_mask))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed visibility-aware MC/RIS smoke across aligned assets.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-ids", type=parse_asset_ids, default=None)
    parser.add_argument("--asset-set", default="testing")
    parser.add_argument("--frame-index", type=int, default=49)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--num-lights", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--candidate-seed", type=int, default=23100)
    parser.add_argument("--selection-seed", type=int, default=24100)
    parser.add_argument("--shadow-resolution", type=int, default=128)
    parser.add_argument("--shadow-bias-scale", type=float, default=0.02)
    parser.add_argument("--shadow-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--max-gaussians", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    manifest = load_aligned_asset_manifest(args.manifest)
    asset_ids = resolve_requested_asset_ids(manifest, asset_ids=args.asset_ids, asset_set=args.asset_set)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int | float | str]] = []
    assets: list[dict[str, Any]] = []
    contact_records: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        spec = get_aligned_asset_spec(manifest, asset_id)
        resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)
        asset = load_registered_aligned_asset(resolved, device=device, max_gaussians_override=args.max_gaussians)
        frame_index = args.frame_index
        if frame_index >= asset.transforms.frame_count:
            raise ValueError(f"{asset_id} frame {frame_index} exceeds frame count {asset.transforms.frame_count}")

        world_lights, light_info = make_asset_scaled_world_lights(asset.loaded.scene.means, args.num_lights, seed=args.light_seed, device=device)
        target_world = torch.tensor(light_info["center"], dtype=torch.float32, device=device)
        shadow_bundle = make_shadow_map_bundle(
            asset.loaded.scene,
            world_lights.positions_world,
            torch.arange(args.num_lights, dtype=torch.long, device=device),
            target_world,
            scene_radius=float(light_info["radius"]),
            resolution=args.shadow_resolution,
            shadow_bias_scale=args.shadow_bias_scale,
        )
        camera = scale_camera(asset.transforms.frames[frame_index].camera, args.width, args.height)
        gbuffer = make_pseudo_gbuffer(render_rgbd(asset.loaded.scene, camera), camera)
        lights = world_lights_to_camera_lights(world_lights, camera)
        unshadowed_reference = shade_deferred_lambertian(gbuffer, lights, ambient=args.ambient)
        reference = shade_deferred_lambertian_visible(
            gbuffer,
            camera,
            lights,
            shadow_bundle,
            ambient=args.ambient,
            alpha_threshold=args.shadow_alpha_threshold,
        )

        height, width = gbuffer.depth.shape
        uniform_indices = sample_uniform_light_candidates(height, width, args.candidate_count, args.num_lights, seed=args.candidate_seed, device=device)
        uniform_samples = CandidateSamples(
            light_indices=uniform_indices,
            proposal_probs=torch.full(uniform_indices.shape, 1.0 / float(args.num_lights), dtype=gbuffer.rgb.dtype, device=device),
        )
        geometric_distribution = compute_geometric_proposal_distribution(gbuffer, lights)
        geometric_samples = sample_light_candidates_from_distribution(
            geometric_distribution,
            args.candidate_count,
            seed=args.candidate_seed,
            device=device,
        )
        uniform_mc = estimate_visibility_proposal_lighting(gbuffer, camera, lights, shadow_bundle, uniform_samples, ambient=args.ambient)
        geometric_mc = estimate_visibility_proposal_lighting(gbuffer, camera, lights, shadow_bundle, geometric_samples, ambient=args.ambient)
        uniform_ris, _ = estimate_visibility_ris_initial_lighting(
            gbuffer,
            camera,
            lights,
            shadow_bundle,
            uniform_indices,
            selection_seed=args.selection_seed,
            ambient=args.ambient,
        )
        geometric_ris, _ = estimate_visibility_ris_initial_lighting(
            gbuffer,
            camera,
            lights,
            shadow_bundle,
            geometric_samples.light_indices,
            proposal_probs=geometric_samples.proposal_probs,
            selection_seed=args.selection_seed,
            ambient=args.ambient,
        )

        estimates = {
            ("uniform", "mc"): uniform_mc,
            ("geometric", "mc"): geometric_mc,
            ("uniform", "ris"): uniform_ris,
            ("geometric", "ris"): geometric_ris,
        }
        for (proposal, estimator), estimate in estimates.items():
            rows.append(
                make_row(
                    asset_id,
                    proposal,
                    estimator,
                    args.candidate_count,
                    frame_index,
                    args.candidate_seed,
                    args.selection_seed,
                    "visible_contribution_rgb",
                    estimate.contribution_rgb,
                    reference.diffuse_rgb,
                    reference.valid_mask,
                )
            )
            rows.append(
                make_row(
                    asset_id,
                    proposal,
                    estimator,
                    args.candidate_count,
                    frame_index,
                    args.candidate_seed,
                    args.selection_seed,
                    "visible_composite_rgb",
                    estimate.composite_rgb,
                    reference.composite_rgb,
                    reference.valid_mask,
                )
            )

        shadow_delta = compute_rgb_error_metrics(reference.composite_rgb, unshadowed_reference.composite_rgb, reference.valid_mask)
        shadowed_mean = float(reference.composite_rgb[reference.valid_mask].mean().detach().cpu()) if bool(reference.valid_mask.any()) else 0.0
        geometric_ris_error = torch.abs(geometric_ris.contribution_rgb - reference.diffuse_rgb).mean(dim=-1)
        geom_row = _find_row(rows, asset_id, "geometric", "ris", "visible_contribution_rgb")
        asset_dir = args.output_dir / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        images = {
            "Shadowed Ref": Image.fromarray(to_u8_rgb(reference.composite_rgb)),
            "Geom MC": Image.fromarray(to_u8_rgb(geometric_mc.composite_rgb)),
            "Geom RIS": Image.fromarray(to_u8_rgb(geometric_ris.composite_rgb)),
            "Geom RIS Error": Image.fromarray(to_u8_scalar(geometric_ris_error, reference.valid_mask)).convert("RGB"),
        }
        imageio.imwrite(asset_dir / "shadowed_reference.png", np.asarray(images["Shadowed Ref"]))
        imageio.imwrite(asset_dir / "geometric_ris.png", np.asarray(images["Geom RIS"]))
        imageio.imwrite(asset_dir / "geometric_ris_abs_error.png", np.asarray(images["Geom RIS Error"]))
        contact_records.append(
            {
                "asset_id": asset_id,
                "frame_index": frame_index,
                "shadow_delta_mae": float(shadow_delta["mae"]),
                "geom_ris_mae": float(geom_row["mae"]),
                "images": images,
            }
        )
        assets.append(
            {
                "asset_id": asset_id,
                "frame_index": frame_index,
                "valid_pixels": int(reference.valid_mask.sum().detach().cpu()),
                "shadowed_vs_unshadowed_mae": float(shadow_delta["mae"]),
                "shadowed_mean": shadowed_mean,
                "geometric_ris_visible_mae": float(geom_row["mae"]),
                "light_info": light_info,
            }
        )

    csv_path = args.output_dir / "visibility_smoke_matrix_rows.csv"
    summary_path = args.output_dir / "visibility_smoke_matrix_summary.json"
    contact_path = args.output_dir / "visibility_smoke_matrix_contact.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(contact_records, contact_path)
    finite = all(_row_finite(row) for row in rows)
    summary = {
        "version": 1,
        "manifest": str(args.manifest),
        "asset_ids": asset_ids,
        "row_count": len(rows),
        "all_numeric_finite": finite,
        "target_mode": "visibility",
        "settings": {
            "frame_index": args.frame_index,
            "width": args.width,
            "height": args.height,
            "num_lights": args.num_lights,
            "candidate_count": args.candidate_count,
            "shadow_resolution": args.shadow_resolution,
            "shadow_bias_scale": args.shadow_bias_scale,
        },
        "assets": assets,
        "outputs": {"csv": str(csv_path), "summary": str(summary_path), "contact": str(contact_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"assets:    {asset_ids}")
    print(f"rows:      {len(rows)}")
    print(f"finite:    {finite}")
    print(f"wrote:     {summary_path.resolve()}")
    return 0


def _find_row(rows: list[dict[str, int | float | str]], asset_id: str, proposal: str, estimator: str, quantity: str) -> dict[str, int | float | str]:
    for row in rows:
        if row["asset_id"] == asset_id and row["proposal"] == proposal and row["estimator"] == estimator and row["reference_quantity"] == quantity:
            return row
    raise RuntimeError(f"Missing row for {asset_id} {proposal}/{estimator}/{quantity}")


def _row_finite(row: dict[str, int | float | str]) -> bool:
    for value in row.values():
        if isinstance(value, float) and not np.isfinite(value):
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
