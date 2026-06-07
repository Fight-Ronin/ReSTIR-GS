from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

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
)
from restir_gs.render.dxgl_asset import scale_camera
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.restir.initial import sample_uniform_light_candidates
from restir_gs.restir.proposal import CandidateSamples, compute_geometric_proposal_distribution, sample_light_candidates_from_distribution
from restir_gs.restir.visibility import estimate_visibility_proposal_lighting, estimate_visibility_ris_initial_lighting


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_visibility_ris")


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


def make_contact_sheet(images: dict[str, np.ndarray], output_path: Path) -> None:
    labels = list(images)
    cell_w = 180
    cell_h = 150
    sheet = Image.new("RGB", (len(labels) * cell_w, cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(labels):
        x = col * cell_w
        draw.text((x + 8, 8), label, fill=(0, 0, 0))
        image = Image.fromarray(images[label]).convert("RGB")
        image.thumbnail((cell_w - 16, cell_h - 34))
        sheet.paste(image, (x + (cell_w - image.width) // 2, 28 + (cell_h - 34 - image.height) // 2))
        draw.rectangle((x + 4, 24, x + cell_w - 4, cell_h - 4), outline=(180, 180, 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def make_row(
    proposal: str,
    estimator: str,
    k: int,
    candidate_seed: int,
    selection_seed: int,
    reference_quantity: str,
    estimate: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, int | float | str]:
    row: dict[str, int | float | str] = {
        "proposal": proposal,
        "estimator": estimator,
        "k": k,
        "candidate_seed": candidate_seed,
        "selection_seed": selection_seed,
        "reference_quantity": reference_quantity,
    }
    row.update(compute_rgb_error_metrics(estimate, reference, valid_mask))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run visibility-aware initial RIS smoke on one aligned asset frame.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-id", default="dxgl_apple")
    parser.add_argument("--frame-index", type=int, default=49)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-lights", type=int, default=16)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--candidate-seed", type=int, default=21100)
    parser.add_argument("--selection-seed", type=int, default=22100)
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
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Expected positive render size, got {args.width}x{args.height}")
    if args.num_lights <= 0 or args.candidate_count <= 0:
        raise ValueError("Expected positive light and candidate counts.")

    manifest = load_aligned_asset_manifest(args.manifest)
    spec = get_aligned_asset_spec(manifest, args.asset_id)
    resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)
    device = torch.device(args.device)
    asset = load_registered_aligned_asset(resolved, device=device, max_gaussians_override=args.max_gaussians)
    if args.frame_index < 0 or args.frame_index >= asset.transforms.frame_count:
        raise ValueError(f"Frame index {args.frame_index} exceeds frame count {asset.transforms.frame_count}")

    world_lights, light_info = make_asset_scaled_world_lights(
        asset.loaded.scene.means,
        count=args.num_lights,
        seed=args.light_seed,
        device=device,
    )
    target_world = torch.tensor(light_info["center"], dtype=torch.float32, device=device)
    scene_radius = float(light_info["radius"])
    shadow_bundle = make_shadow_map_bundle(
        asset.loaded.scene,
        world_lights.positions_world,
        torch.arange(args.num_lights, dtype=torch.long, device=device),
        target_world,
        scene_radius=scene_radius,
        resolution=args.shadow_resolution,
        shadow_bias_scale=args.shadow_bias_scale,
    )

    camera = scale_camera(asset.transforms.frames[args.frame_index].camera, args.width, args.height)
    render_buffers = render_rgbd(asset.loaded.scene, camera)
    gbuffer = make_pseudo_gbuffer(render_buffers, camera)
    lights = world_lights_to_camera_lights(world_lights, camera)
    unshadowed_reference = shade_deferred_lambertian(gbuffer, lights, ambient=args.ambient)
    shadowed_reference = shade_deferred_lambertian_visible(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        ambient=args.ambient,
        alpha_threshold=args.shadow_alpha_threshold,
    )

    height, width = gbuffer.rgb.shape[:2]
    uniform_indices = sample_uniform_light_candidates(
        height,
        width,
        args.candidate_count,
        args.num_lights,
        seed=args.candidate_seed,
        device=device,
    )
    uniform_samples = CandidateSamples(
        light_indices=uniform_indices,
        proposal_probs=torch.full(
            uniform_indices.shape,
            1.0 / float(args.num_lights),
            dtype=gbuffer.rgb.dtype,
            device=device,
        ),
    )
    geometric_distribution = compute_geometric_proposal_distribution(gbuffer, lights)
    geometric_samples = sample_light_candidates_from_distribution(
        geometric_distribution,
        args.candidate_count,
        seed=args.candidate_seed,
        device=device,
    )

    uniform_mc = estimate_visibility_proposal_lighting(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        uniform_samples,
        ambient=args.ambient,
        alpha_threshold=args.shadow_alpha_threshold,
    )
    geometric_mc = estimate_visibility_proposal_lighting(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        geometric_samples,
        ambient=args.ambient,
        alpha_threshold=args.shadow_alpha_threshold,
    )
    uniform_ris, uniform_reservoir = estimate_visibility_ris_initial_lighting(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        uniform_indices,
        selection_seed=args.selection_seed,
        ambient=args.ambient,
        alpha_threshold=args.shadow_alpha_threshold,
    )
    geometric_ris, geometric_reservoir = estimate_visibility_ris_initial_lighting(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        geometric_samples.light_indices,
        proposal_probs=geometric_samples.proposal_probs,
        selection_seed=args.selection_seed,
        ambient=args.ambient,
        alpha_threshold=args.shadow_alpha_threshold,
    )

    estimates = {
        ("uniform", "mc"): uniform_mc,
        ("geometric", "mc"): geometric_mc,
        ("uniform", "ris"): uniform_ris,
        ("geometric", "ris"): geometric_ris,
    }
    rows: list[dict[str, int | float | str]] = []
    for (proposal, estimator), estimate in estimates.items():
        rows.append(
            make_row(
                proposal,
                estimator,
                args.candidate_count,
                args.candidate_seed,
                args.selection_seed,
                "visible_contribution_rgb",
                estimate.contribution_rgb,
                shadowed_reference.diffuse_rgb,
                shadowed_reference.valid_mask,
            )
        )
        rows.append(
            make_row(
                proposal,
                estimator,
                args.candidate_count,
                args.candidate_seed,
                args.selection_seed,
                "visible_composite_rgb",
                estimate.composite_rgb,
                shadowed_reference.composite_rgb,
                shadowed_reference.valid_mask,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": args.output_dir / "visibility_ris_rows.csv",
        "summary": args.output_dir / "visibility_ris_summary.json",
        "unshadowed_reference": args.output_dir / "visibility_ris_unshadowed_reference.png",
        "shadowed_reference": args.output_dir / "visibility_ris_shadowed_reference.png",
        "uniform_mc": args.output_dir / "visibility_ris_uniform_mc.png",
        "geometric_mc": args.output_dir / "visibility_ris_geometric_mc.png",
        "uniform_ris": args.output_dir / "visibility_ris_uniform_ris.png",
        "geometric_ris": args.output_dir / "visibility_ris_geometric_ris.png",
        "geometric_ris_abs_error": args.output_dir / "visibility_ris_geometric_ris_abs_error.png",
        "contact": args.output_dir / "visibility_ris_contact.png",
    }

    with paths["rows"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    error = torch.abs(geometric_ris.contribution_rgb - shadowed_reference.diffuse_rgb).mean(dim=-1)
    images = {
        "shadowed ref": to_u8_rgb(shadowed_reference.composite_rgb),
        "uniform MC": to_u8_rgb(uniform_mc.composite_rgb),
        "geometric MC": to_u8_rgb(geometric_mc.composite_rgb),
        "uniform RIS": to_u8_rgb(uniform_ris.composite_rgb),
        "geometric RIS": to_u8_rgb(geometric_ris.composite_rgb),
        "geom RIS error": to_u8_scalar(error, shadowed_reference.valid_mask),
    }
    imageio.imwrite(paths["unshadowed_reference"], to_u8_rgb(unshadowed_reference.composite_rgb))
    imageio.imwrite(paths["shadowed_reference"], images["shadowed ref"])
    imageio.imwrite(paths["uniform_mc"], images["uniform MC"])
    imageio.imwrite(paths["geometric_mc"], images["geometric MC"])
    imageio.imwrite(paths["uniform_ris"], images["uniform RIS"])
    imageio.imwrite(paths["geometric_ris"], images["geometric RIS"])
    imageio.imwrite(paths["geometric_ris_abs_error"], images["geom RIS error"])
    make_contact_sheet(images, paths["contact"])

    finite = all(_row_finite(row) for row in rows)
    reference_delta = compute_rgb_error_metrics(
        shadowed_reference.composite_rgb,
        unshadowed_reference.composite_rgb,
        shadowed_reference.valid_mask,
    )
    summary = {
        "version": 1,
        "asset_id": args.asset_id,
        "frame_index": args.frame_index,
        "render": {"width": args.width, "height": args.height, "device": str(device)},
        "shadow": {
            "num_lights": args.num_lights,
            "shadow_resolution": args.shadow_resolution,
            "shadow_bias_scale": args.shadow_bias_scale,
            "depth_bias": shadow_bundle.depth_bias,
            "alpha_threshold": args.shadow_alpha_threshold,
        },
        "candidate_count": args.candidate_count,
        "candidate_seed": args.candidate_seed,
        "selection_seed": args.selection_seed,
        "valid_pixels": int(shadowed_reference.valid_mask.sum().detach().cpu()),
        "uniform_ris_valid_pixels": int(uniform_reservoir.valid_mask.sum().detach().cpu()),
        "geometric_ris_valid_pixels": int(geometric_reservoir.valid_mask.sum().detach().cpu()),
        "shadowed_vs_unshadowed": reference_delta,
        "rows": rows,
        "all_numeric_finite": finite,
        "light_info": light_info,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"asset:       {args.asset_id}")
    print(f"frame:       {args.frame_index}")
    print(f"valid:       {summary['valid_pixels']}")
    print(f"rows:        {len(rows)}")
    print(f"finite:      {finite}")
    print(f"geom RIS MAE:{_find_mae(rows, 'geometric', 'ris', 'visible_contribution_rgb'):.6f}")
    print(f"wrote:       {paths['summary'].resolve()}")
    return 0


def _row_finite(row: dict[str, int | float | str]) -> bool:
    for value in row.values():
        if isinstance(value, float) and not np.isfinite(value):
            return False
    return True


def _find_mae(rows: list[dict[str, int | float | str]], proposal: str, estimator: str, quantity: str) -> float:
    for row in rows:
        if row["proposal"] == proposal and row["estimator"] == estimator and row["reference_quantity"] == quantity:
            return float(row["mae"])
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
