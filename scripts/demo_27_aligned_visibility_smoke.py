from __future__ import annotations

import argparse
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
from restir_gs.lighting.visibility import (
    evaluate_shadow_visibility,
    make_shadow_map_bundle,
    shade_deferred_lambertian_visible,
)
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


DEFAULT_OUTPUT_DIR = Path("outputs/aligned_visibility")


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
    cell_w = 190
    cell_h = 160
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an aligned visibility-aware direct lighting smoke.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-id", default="dxgl_apple")
    parser.add_argument("--frame-index", type=int, default=49)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--num-lights", type=int, default=16)
    parser.add_argument("--light-seed", type=int, default=2027)
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
    if args.num_lights <= 0:
        raise ValueError(f"Expected positive num_lights, got {args.num_lights}")

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
    unshadowed = shade_deferred_lambertian(gbuffer, lights, ambient=args.ambient)
    shadowed = shade_deferred_lambertian_visible(
        gbuffer,
        camera,
        lights,
        shadow_bundle,
        ambient=args.ambient,
        alpha_threshold=args.shadow_alpha_threshold,
    )

    debug_light_index = torch.zeros((*gbuffer.depth.shape, 1), dtype=torch.long, device=device)
    debug_visibility = evaluate_shadow_visibility(
        gbuffer,
        camera,
        shadow_bundle,
        debug_light_index,
        alpha_threshold=args.shadow_alpha_threshold,
    )[..., 0]
    valid = unshadowed.valid_mask
    difference = torch.abs(unshadowed.composite_rgb - shadowed.composite_rgb).mean(dim=-1)
    metrics = compute_rgb_error_metrics(shadowed.composite_rgb, unshadowed.composite_rgb, valid)
    visibility_values = debug_visibility[valid].detach().cpu() if bool(valid.any()) else torch.empty((0,))
    visibility_mean = float(visibility_values.float().mean()) if visibility_values.numel() else 0.0
    visibility_min = float(visibility_values.min()) if visibility_values.numel() else 0.0
    visibility_max = float(visibility_values.max()) if visibility_values.numel() else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "unshadowed": args.output_dir / "visibility_unshadowed_reference.png",
        "shadowed": args.output_dir / "visibility_shadowed_reference.png",
        "difference": args.output_dir / "visibility_difference.png",
        "debug_light_mask": args.output_dir / "visibility_debug_light_mask.png",
        "contact": args.output_dir / "visibility_contact.png",
        "summary": args.output_dir / "visibility_smoke_summary.json",
    }
    images = {
        "unshadowed": to_u8_rgb(unshadowed.composite_rgb),
        "shadowed": to_u8_rgb(shadowed.composite_rgb),
        "difference": to_u8_scalar(difference, valid),
        "debug light visibility": to_u8_scalar(debug_visibility, valid),
    }
    imageio.imwrite(paths["unshadowed"], images["unshadowed"])
    imageio.imwrite(paths["shadowed"], images["shadowed"])
    imageio.imwrite(paths["difference"], images["difference"])
    imageio.imwrite(paths["debug_light_mask"], images["debug light visibility"])
    make_contact_sheet(images, paths["contact"])

    nontrivial = visibility_min < 1.0 and visibility_max > 0.0
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
        "valid_pixels": int(valid.sum().detach().cpu()),
        "visibility_mean": visibility_mean,
        "visibility_min": visibility_min,
        "visibility_max": visibility_max,
        "visibility_nontrivial": bool(nontrivial),
        "shadowed_vs_unshadowed": metrics,
        "light_info": light_info,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"asset:     {args.asset_id}")
    print(f"frame:     {args.frame_index}")
    print(f"valid:     {summary['valid_pixels']}")
    print(f"vis mean:  {visibility_mean:.4f}")
    print(f"nontriv:   {nontrivial}")
    print(f"wrote:     {paths['summary'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
