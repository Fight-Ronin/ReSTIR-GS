from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.lighting.deferred import make_deterministic_point_lights, shade_deferred_lambertian
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.synthetic_scene import make_pinhole_camera, make_synthetic_gaussians


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Render synthetic Gaussians with naive deferred point lighting.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    scene = make_synthetic_gaussians(device=device)
    camera = make_pinhole_camera(width=args.width, height=args.height, device=device)
    render_buffers = render_rgbd(scene, camera)
    gbuffer = make_pseudo_gbuffer(render_buffers, camera)
    lights = make_deterministic_point_lights(count=args.num_lights, device=device)
    lighting = shade_deferred_lambertian(gbuffer, lights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output_dir / "deferred_base_rgb.png", to_u8_rgb(gbuffer.rgb))
    imageio.imwrite(args.output_dir / "deferred_irradiance.png", to_u8_vector(lighting.irradiance_rgb, lighting.valid_mask))
    imageio.imwrite(args.output_dir / "deferred_diffuse.png", to_u8_vector(lighting.diffuse_rgb, lighting.valid_mask))
    imageio.imwrite(args.output_dir / "deferred_composite.png", to_u8_rgb(lighting.composite_rgb))
    imageio.imwrite(args.output_dir / "deferred_normal.png", to_u8_normal(gbuffer.normal_cam, lighting.valid_mask))

    valid_count = int(lighting.valid_mask.sum().detach().cpu())
    print(f"irradiance_rgb: {tuple(lighting.irradiance_rgb.shape)}")
    print(f"diffuse_rgb:    {tuple(lighting.diffuse_rgb.shape)}")
    print(f"shade_rgb:      {tuple(lighting.shade_rgb.shape)}")
    print(f"composite_rgb:  {tuple(lighting.composite_rgb.shape)}")
    print(f"valid pixels:   {valid_count}")
    print(f"light count:    {args.num_lights}")
    print(f"wrote: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
