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

from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.synthetic_scene import make_pinhole_camera, make_synthetic_gaussians


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def to_u8_scalar(values: torch.Tensor, valid: torch.Tensor | None = None) -> np.ndarray:
    data = values.detach().cpu()
    if valid is not None:
        mask = valid.detach().cpu().to(torch.bool)
    else:
        mask = torch.ones_like(data, dtype=torch.bool)

    out = torch.zeros_like(data, dtype=torch.float32)
    if bool(mask.any()):
        selected = data[mask]
        lo = selected.min()
        hi = selected.max()
        if float(hi - lo) > 1e-8:
            out[mask] = (selected - lo) / (hi - lo)
        else:
            out[mask] = 1.0
    return (out.numpy() * 255.0).astype(np.uint8)


def to_u8_position(position: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    data = position.detach().cpu()
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
    parser = argparse.ArgumentParser(description="Render synthetic Gaussians and build a pseudo G-buffer.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha-threshold", type=float, default=1e-4)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    scene = make_synthetic_gaussians(device=device)
    camera = make_pinhole_camera(width=args.width, height=args.height, device=device)
    render_buffers = render_rgbd(scene, camera)
    gbuffer = make_pseudo_gbuffer(render_buffers, camera, alpha_threshold=args.alpha_threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output_dir / "gbuffer_rgb.png", to_u8_rgb(gbuffer.rgb))
    imageio.imwrite(args.output_dir / "gbuffer_depth.png", to_u8_scalar(gbuffer.depth, gbuffer.valid_mask))
    imageio.imwrite(args.output_dir / "gbuffer_alpha.png", to_u8_scalar(gbuffer.alpha))
    imageio.imwrite(args.output_dir / "gbuffer_position.png", to_u8_position(gbuffer.position_cam, gbuffer.valid_mask))
    imageio.imwrite(args.output_dir / "gbuffer_normal.png", to_u8_normal(gbuffer.normal_cam, gbuffer.normal_mask))

    valid_count = int(gbuffer.valid_mask.sum().detach().cpu())
    normal_count = int(gbuffer.normal_mask.sum().detach().cpu())
    print(f"rgb:          {tuple(gbuffer.rgb.shape)}")
    print(f"depth:        {tuple(gbuffer.depth.shape)}")
    print(f"alpha:        {tuple(gbuffer.alpha.shape)}")
    print(f"position_cam: {tuple(gbuffer.position_cam.shape)}")
    print(f"normal_cam:   {tuple(gbuffer.normal_cam.shape)}")
    print(f"valid pixels: {valid_count}")
    print(f"normal pixels: {normal_count}")
    print(f"wrote: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
