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

from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.synthetic_scene import make_pinhole_camera, make_synthetic_gaussians


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def to_u8_scalar(values: torch.Tensor, valid: torch.Tensor | None = None) -> np.ndarray:
    data = values.detach().cpu()
    if valid is not None:
        mask = valid.detach().cpu() > 0.0
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Render synthetic Gaussians with gsplat RGB+ED.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    scene = make_synthetic_gaussians(device=device)
    camera = make_pinhole_camera(width=args.width, height=args.height, device=device)
    buffers = render_rgbd(scene, camera)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output_dir / "synthetic_rgb.png", to_u8_rgb(buffers.rgb))
    imageio.imwrite(args.output_dir / "synthetic_depth.png", to_u8_scalar(buffers.depth, buffers.alpha))
    imageio.imwrite(args.output_dir / "synthetic_alpha.png", to_u8_scalar(buffers.alpha))

    print(f"rgb:   {tuple(buffers.rgb.shape)} sum={float(buffers.rgb.sum().detach().cpu()):.6f}")
    print(f"depth: {tuple(buffers.depth.shape)} sum={float(buffers.depth.sum().detach().cpu()):.6f}")
    print(f"alpha: {tuple(buffers.alpha.shape)} sum={float(buffers.alpha.sum().detach().cpu()):.6f}")
    print(f"wrote: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
