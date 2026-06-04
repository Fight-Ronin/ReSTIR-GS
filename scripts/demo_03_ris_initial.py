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
from restir_gs.restir.initial import estimate_ris_initial_diffuse, estimate_uniform_diffuse, sample_uniform_light_candidates


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


def mean_absolute_error(values: torch.Tensor, reference: torch.Tensor, valid: torch.Tensor) -> float:
    mask = valid.detach().cpu().to(torch.bool)
    if not bool(mask.any()):
        return 0.0
    error = (values.detach().cpu() - reference.detach().cpu()).abs()
    return float(error[mask].mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare initial RIS against uniform light sampling.")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--candidate-count", type=int, default=8)
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
    reference = shade_deferred_lambertian(gbuffer, lights)

    candidates_1 = sample_uniform_light_candidates(args.height, args.width, 1, args.num_lights, seed=2028, device=device)
    candidates_k = sample_uniform_light_candidates(
        args.height,
        args.width,
        args.candidate_count,
        args.num_lights,
        seed=2028,
        device=device,
    )
    uniform_1 = estimate_uniform_diffuse(gbuffer, lights, candidates_1)
    uniform_k = estimate_uniform_diffuse(gbuffer, lights, candidates_k)
    ris, _ = estimate_ris_initial_diffuse(gbuffer, lights, candidates_k)

    uniform_1_mae = mean_absolute_error(uniform_1.diffuse_rgb, reference.diffuse_rgb, reference.valid_mask)
    uniform_k_mae = mean_absolute_error(uniform_k.diffuse_rgb, reference.diffuse_rgb, reference.valid_mask)
    ris_mae = mean_absolute_error(ris.diffuse_rgb, reference.diffuse_rgb, reference.valid_mask)
    uniform_k_error = (uniform_k.diffuse_rgb - reference.diffuse_rgb).abs()
    ris_error = (ris.diffuse_rgb - reference.diffuse_rgb).abs()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output_dir / "ris_reference_composite.png", to_u8_rgb(reference.composite_rgb))
    imageio.imwrite(args.output_dir / "ris_uniform1_composite.png", to_u8_rgb(uniform_1.composite_rgb))
    imageio.imwrite(args.output_dir / "ris_uniformK_composite.png", to_u8_rgb(uniform_k.composite_rgb))
    imageio.imwrite(args.output_dir / "ris_initial_composite.png", to_u8_rgb(ris.composite_rgb))
    imageio.imwrite(args.output_dir / "ris_uniformK_abs_error.png", to_u8_vector(uniform_k_error, reference.valid_mask))
    imageio.imwrite(args.output_dir / "ris_initial_abs_error.png", to_u8_vector(ris_error, reference.valid_mask))

    print(f"reference diffuse: {tuple(reference.diffuse_rgb.shape)}")
    print(f"uniform1 diffuse:  {tuple(uniform_1.diffuse_rgb.shape)}")
    print(f"uniformK diffuse:  {tuple(uniform_k.diffuse_rgb.shape)}")
    print(f"ris diffuse:       {tuple(ris.diffuse_rgb.shape)}")
    print(f"valid pixels:      {int(reference.valid_mask.sum().detach().cpu())}")
    print(f"light count:       {args.num_lights}")
    print(f"candidate count:   {args.candidate_count}")
    print(f"uniform1 MAE:      {uniform_1_mae:.8f}")
    print(f"uniformK MAE:      {uniform_k_mae:.8f}")
    print(f"RIS MAE:           {ris_mae:.8f}")
    print(f"wrote: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
