from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.eval.proposal_ablation import run_proposal_ablation, summarize_rows
from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.render.camera_probe import load_camera_config
from restir_gs.render.gbuffer import make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.ply_loader import load_gaussian_ply_with_stats


CSV_FIELDS = [
    "proposal",
    "estimator",
    "k",
    "seed_index",
    "candidate_seed",
    "selection_seed",
    "reference_quantity",
    "mae",
    "rmse",
    "bias_r",
    "bias_g",
    "bias_b",
    "mean_abs_bias",
]


def parse_k_values(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one K value.")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"K values must be positive: {values}")
    return values


def write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run proposal ablation on a real 3DGS PLY using a selected camera.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--max-gaussians", type=int, default=200000)
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values("1,2,4,8,16,32"))
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--candidate-seed-base", type=int, default=9100)
    parser.add_argument("--selection-seed-base", type=int, default=10100)
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
    rows = run_proposal_ablation(
        gbuffer,
        lights,
        args.k_values,
        args.seed_count,
        candidate_seed_base=args.candidate_seed_base,
        selection_seed_base=args.selection_seed_base,
    )
    summary = summarize_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "real_asset_proposal_ablation.csv"
    json_path = args.output_dir / "real_asset_proposal_ablation_summary.json"
    write_csv(csv_path, rows)
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
            "k_values": args.k_values,
            "seed_count": args.seed_count,
            "candidate_seed_base": args.candidate_seed_base,
            "selection_seed_base": args.selection_seed_base,
            "row_count": len(rows),
        },
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("quantity, proposal, estimator, K, samples, MAE mean/std, RMSE mean/std")
    for row in summary:
        print(
            f"{row['reference_quantity']}, {row['proposal']}, {row['estimator']}, K={row['k']}, "
            f"n={row['sample_count']}, "
            f"MAE={float(row['mae_mean']):.8f}+/-{float(row['mae_std']):.8f}, "
            f"RMSE={float(row['rmse_mean']):.8f}+/-{float(row['rmse_std']):.8f}"
        )
    print(f"valid pixels: {valid_pixels}")
    print(f"rows: {len(rows)}")
    print(f"wrote: {csv_path.resolve()}")
    print(f"wrote: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
