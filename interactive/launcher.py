from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
)
from restir_gs.render.camera_probe import load_camera_config
from interactive.rendering import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
    ViewerAsset,
    ViewerSettings,
    make_visibility_cache,
    render_view,
    reset_state_from_frame,
    save_outputs,
    viewer_save_metadata,
)
from interactive.viewer import InteractiveViewer
from restir_gs.render.ply_loader import load_gaussian_asset, make_asset_camera


DEFAULT_TORCH_EXTENSIONS_DIR = Path("outputs/torch_extensions_restirgs")
DEFAULT_WINDOWS_TORCH_EXTENSIONS_DIR = Path("C:/tmp/torch_extensions_restirgs_cu124_patched")
DEFAULT_MPLCONFIGDIR = Path("outputs/matplotlib_cache")


def load_viewer_asset(args: argparse.Namespace, device: torch.device) -> ViewerAsset:
    if args.ply is not None:
        return load_generic_ply_viewer_asset(args, device=device)
    return load_registered_viewer_asset(args, device=device)


def load_registered_viewer_asset(args: argparse.Namespace, device: torch.device) -> ViewerAsset:
    manifest = load_aligned_asset_manifest(args.manifest)
    spec = get_aligned_asset_spec(manifest, args.asset_id)
    resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)
    asset = load_registered_aligned_asset(resolved, device=device, max_gaussians_override=args.max_gaussians)
    return ViewerAsset(
        label=f"Aligned {spec.asset_id}",
        scene=asset.loaded.scene,
        source_path=resolved.splat_path,
        frame_cameras=[frame.camera for frame in asset.transforms.frames],
        frame_labels=[str(frame.index) for frame in asset.transforms.frames],
        metadata={
            "source_mode": "aligned_registry",
            "asset_id": spec.asset_id,
            "dataset_type": spec.dataset_type,
            "manifest": str(args.manifest),
            "dataset_root": str(resolved.dataset_root),
            "splat_path": str(resolved.splat_path),
            "original_count": asset.loaded.stats.original_count,
            "loaded_count": asset.loaded.stats.loaded_count,
        },
    )


def load_generic_ply_viewer_asset(args: argparse.Namespace, device: torch.device) -> ViewerAsset:
    max_gaussians = None if args.max_gaussians <= 0 else args.max_gaussians
    loaded = load_gaussian_asset(args.ply, device=device, max_gaussians=max_gaussians)
    if args.camera_config is not None:
        camera = load_camera_config(args.camera_config, device=device)
        camera_source: dict[str, object] = {"mode": "camera_config", "path": str(args.camera_config)}
    else:
        camera, info = make_asset_camera(
            loaded.scene.means,
            width=args.width,
            height=args.height,
            bbox_percentile=args.auto_camera_bbox_percentile,
            radius_scale=args.auto_camera_radius_scale,
        )
        camera_source = {
            "mode": "auto_asset_camera",
            "target": info.target,
            "eye": info.eye,
            "bbox_min": info.bbox_min,
            "bbox_max": info.bbox_max,
            "bbox_diagonal": info.bbox_diagonal,
            "radius": info.radius,
            "focal": info.focal,
            "bbox_percentile": info.bbox_percentile,
            "radius_scale": info.radius_scale,
        }
    return ViewerAsset(
        label=f"3DGS {Path(args.ply).stem}",
        scene=loaded.scene,
        source_path=Path(args.ply),
        frame_cameras=[camera],
        frame_labels=["0"],
        metadata={
            "source_mode": "generic_3dgs_ply",
            "ply": str(args.ply),
            "stats": {
                "source_format": loaded.stats.source_format,
                "schema": loaded.stats.schema,
                "original_count": loaded.stats.original_count,
                "loaded_count": loaded.stats.loaded_count,
                "color_source": loaded.stats.color_source,
                "has_sh_rest": loaded.stats.has_sh_rest,
            },
            "camera_source": camera_source,
        },
    )


def make_viewer_settings(args: argparse.Namespace) -> ViewerSettings:
    return ViewerSettings(
        width=args.width,
        height=args.height,
        num_lights=args.num_lights,
        light_seed=args.light_seed,
        restir_candidate_count=args.restir_candidate_count,
        restir_candidate_seed=args.restir_candidate_seed,
        restir_selection_seed=args.restir_selection_seed,
        visibility_num_lights=args.visibility_num_lights,
        visibility_light_seed=args.visibility_light_seed,
        visibility_candidate_count=args.visibility_candidate_count,
        visibility_candidate_seed=args.visibility_candidate_seed,
        visibility_selection_seed=args.visibility_selection_seed,
        visibility_shadow_resolution=args.visibility_shadow_resolution,
        visibility_shadow_bias_scale=args.visibility_shadow_bias_scale,
        visibility_shadow_alpha_threshold=args.visibility_shadow_alpha_threshold,
        visibility_shadow_pcf_radius=args.visibility_shadow_pcf_radius,
        ambient=args.ambient,
        specular_strength=args.specular_strength,
        shininess=args.shininess,
        output_dir=args.output_dir,
    )


def _write_save_and_exit_summary(paths: dict[str, str], output_dir: Path, result) -> None:
    summary_path = output_dir / "interactive_viewer_save_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "version": 1,
                "outputs": paths,
                "render_ms": result.render_ms,
                "valid_pixels": result.valid_pixels,
                "timings": result.timings.as_dict(),
                "computed_views": list(result.computed_views),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def configure_viewer_runtime_environment(device: torch.device) -> None:
    if device.type != "cuda":
        return

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("MAX_JOBS", "4")
    default_extensions_dir = (
        DEFAULT_WINDOWS_TORCH_EXTENSIONS_DIR
        if platform.system() == "Windows"
        else (ROOT / DEFAULT_TORCH_EXTENSIONS_DIR).resolve()
    )
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(default_extensions_dir))
    os.environ.setdefault("MPLCONFIGDIR", str((ROOT / DEFAULT_MPLCONFIGDIR).resolve()))
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    if platform.system() == "Windows" and shutil.which("cl") is None and os.environ.get("RESTIRGS_SKIP_CL_WARNING") != "1":
        print(
            "warning: MSVC cl.exe is not on PATH. Existing gsplat CUDA extension cache may still work; "
            "if JIT compilation fails, run scripts\\run_interactive_viewer_windows.bat or use an x64 Native Tools shell.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a lightweight interactive 3DGS viewer.")
    parser.add_argument("--ply", type=Path, default=None, help="Generic compatible 3DGS PLY to view. Omit to use a registered aligned asset.")
    parser.add_argument("--camera-config", type=Path, default=None, help="Optional camera config for --ply mode.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-id", default="dxgl_apple", help="Registered aligned asset id to view. Ignored when --ply is provided.")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWER_HEIGHT)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--auto-camera-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--auto-camera-radius-scale", type=float, default=1.8)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--restir-candidate-count", type=int, default=8)
    parser.add_argument("--restir-candidate-seed", type=int, default=34100)
    parser.add_argument("--restir-selection-seed", type=int, default=35100)
    parser.add_argument("--visibility-num-lights", type=int, default=16)
    parser.add_argument("--visibility-light-seed", type=int, default=2027)
    parser.add_argument("--visibility-candidate-count", type=int, default=8)
    parser.add_argument("--visibility-candidate-seed", type=int, default=36100)
    parser.add_argument("--visibility-selection-seed", type=int, default=37100)
    parser.add_argument("--visibility-shadow-resolution", type=int, default=128)
    parser.add_argument("--visibility-shadow-bias-scale", type=float, default=0.02)
    parser.add_argument("--visibility-shadow-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--visibility-shadow-pcf-radius", type=int, default=1)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--specular-strength", type=float, default=0.15)
    parser.add_argument("--shininess", type=float, default=24.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-and-exit", action="store_true")
    parser.add_argument("--save-visibility", action="store_true", help="With --save-and-exit, also save the visibility RIS display output.")
    parser.add_argument(
        "--save-visibility-reference",
        action="store_true",
        help="With --save-and-exit, also compute and save visibility reference/error outputs.",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Expected positive viewer size, got {args.width}x{args.height}")
    if args.restir_candidate_count <= 0:
        raise ValueError(f"Expected positive restir_candidate_count, got {args.restir_candidate_count}")
    if args.visibility_num_lights <= 0 or args.visibility_candidate_count <= 0:
        raise ValueError("Expected positive visibility light and candidate counts.")
    if args.visibility_shadow_resolution <= 0:
        raise ValueError(f"Expected positive visibility_shadow_resolution, got {args.visibility_shadow_resolution}")
    if args.visibility_shadow_bias_scale < 0.0 or args.visibility_shadow_alpha_threshold < 0.0:
        raise ValueError("Expected non-negative visibility shadow bias scale and alpha threshold.")
    if args.visibility_shadow_pcf_radius < 0:
        raise ValueError(f"Expected non-negative visibility_shadow_pcf_radius, got {args.visibility_shadow_pcf_radius}")
    if args.save_visibility_reference:
        args.save_visibility = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    settings = make_viewer_settings(args)
    device = torch.device(args.device)
    configure_viewer_runtime_environment(device)
    asset = load_viewer_asset(args, device=device)
    if args.frame_index is None:
        args.frame_index = 49 if len(asset.frame_cameras) > 49 else 0
    if args.frame_index < 0 or args.frame_index >= len(asset.frame_cameras):
        raise ValueError(f"Frame index {args.frame_index} exceeds frame count {len(asset.frame_cameras)}.")

    if args.save_and_exit:
        state = reset_state_from_frame(asset, args.frame_index, settings.width, settings.height, device)
        visibility_cache = None
        if args.save_visibility:
            visibility_cache = make_visibility_cache(
                asset,
                num_lights=settings.visibility_num_lights,
                light_seed=settings.visibility_light_seed,
                shadow_resolution=settings.visibility_shadow_resolution,
                shadow_bias_scale=settings.visibility_shadow_bias_scale,
                device=device,
            )
        result = render_view(
            asset,
            args.frame_index,
            state,
            settings,
            visibility_cache,
            device=device,
            required_view="blinn_phong",
            include_visibility_reference=args.save_visibility_reference,
        )
        paths = save_outputs(result, settings.output_dir, metadata=viewer_save_metadata(settings, result, asset))
        _write_save_and_exit_summary(paths, settings.output_dir, result)
        for key, path in paths.items():
            print(f"{key}: {Path(path).resolve()}")
        return 0

    viewer = InteractiveViewer(asset, settings, device, args.frame_index)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
