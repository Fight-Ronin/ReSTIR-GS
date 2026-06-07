from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights, make_asset_scaled_world_lights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import LightingBuffers, shade_deferred_blinn_phong, shade_deferred_lambertian
from restir_gs.lighting.visibility import ShadowMapBundle, make_shadow_map_bundle, shade_deferred_lambertian_visible
from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    load_registered_aligned_asset,
    resolve_aligned_asset_paths,
)
from restir_gs.render.camera_probe import load_camera_config
from restir_gs.render.dxgl_asset import scale_camera
from restir_gs.render.gbuffer import GBuffer, make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.orbit_camera import (
    OrbitCameraState,
    orbit_state_dolly,
    orbit_state_from_camera,
    orbit_state_orbit,
    orbit_state_pan,
    orbit_state_to_camera,
    save_orbit_camera_config,
)
from restir_gs.render.ply_loader import load_gaussian_asset, make_asset_camera
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.render.synthetic_scene import SyntheticGaussians
from restir_gs.restir.initial import LightingEstimatorBuffers, estimate_proposal_lighting, estimate_ris_initial_lighting
from restir_gs.restir.proposal import (
    compute_geometric_proposal_distribution,
    compute_visibility_geometric_proposal_distribution,
    sample_light_candidates_from_distribution,
)
from restir_gs.restir.visibility import estimate_visibility_proposal_lighting, estimate_visibility_ris_initial_lighting


DEFAULT_OUTPUT_DIR = Path("outputs/interactive_viewer")
DEFAULT_TORCH_EXTENSIONS_DIR = Path("outputs/torch_extensions_restirgs")
DEFAULT_WINDOWS_TORCH_EXTENSIONS_DIR = Path("C:/tmp/torch_extensions_restirgs_cu124_patched")
DEFAULT_MPLCONFIGDIR = Path("outputs/matplotlib_cache")


@dataclass(frozen=True)
class ViewerAsset:
    label: str
    scene: SyntheticGaussians
    source_path: Path
    frame_cameras: list[PinholeCamera]
    frame_labels: list[str]
    metadata: dict[str, object]


@dataclass
class ViewerRestirResult:
    reference: LightingBuffers
    geometric_mc: LightingEstimatorBuffers
    initial_ris: LightingEstimatorBuffers
    proposal_confidence: torch.Tensor


@dataclass
class ViewerVisibilityCache:
    world_lights: Any
    shadow_bundle: ShadowMapBundle
    light_info: dict[str, object]


@dataclass
class ViewerVisibilityResult:
    reference: LightingBuffers
    geometric_mc: LightingEstimatorBuffers
    initial_ris: LightingEstimatorBuffers
    error: torch.Tensor


@dataclass
class ViewerRenderResult:
    frame_index: int
    state: OrbitCameraState
    gbuffer: GBuffer
    lambertian: LightingBuffers
    blinn_phong: LightingBuffers
    restir: ViewerRestirResult
    visibility: ViewerVisibilityResult | None
    valid_pixels: int
    render_ms: float
    light_info: dict[str, object]


def to_u8_rgb(rgb: torch.Tensor) -> np.ndarray:
    return (rgb.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)


def to_u8_scalar(values: torch.Tensor, mask: torch.Tensor | None = None) -> np.ndarray:
    data = values.detach().cpu().float()
    valid = torch.isfinite(data)
    if mask is not None:
        valid = valid & mask.detach().cpu().to(torch.bool)
    valid = valid & (data > 0.0)
    out = torch.zeros_like(data)
    if bool(valid.any()):
        selected = data[valid]
        lo = selected.min()
        hi = selected.max()
        denom = hi - lo if float(hi - lo) > 1e-8 else torch.tensor(1.0)
        out[valid] = (selected - lo) / denom
    return (out.numpy() * 255.0).astype(np.uint8)


def to_u8_normal(normal_cam: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    data = ((normal_cam.detach().cpu() * 0.5) + 0.5).clamp(0.0, 1.0)
    out = torch.zeros_like(data)
    valid = mask.detach().cpu().to(torch.bool)
    out[valid] = data[valid]
    return (out.numpy() * 255.0).astype(np.uint8)


def to_u8_normalized_rgb(rgb: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    data = rgb.detach().cpu().float()
    valid = valid_mask.detach().cpu().to(torch.bool)
    out = torch.zeros_like(data)
    if bool(valid.any()):
        selected = data[valid]
        hi = torch.clamp(selected.max(), min=1e-8)
        out[valid] = (selected / hi).clamp(0.0, 1.0)
    return (out.numpy() * 255.0).astype(np.uint8)


def render_view(
    asset: ViewerAsset,
    frame_index: int,
    state: OrbitCameraState,
    num_lights: int,
    light_seed: int,
    restir_candidate_count: int,
    restir_candidate_seed: int,
    restir_selection_seed: int,
    ambient: float,
    specular_strength: float,
    shininess: float,
    visibility_cache: ViewerVisibilityCache | None,
    visibility_candidate_count: int,
    visibility_candidate_seed: int,
    visibility_selection_seed: int,
    visibility_shadow_alpha_threshold: float,
    visibility_shadow_pcf_radius: int,
    device: torch.device,
) -> ViewerRenderResult:
    camera = orbit_state_to_camera(state, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        render_buffers = render_rgbd(asset.scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        lights, light_info = make_asset_scaled_point_lights(gbuffer, count=num_lights, seed=light_seed, device=device)
        lambertian = shade_deferred_lambertian(gbuffer, lights, ambient=ambient)
        blinn_phong = shade_deferred_blinn_phong(
            gbuffer,
            lights,
            ambient=ambient,
            specular_strength=specular_strength,
            shininess=shininess,
        )
        proposal = compute_geometric_proposal_distribution(gbuffer, lights)
        samples = sample_light_candidates_from_distribution(
            proposal,
            restir_candidate_count,
            seed=restir_candidate_seed,
            device=device,
        )
        geometric_mc = estimate_proposal_lighting(
            gbuffer,
            lights,
            samples,
            ambient=ambient,
            target_mode="diffuse",
        )
        initial_ris, _ = estimate_ris_initial_lighting(
            gbuffer,
            lights,
            samples.light_indices,
            selection_seed=restir_selection_seed,
            ambient=ambient,
            proposal_probs=samples.proposal_probs,
            target_mode="diffuse",
        )
        proposal_confidence = proposal.max(dim=-1).values
        visibility_result = None
        if visibility_cache is not None:
            visibility_lights = world_lights_to_camera_lights(visibility_cache.world_lights, camera)
            visibility_reference = shade_deferred_lambertian_visible(
                gbuffer,
                camera,
                visibility_lights,
                visibility_cache.shadow_bundle,
                ambient=ambient,
                alpha_threshold=visibility_shadow_alpha_threshold,
                pcf_radius=visibility_shadow_pcf_radius,
            )
            visibility_proposal = compute_visibility_geometric_proposal_distribution(
                gbuffer,
                camera,
                visibility_lights,
                visibility_cache.shadow_bundle,
                alpha_threshold=visibility_shadow_alpha_threshold,
                pcf_radius=visibility_shadow_pcf_radius,
            )
            visibility_samples = sample_light_candidates_from_distribution(
                visibility_proposal,
                visibility_candidate_count,
                seed=visibility_candidate_seed,
                device=device,
            )
            visibility_mc = estimate_visibility_proposal_lighting(
                gbuffer,
                camera,
                visibility_lights,
                visibility_cache.shadow_bundle,
                visibility_samples,
                ambient=ambient,
                alpha_threshold=visibility_shadow_alpha_threshold,
                pcf_radius=visibility_shadow_pcf_radius,
            )
            visibility_ris, _ = estimate_visibility_ris_initial_lighting(
                gbuffer,
                camera,
                visibility_lights,
                visibility_cache.shadow_bundle,
                visibility_samples.light_indices,
                proposal_probs=visibility_samples.proposal_probs,
                selection_seed=visibility_selection_seed,
                ambient=ambient,
                alpha_threshold=visibility_shadow_alpha_threshold,
                pcf_radius=visibility_shadow_pcf_radius,
            )
            visibility_error = torch.abs(visibility_ris.contribution_rgb - visibility_reference.diffuse_rgb).mean(dim=-1)
            visibility_result = ViewerVisibilityResult(
                reference=visibility_reference,
                geometric_mc=visibility_mc,
                initial_ris=visibility_ris,
                error=visibility_error,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    render_ms = (time.perf_counter() - start) * 1000.0
    valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
    return ViewerRenderResult(
        frame_index=frame_index,
        state=state,
        gbuffer=gbuffer,
        lambertian=lambertian,
        blinn_phong=blinn_phong,
        restir=ViewerRestirResult(
            reference=lambertian,
            geometric_mc=geometric_mc,
            initial_ris=initial_ris,
            proposal_confidence=proposal_confidence,
        ),
        visibility=visibility_result,
        valid_pixels=valid_pixels,
        render_ms=render_ms,
        light_info=light_info,
    )


def reset_state_from_frame(
    asset: ViewerAsset,
    frame_index: int,
    width: int,
    height: int,
    device: torch.device,
) -> OrbitCameraState:
    camera = scale_camera(asset.frame_cameras[frame_index], width, height)
    with torch.no_grad():
        render_buffers = render_rgbd(asset.scene, camera)
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
    target = _target_on_camera_forward_from_visible_depth(camera, gbuffer)
    return orbit_state_from_camera(camera, target=target)


def make_visibility_cache(
    asset: ViewerAsset,
    num_lights: int,
    light_seed: int,
    shadow_resolution: int,
    shadow_bias_scale: float,
    device: torch.device,
) -> ViewerVisibilityCache:
    world_lights, light_info = make_asset_scaled_world_lights(
        asset.scene.means,
        count=num_lights,
        seed=light_seed,
        device=device,
    )
    target_world = torch.tensor(light_info["center"], dtype=torch.float32, device=device)
    shadow_bundle = make_shadow_map_bundle(
        asset.scene,
        world_lights.positions_world,
        torch.arange(num_lights, dtype=torch.long, device=device),
        target_world,
        scene_radius=float(light_info["radius"]),
        resolution=shadow_resolution,
        shadow_bias_scale=shadow_bias_scale,
    )
    return ViewerVisibilityCache(world_lights=world_lights, shadow_bundle=shadow_bundle, light_info=light_info)


def save_outputs(
    result: ViewerRenderResult,
    output_dir: Path,
    metadata: dict[str, object] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "camera": output_dir / "current_camera.json",
        "rgb": output_dir / "current_rgb.png",
        "alpha": output_dir / "current_alpha.png",
        "normal": output_dir / "current_normal.png",
        "blinn_phong": output_dir / "current_blinn_phong.png",
    }
    if result.visibility is not None:
        paths["visibility_reference"] = output_dir / "current_visibility_reference.png"
        paths["visibility_ris"] = output_dir / "current_visibility_ris.png"
        paths["visibility_error"] = output_dir / "current_visibility_error.png"
    save_orbit_camera_config(result.state, paths["camera"], metadata=metadata)
    imageio.imwrite(paths["rgb"], to_u8_rgb(result.gbuffer.rgb))
    imageio.imwrite(paths["alpha"], (result.gbuffer.alpha.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8))
    imageio.imwrite(paths["normal"], to_u8_normal(result.gbuffer.normal_cam, result.gbuffer.normal_mask))
    imageio.imwrite(paths["blinn_phong"], to_u8_rgb(result.blinn_phong.composite_rgb))
    if result.visibility is not None:
        imageio.imwrite(paths["visibility_reference"], to_u8_rgb(result.visibility.reference.composite_rgb))
        imageio.imwrite(paths["visibility_ris"], to_u8_rgb(result.visibility.initial_ris.composite_rgb))
        imageio.imwrite(paths["visibility_error"], to_u8_scalar(result.visibility.error, result.visibility.reference.valid_mask))
    return {key: str(path) for key, path in paths.items()}


def panel_images(result: ViewerRenderResult, mode: str) -> list[tuple[str, np.ndarray]]:
    gbuffer = result.gbuffer
    valid = gbuffer.valid_mask & gbuffer.normal_mask
    restir_error = torch.abs(result.restir.initial_ris.contribution_rgb - result.restir.reference.diffuse_rgb).mean(dim=-1)
    shared = {
        "RGB": to_u8_rgb(gbuffer.rgb),
        "Alpha": (gbuffer.alpha.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8),
        "Depth": to_u8_scalar(gbuffer.depth, gbuffer.valid_mask),
        "Normal": to_u8_normal(gbuffer.normal_cam, gbuffer.normal_mask),
        "Valid": (valid.detach().cpu().numpy().astype(np.uint8) * 255),
        "Lambertian": to_u8_rgb(result.lambertian.composite_rgb),
        "Blinn-Phong": to_u8_rgb(result.blinn_phong.composite_rgb),
        "Specular": to_u8_normalized_rgb(result.blinn_phong.specular_rgb, result.blinn_phong.valid_mask),
        "Reference": to_u8_rgb(result.restir.reference.composite_rgb),
        "Geometric MC": to_u8_rgb(result.restir.geometric_mc.composite_rgb),
        "Initial RIS": to_u8_rgb(result.restir.initial_ris.composite_rgb),
        "Initial Error": to_u8_scalar(restir_error, result.restir.reference.valid_mask),
        "Proposal Max": to_u8_scalar(result.restir.proposal_confidence, valid),
    }
    if result.visibility is not None:
        shared.update(
            {
                "Visible Reference": to_u8_rgb(result.visibility.reference.composite_rgb),
                "Visible Geom MC": to_u8_rgb(result.visibility.geometric_mc.composite_rgb),
                "Visible RIS": to_u8_rgb(result.visibility.initial_ris.composite_rgb),
                "Visible Error": to_u8_scalar(result.visibility.error, result.visibility.reference.valid_mask),
            }
        )
    if mode == "gbuffer":
        keys = ["RGB", "Alpha", "Depth", "Normal", "Valid", "Blinn-Phong"]
    elif mode == "lighting":
        keys = ["RGB", "Lambertian", "Blinn-Phong", "Specular", "Normal", "Alpha"]
    elif mode == "restir":
        keys = ["Reference", "Geometric MC", "Initial RIS", "Initial Error", "Proposal Max", "Alpha"]
    elif mode == "visibility":
        if result.visibility is None:
            keys = ["RGB", "Alpha", "Depth", "Normal", "Lambertian", "Blinn-Phong"]
        else:
            keys = ["Visible Reference", "Visible Geom MC", "Visible RIS", "Visible Error", "Proposal Max", "Alpha"]
    else:
        keys = ["RGB", "Alpha", "Depth", "Normal", "Lambertian", "Blinn-Phong"]
    return [(key, shared[key]) for key in keys]


class InteractiveViewer:
    def __init__(self, asset: ViewerAsset, args: argparse.Namespace, device: torch.device) -> None:
        self.asset = asset
        self.args = args
        self.device = device
        self.frame_index = int(args.frame_index)
        self.mode = "rgb"
        self.visibility_cache: ViewerVisibilityCache | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_mode: str | None = None
        self.closed = False
        self.state = reset_state_from_frame(asset, self.frame_index, args.width, args.height, device)
        self.result = self._render()

    def _render(self) -> ViewerRenderResult:
        return render_view(
            self.asset,
            self.frame_index,
            self.state,
            num_lights=self.args.num_lights,
            light_seed=self.args.light_seed,
            restir_candidate_count=self.args.restir_candidate_count,
            restir_candidate_seed=self.args.restir_candidate_seed,
            restir_selection_seed=self.args.restir_selection_seed,
            ambient=self.args.ambient,
            specular_strength=self.args.specular_strength,
            shininess=self.args.shininess,
            visibility_cache=self.visibility_cache if self.mode == "visibility" else None,
            visibility_candidate_count=self.args.visibility_candidate_count,
            visibility_candidate_seed=self.args.visibility_candidate_seed,
            visibility_selection_seed=self.args.visibility_selection_seed,
            visibility_shadow_alpha_threshold=self.args.visibility_shadow_alpha_threshold,
            visibility_shadow_pcf_radius=self.args.visibility_shadow_pcf_radius,
            device=self.device,
        )

    def run(self) -> None:
        import matplotlib.pyplot as plt

        self.fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        self.axes = list(axes.flat)
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.draw()
        plt.show()

    def draw(self) -> None:
        for ax, (title, image) in zip(self.axes, panel_images(self.result, self.mode), strict=True):
            ax.clear()
            ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        state = self.result.state
        self.fig.suptitle(
            f"{self.asset.label} interactive viewer | frame={self.frame_index} | mode={self.mode} | "
            f"yaw={state.yaw_degrees:.1f} pitch={state.pitch_degrees:.1f} radius={state.radius:.3f} | "
            f"valid={self.result.valid_pixels} | render={self.result.render_ms:.1f} ms",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

    def rerender_and_draw(self) -> None:
        try:
            self.result = self._render()
        except RuntimeError as exc:
            print(f"render failed: {exc}")
            return
        self.draw()

    def on_button_press(self, event: Any) -> None:
        if event.x is None or event.y is None:
            return
        shift = isinstance(event.key, str) and "shift" in event.key.lower()
        if event.button == 2 or (event.button == 1 and shift):
            self.drag_mode = "pan"
        elif event.button == 1:
            self.drag_mode = "orbit"
        else:
            self.drag_mode = None
        self.drag_start = (float(event.x), float(event.y))

    def on_button_release(self, event: Any) -> None:
        self.drag_start = None
        self.drag_mode = None

    def on_motion(self, event: Any) -> None:
        if self.drag_start is None or self.drag_mode is None or event.x is None or event.y is None:
            return
        x0, y0 = self.drag_start
        dx = float(event.x) - x0
        dy = float(event.y) - y0
        self.drag_start = (float(event.x), float(event.y))
        if self.drag_mode == "orbit":
            self.state = orbit_state_orbit(self.state, delta_yaw_degrees=dx * 0.25, delta_pitch_degrees=-dy * 0.25)
        else:
            pan_scale = self.state.radius / float(max(self.state.width, self.state.height))
            self.state = orbit_state_pan(
                self.state,
                delta_right=-dx * pan_scale,
                delta_up=dy * pan_scale,
                device="cpu",
            )
        self.rerender_and_draw()

    def on_scroll(self, event: Any) -> None:
        step = float(getattr(event, "step", 0.0))
        if step == 0.0:
            return
        self.state = orbit_state_dolly(self.state, scale=0.9**step)
        self.rerender_and_draw()

    def on_key(self, event: Any) -> None:
        key = event.key
        if key in ("q", "escape"):
            self.on_close(event)
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            return
        if key == "1":
            self.mode = "rgb"
            self.draw()
            return
        if key == "2":
            self.mode = "gbuffer"
            self.draw()
            return
        if key == "3":
            self.mode = "lighting"
            self.draw()
            return
        if key == "4":
            self.mode = "restir"
            self.draw()
            return
        if key == "5":
            self.mode = "visibility"
            if self.visibility_cache is None:
                print("building visibility shadow-map cache...")
                self.visibility_cache = make_visibility_cache(
                    self.asset,
                    num_lights=self.args.visibility_num_lights,
                    light_seed=self.args.visibility_light_seed,
                    shadow_resolution=self.args.visibility_shadow_resolution,
                    shadow_bias_scale=self.args.visibility_shadow_bias_scale,
                    device=self.device,
                )
            self.rerender_and_draw()
            return
        if key == "r":
            self.state = reset_state_from_frame(self.asset, self.frame_index, self.args.width, self.args.height, self.device)
            self.rerender_and_draw()
            return
        if key == "s":
            self.save_current()
            return
        if key in ("[", "]"):
            delta = -1 if key == "[" else 1
            self.frame_index = max(0, min(len(self.asset.frame_cameras) - 1, self.frame_index + delta))
            self.state = reset_state_from_frame(self.asset, self.frame_index, self.args.width, self.args.height, self.device)
            self.rerender_and_draw()

    def save_current(self) -> None:
        metadata = _save_metadata(self.args, self.result, self.asset)
        paths = save_outputs(self.result, self.args.output_dir, metadata=metadata)
        print("saved interactive viewer outputs:")
        for key, path in paths.items():
            print(f"  {key}: {Path(path).resolve()}")

    def on_close(self, event: Any) -> None:
        self.closed = True


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


def _target_on_camera_forward_from_visible_depth(camera: PinholeCamera, gbuffer: GBuffer) -> tuple[float, float, float]:
    valid = gbuffer.valid_mask & torch.isfinite(gbuffer.depth) & (gbuffer.depth > 0.0)
    if not bool(valid.any()):
        raise RuntimeError("Cannot initialize orbit target from a frame with no valid rendered depth.")
    depths = gbuffer.depth[valid].detach()
    depth = float(torch.median(depths).detach().cpu())
    target_cam = torch.tensor([0.0, 0.0, depth, 1.0], dtype=torch.float32, device=camera.viewmats.device)
    target_world = torch.linalg.inv(camera.viewmats[0]) @ target_cam
    data = target_world[:3].detach().cpu().tolist()
    return (float(data[0]), float(data[1]), float(data[2]))


def _save_metadata(args: argparse.Namespace, result: ViewerRenderResult, asset: ViewerAsset) -> dict[str, object]:
    return {
        "phase": "phase25_interactive_viewer",
        "asset_label": asset.label,
        "source_path": str(asset.source_path),
        "asset_metadata": asset.metadata,
        "frame_index": result.frame_index,
        "frame_label": asset.frame_labels[result.frame_index],
        "valid_pixels": result.valid_pixels,
        "render_ms": result.render_ms,
        "width": args.width,
        "height": args.height,
        "num_lights": args.num_lights,
        "light_seed": args.light_seed,
        "restir_candidate_count": args.restir_candidate_count,
        "restir_candidate_seed": args.restir_candidate_seed,
        "restir_selection_seed": args.restir_selection_seed,
        "visibility_num_lights": args.visibility_num_lights,
        "visibility_light_seed": args.visibility_light_seed,
        "visibility_candidate_count": args.visibility_candidate_count,
        "visibility_candidate_seed": args.visibility_candidate_seed,
        "visibility_selection_seed": args.visibility_selection_seed,
        "visibility_shadow_resolution": args.visibility_shadow_resolution,
        "visibility_shadow_bias_scale": args.visibility_shadow_bias_scale,
        "visibility_shadow_alpha_threshold": args.visibility_shadow_alpha_threshold,
        "visibility_shadow_pcf_radius": args.visibility_shadow_pcf_radius,
        "ambient": args.ambient,
        "specular_strength": args.specular_strength,
        "shininess": args.shininess,
        "light_info": result.light_info,
    }


def _write_save_and_exit_summary(paths: dict[str, str], output_dir: Path) -> None:
    summary_path = output_dir / "interactive_viewer_save_summary.json"
    summary_path.write_text(json.dumps({"version": 1, "outputs": paths}, indent=2), encoding="utf-8")


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
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
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
    parser.add_argument("--save-visibility", action="store_true", help="With --save-and-exit, also save the visibility inspection outputs.")
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
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    configure_viewer_runtime_environment(device)
    asset = load_viewer_asset(args, device=device)
    if args.frame_index is None:
        args.frame_index = 49 if len(asset.frame_cameras) > 49 else 0
    if args.frame_index < 0 or args.frame_index >= len(asset.frame_cameras):
        raise ValueError(f"Frame index {args.frame_index} exceeds frame count {len(asset.frame_cameras)}.")

    if args.save_and_exit:
        state = reset_state_from_frame(asset, args.frame_index, args.width, args.height, device)
        visibility_cache = None
        if args.save_visibility:
            visibility_cache = make_visibility_cache(
                asset,
                num_lights=args.visibility_num_lights,
                light_seed=args.visibility_light_seed,
                shadow_resolution=args.visibility_shadow_resolution,
                shadow_bias_scale=args.visibility_shadow_bias_scale,
                device=device,
            )
        result = render_view(
            asset,
            args.frame_index,
            state,
            num_lights=args.num_lights,
            light_seed=args.light_seed,
            restir_candidate_count=args.restir_candidate_count,
            restir_candidate_seed=args.restir_candidate_seed,
            restir_selection_seed=args.restir_selection_seed,
            ambient=args.ambient,
            specular_strength=args.specular_strength,
            shininess=args.shininess,
            visibility_cache=visibility_cache,
            visibility_candidate_count=args.visibility_candidate_count,
            visibility_candidate_seed=args.visibility_candidate_seed,
            visibility_selection_seed=args.visibility_selection_seed,
            visibility_shadow_alpha_threshold=args.visibility_shadow_alpha_threshold,
            visibility_shadow_pcf_radius=args.visibility_shadow_pcf_radius,
            device=device,
        )
        paths = save_outputs(result, args.output_dir, metadata=_save_metadata(args, result, asset))
        _write_save_and_exit_summary(paths, args.output_dir)
        for key, path in paths.items():
            print(f"{key}: {Path(path).resolve()}")
        return 0

    viewer = InteractiveViewer(asset, args, device)
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
