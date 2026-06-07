from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from interactive.layers import viewer_computed_views, viewer_render_requirements
from restir_gs.lighting.asset_lights import make_asset_scaled_world_lights, world_lights_to_camera_lights
from restir_gs.lighting.deferred import LightingBuffers, shade_deferred_blinn_phong
from restir_gs.lighting.visibility import ShadowMapBundle, make_shadow_map_bundle
from restir_gs.render.gbuffer import GBuffer, make_pseudo_gbuffer
from restir_gs.render.gsplat_renderer import render_rgbd
from restir_gs.render.orbit_camera import (
    OrbitCameraState,
    orbit_state_from_camera,
    orbit_state_to_camera,
    save_orbit_camera_config,
)
from restir_gs.render.synthetic_scene import PinholeCamera, SyntheticGaussians
from restir_gs.restir.initial import LightingEstimatorBuffers
from restir_gs.restir.proposal import compute_geometric_proposal_distribution
from restir_gs.restir.renderer import (
    RestirDisplayFrameResult,
    RestirFrameResult,
    RestirRenderSettings,
    evaluate_restir_display_frame_from_gbuffer,
    evaluate_restir_frame_from_gbuffer,
)


DEFAULT_OUTPUT_DIR = Path("outputs/interactive_viewer")
DEFAULT_VIEWER_WIDTH = 768
DEFAULT_VIEWER_HEIGHT = 768


@dataclass(frozen=True)
class ViewerSettings:
    width: int = DEFAULT_VIEWER_WIDTH
    height: int = DEFAULT_VIEWER_HEIGHT
    num_lights: int = 128
    light_seed: int = 2027
    restir_candidate_count: int = 8
    restir_candidate_seed: int = 34100
    restir_selection_seed: int = 35100
    visibility_num_lights: int = 16
    visibility_light_seed: int = 2027
    visibility_candidate_count: int = 8
    visibility_candidate_seed: int = 36100
    visibility_selection_seed: int = 37100
    visibility_shadow_resolution: int = 128
    visibility_shadow_bias_scale: float = 0.02
    visibility_shadow_alpha_threshold: float = 1e-4
    visibility_shadow_pcf_radius: int = 1
    ambient: float = 0.2
    specular_strength: float = 0.15
    shininess: float = 24.0
    output_dir: Path = DEFAULT_OUTPUT_DIR


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
    reference: LightingBuffers | None
    geometric_mc: LightingEstimatorBuffers | None
    initial_ris: LightingEstimatorBuffers
    error: torch.Tensor | None


@dataclass(frozen=True)
class ViewerTimings:
    render_rgbd_gpu_ms: float = 0.0
    gbuffer_gpu_ms: float = 0.0
    world_lights_gpu_ms: float = 0.0
    diffuse_restir_gpu_ms: float = 0.0
    blinn_phong_gpu_ms: float = 0.0
    proposal_confidence_gpu_ms: float = 0.0
    visibility_gpu_ms: float = 0.0
    total_gpu_ms: float = 0.0
    total_wall_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "render_rgbd_gpu_ms": float(self.render_rgbd_gpu_ms),
            "gbuffer_gpu_ms": float(self.gbuffer_gpu_ms),
            "world_lights_gpu_ms": float(self.world_lights_gpu_ms),
            "diffuse_restir_gpu_ms": float(self.diffuse_restir_gpu_ms),
            "blinn_phong_gpu_ms": float(self.blinn_phong_gpu_ms),
            "proposal_confidence_gpu_ms": float(self.proposal_confidence_gpu_ms),
            "visibility_gpu_ms": float(self.visibility_gpu_ms),
            "total_gpu_ms": float(self.total_gpu_ms),
            "total_wall_ms": float(self.total_wall_ms),
        }


@dataclass
class ViewerRenderResult:
    frame_index: int
    state: OrbitCameraState
    gbuffer: GBuffer
    lambertian: LightingBuffers | None
    blinn_phong: LightingBuffers | None
    restir: ViewerRestirResult | None
    visibility: ViewerVisibilityResult | None
    valid_pixels: int
    render_ms: float
    light_info: dict[str, object]
    timings: ViewerTimings = ViewerTimings()
    computed_views: tuple[str, ...] = ()


class _ViewerGpuTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.enabled = device.type == "cuda" and torch.cuda.is_available()
        self.events: dict[str, torch.cuda.Event] = {}

    def mark(self, label: str) -> None:
        if not self.enabled:
            return
        with torch.cuda.device(self.device):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        self.events[label] = event

    def elapsed(self, start: str, end: str) -> float:
        if not self.enabled or start not in self.events or end not in self.events:
            return 0.0
        return float(self.events[start].elapsed_time(self.events[end]))

    def to_timings(self) -> ViewerTimings:
        if not self.enabled:
            return ViewerTimings()
        torch.cuda.synchronize(self.device)
        return ViewerTimings(
            render_rgbd_gpu_ms=self.elapsed("start", "after_render_rgbd"),
            gbuffer_gpu_ms=self.elapsed("after_render_rgbd", "after_gbuffer"),
            world_lights_gpu_ms=self.elapsed("after_gbuffer", "after_world_lights"),
            diffuse_restir_gpu_ms=self.elapsed("after_world_lights", "after_diffuse_restir"),
            blinn_phong_gpu_ms=self.elapsed("after_diffuse_restir", "after_blinn_phong"),
            proposal_confidence_gpu_ms=self.elapsed("after_blinn_phong", "after_proposal_confidence"),
            visibility_gpu_ms=self.elapsed("after_proposal_confidence", "after_visibility"),
            total_gpu_ms=self.elapsed("start", "after_visibility"),
        )


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
    settings: ViewerSettings,
    visibility_cache: ViewerVisibilityCache | None,
    device: torch.device,
    required_view: str = "blinn_phong",
    include_visibility_reference: bool = False,
) -> ViewerRenderResult:
    requirements = viewer_render_requirements(required_view, include_visibility=visibility_cache is not None)
    camera = orbit_state_to_camera(state, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    timer = _ViewerGpuTimer(device)
    with torch.no_grad():
        timer.mark("start")
        render_buffers = render_rgbd(asset.scene, camera)
        timer.mark("after_render_rgbd")
        gbuffer = make_pseudo_gbuffer(render_buffers, camera)
        timer.mark("after_gbuffer")

        lights = None
        light_info: dict[str, object] = {}
        if requirements["world_lights"]:
            world_lights, light_info = make_asset_scaled_world_lights(
                asset.scene.means,
                count=settings.num_lights,
                seed=settings.light_seed,
                device=device,
            )
            lights = world_lights_to_camera_lights(world_lights, camera)
        timer.mark("after_world_lights")

        lambertian = None
        restir = None
        if requirements["diffuse_restir"]:
            if lights is None:
                raise RuntimeError("Internal viewer error: diffuse rendering requires camera-space lights.")
            diffuse_frame = evaluate_restir_frame_from_gbuffer(
                gbuffer,
                camera,
                lights,
                frame_index=frame_index,
                settings=_diffuse_backend_settings(settings, frame_index),
            )
            proposal_confidence = compute_geometric_proposal_distribution(gbuffer, lights).max(dim=-1).values
            lambertian = diffuse_frame.reference
            restir = ViewerRestirResult(
                reference=diffuse_frame.reference,
                geometric_mc=_require_geometric_mc(diffuse_frame),
                initial_ris=diffuse_frame.initial,
                proposal_confidence=proposal_confidence,
            )
        timer.mark("after_diffuse_restir")

        blinn_phong = None
        if requirements["blinn_phong"]:
            if lights is None:
                raise RuntimeError("Internal viewer error: Blinn-Phong rendering requires camera-space lights.")
            blinn_phong = shade_deferred_blinn_phong(
                gbuffer,
                lights,
                ambient=settings.ambient,
                specular_strength=settings.specular_strength,
                shininess=settings.shininess,
            )
        timer.mark("after_blinn_phong")
        timer.mark("after_proposal_confidence")

        visibility_result = None
        if visibility_cache is not None:
            visibility_lights = world_lights_to_camera_lights(visibility_cache.world_lights, camera)
            visibility_settings = _visibility_backend_settings(settings, frame_index)
            if include_visibility_reference:
                visibility_frame = evaluate_restir_frame_from_gbuffer(
                    gbuffer,
                    camera,
                    visibility_lights,
                    frame_index=frame_index,
                    settings=visibility_settings,
                    shadow_bundle=visibility_cache.shadow_bundle,
                )
                visibility_reference = visibility_frame.reference
                visibility_error = torch.abs(visibility_frame.initial.contribution_rgb - visibility_reference.diffuse_rgb).mean(dim=-1)
            else:
                visibility_frame = evaluate_restir_display_frame_from_gbuffer(
                    gbuffer,
                    camera,
                    visibility_lights,
                    frame_index=frame_index,
                    settings=visibility_settings,
                    shadow_bundle=visibility_cache.shadow_bundle,
                )
                visibility_reference = None
                visibility_error = None
            visibility_result = ViewerVisibilityResult(
                reference=visibility_reference,
                geometric_mc=visibility_frame.geometric_mc,
                initial_ris=visibility_frame.initial,
                error=visibility_error,
            )
        timer.mark("after_visibility")
    timings = timer.to_timings()
    render_ms = (time.perf_counter() - start) * 1000.0
    timings = replace(timings, total_wall_ms=render_ms)
    valid_pixels = int((gbuffer.valid_mask & gbuffer.normal_mask).sum().detach().cpu())
    return ViewerRenderResult(
        frame_index=frame_index,
        state=state,
        gbuffer=gbuffer,
        lambertian=lambertian,
        blinn_phong=blinn_phong,
        restir=restir,
        visibility=visibility_result,
        valid_pixels=valid_pixels,
        render_ms=render_ms,
        light_info=light_info,
        timings=timings,
        computed_views=viewer_computed_views(requirements),
    )


def reset_state_from_frame(
    asset: ViewerAsset,
    frame_index: int,
    width: int,
    height: int,
    device: torch.device,
) -> OrbitCameraState:
    camera = _scale_camera(asset.frame_cameras[frame_index], width, height)
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
        paths["visibility_ris"] = output_dir / "current_visibility_ris.png"
        if result.visibility.reference is not None and result.visibility.error is not None:
            paths["visibility_reference"] = output_dir / "current_visibility_reference.png"
            paths["visibility_error"] = output_dir / "current_visibility_error.png"
    save_orbit_camera_config(result.state, paths["camera"], metadata=metadata)
    imageio.imwrite(paths["rgb"], to_u8_rgb(result.gbuffer.rgb))
    imageio.imwrite(paths["alpha"], (result.gbuffer.alpha.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8))
    imageio.imwrite(paths["normal"], to_u8_normal(result.gbuffer.normal_cam, result.gbuffer.normal_mask))
    imageio.imwrite(paths["blinn_phong"], to_u8_rgb(_require_blinn_phong(result).composite_rgb))
    if result.visibility is not None:
        imageio.imwrite(paths["visibility_ris"], to_u8_rgb(result.visibility.initial_ris.composite_rgb))
        if result.visibility.reference is not None and result.visibility.error is not None:
            imageio.imwrite(paths["visibility_reference"], to_u8_rgb(result.visibility.reference.composite_rgb))
            imageio.imwrite(paths["visibility_error"], to_u8_scalar(result.visibility.error, result.visibility.reference.valid_mask))
    return {key: str(path) for key, path in paths.items()}


def view_image(result: ViewerRenderResult, view: str) -> tuple[str, np.ndarray]:
    gbuffer = result.gbuffer
    if view == "rgb":
        return "RGB", to_u8_rgb(gbuffer.rgb)
    if view == "alpha":
        return "Alpha", (gbuffer.alpha.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
    if view == "depth":
        return "Depth", to_u8_scalar(gbuffer.depth, gbuffer.valid_mask)
    if view == "normal":
        return "Normal", to_u8_normal(gbuffer.normal_cam, gbuffer.normal_mask)
    if view == "lambertian":
        return "Lambertian", to_u8_rgb(_require_lambertian(result).composite_rgb)
    if view == "blinn_phong":
        return "Blinn-Phong", to_u8_rgb(_require_blinn_phong(result).composite_rgb)
    raise ValueError(f"Unsupported viewer view '{view}'.")


def viewer_save_metadata(settings: ViewerSettings, result: ViewerRenderResult, asset: ViewerAsset) -> dict[str, object]:
    return {
        "phase": "phase25_interactive_viewer",
        "asset_label": asset.label,
        "source_path": str(asset.source_path),
        "asset_metadata": asset.metadata,
        "frame_index": result.frame_index,
        "frame_label": asset.frame_labels[result.frame_index],
        "valid_pixels": result.valid_pixels,
        "render_ms": result.render_ms,
        "timings": result.timings.as_dict(),
        "computed_views": list(result.computed_views),
        "width": settings.width,
        "height": settings.height,
        "num_lights": settings.num_lights,
        "light_seed": settings.light_seed,
        "restir_candidate_count": settings.restir_candidate_count,
        "restir_candidate_seed": settings.restir_candidate_seed,
        "restir_selection_seed": settings.restir_selection_seed,
        "visibility_num_lights": settings.visibility_num_lights,
        "visibility_light_seed": settings.visibility_light_seed,
        "visibility_candidate_count": settings.visibility_candidate_count,
        "visibility_candidate_seed": settings.visibility_candidate_seed,
        "visibility_selection_seed": settings.visibility_selection_seed,
        "visibility_shadow_resolution": settings.visibility_shadow_resolution,
        "visibility_shadow_bias_scale": settings.visibility_shadow_bias_scale,
        "visibility_shadow_alpha_threshold": settings.visibility_shadow_alpha_threshold,
        "visibility_shadow_pcf_radius": settings.visibility_shadow_pcf_radius,
        "ambient": settings.ambient,
        "specular_strength": settings.specular_strength,
        "shininess": settings.shininess,
        "light_info": result.light_info,
        "display_backend_renderer": "restir_gs.restir.renderer.evaluate_restir_display_frame_from_gbuffer",
        "evaluation_backend_renderer": "restir_gs.restir.renderer.evaluate_restir_frame_from_gbuffer",
    }


def _require_lambertian(result: ViewerRenderResult) -> LightingBuffers:
    if result.lambertian is None:
        raise RuntimeError("Lambertian view was not computed; render this frame with required_view='lambertian'.")
    return result.lambertian


def _require_blinn_phong(result: ViewerRenderResult) -> LightingBuffers:
    if result.blinn_phong is None:
        raise RuntimeError("Blinn-Phong view was not computed; render this frame with required_view='blinn_phong'.")
    return result.blinn_phong


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


def _scale_camera(camera: PinholeCamera, width: int, height: int) -> PinholeCamera:
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive output size, got {width}x{height}")
    sx = float(width) / float(camera.width)
    sy = float(height) / float(camera.height)
    intrinsics = camera.intrinsics.clone()
    intrinsics[:, 0, :] *= sx
    intrinsics[:, 1, :] *= sy
    return PinholeCamera(viewmats=camera.viewmats.clone(), intrinsics=intrinsics, width=width, height=height)


def _diffuse_backend_settings(settings: ViewerSettings, frame_index: int) -> RestirRenderSettings:
    return RestirRenderSettings(
        target_mode="diffuse",
        candidate_count=settings.restir_candidate_count,
        candidate_seed_base=settings.restir_candidate_seed - frame_index,
        initial_selection_seed_base=settings.restir_selection_seed - frame_index,
        temporal_selection_seed_base=settings.restir_selection_seed - frame_index,
        ambient=settings.ambient,
        include_mc_baseline=True,
        visibility_shadow_resolution=settings.visibility_shadow_resolution,
        visibility_shadow_bias_scale=settings.visibility_shadow_bias_scale,
        visibility_shadow_alpha_threshold=settings.visibility_shadow_alpha_threshold,
        visibility_shadow_pcf_radius=settings.visibility_shadow_pcf_radius,
    )


def _visibility_backend_settings(settings: ViewerSettings, frame_index: int) -> RestirRenderSettings:
    return RestirRenderSettings(
        target_mode="visibility",
        candidate_count=settings.visibility_candidate_count,
        candidate_seed_base=settings.visibility_candidate_seed - frame_index,
        initial_selection_seed_base=settings.visibility_selection_seed - frame_index,
        temporal_selection_seed_base=settings.visibility_selection_seed - frame_index,
        ambient=settings.ambient,
        include_mc_baseline=True,
        visibility_shadow_resolution=settings.visibility_shadow_resolution,
        visibility_shadow_bias_scale=settings.visibility_shadow_bias_scale,
        visibility_shadow_alpha_threshold=settings.visibility_shadow_alpha_threshold,
        visibility_shadow_pcf_radius=settings.visibility_shadow_pcf_radius,
    )


def _require_geometric_mc(result: RestirFrameResult | RestirDisplayFrameResult) -> LightingEstimatorBuffers:
    if result.geometric_mc is None:
        raise RuntimeError("Expected geometric MC baseline buffers to be present.")
    return result.geometric_mc
