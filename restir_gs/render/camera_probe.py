from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import torch

from restir_gs.render.gsplat_renderer import RenderBuffers
from restir_gs.render.ply_loader import AssetCameraInfo, make_asset_camera
from restir_gs.render.synthetic_scene import PinholeCamera


@dataclass(frozen=True)
class CameraProbeCandidate:
    index: int
    camera: PinholeCamera
    info: AssetCameraInfo


@dataclass(frozen=True)
class CameraProbeScore:
    score: float
    valid_pixels: int
    coverage: float
    central_coverage: float
    border_coverage: float
    brightness: float


def parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Expected at least one float value, got {text!r}")
    return values


def make_probe_camera_candidates(
    means: torch.Tensor,
    yaw_values: list[float],
    pitch_values: list[float],
    radius_scales: list[float],
    width: int = 128,
    height: int = 128,
    bbox_percentile: float = 0.98,
    focal: float | None = None,
) -> list[CameraProbeCandidate]:
    candidates: list[CameraProbeCandidate] = []
    index = 0
    for radius_scale in radius_scales:
        for pitch in pitch_values:
            for yaw in yaw_values:
                camera, info = make_asset_camera(
                    means,
                    width=width,
                    height=height,
                    focal=focal,
                    radius_scale=radius_scale,
                    bbox_percentile=bbox_percentile,
                    yaw_degrees=yaw,
                    pitch_degrees=pitch,
                )
                candidates.append(CameraProbeCandidate(index=index, camera=camera, info=info))
                index += 1
    if not candidates:
        raise ValueError("Expected at least one camera probe candidate.")
    return candidates


def score_render_buffers(
    buffers: RenderBuffers,
    alpha_threshold: float = 1e-4,
    border_pixels: int = 8,
) -> CameraProbeScore:
    rgb = buffers.rgb.detach()
    depth = buffers.depth.detach()
    alpha = buffers.alpha.detach()
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape [H,W,3], got {tuple(rgb.shape)}")
    if depth.shape != rgb.shape[:2] or alpha.shape != rgb.shape[:2]:
        raise ValueError("Expected depth and alpha shapes to match RGB image shape.")

    height, width = depth.shape
    valid = (alpha > alpha_threshold) & torch.isfinite(depth) & (depth > 0.0)
    valid_pixels = int(valid.sum().detach().cpu())
    total_pixels = max(height * width, 1)
    coverage = valid_pixels / float(total_pixels)

    y0 = height // 4
    y1 = height - y0
    x0 = width // 4
    x1 = width - x0
    central = valid[y0:y1, x0:x1]
    central_coverage = float(central.float().mean().detach().cpu()) if central.numel() > 0 else 0.0

    border = _border_mask(height, width, border_pixels, device=valid.device)
    border_coverage = float(valid[border].float().mean().detach().cpu()) if bool(border.any()) else 0.0

    if valid_pixels > 0:
        luminance = _luminance(rgb)
        brightness = float(luminance[valid].mean().detach().cpu())
    else:
        brightness = 0.0

    score = coverage + 0.75 * central_coverage + 0.15 * brightness - 0.5 * border_coverage
    return CameraProbeScore(
        score=float(score),
        valid_pixels=valid_pixels,
        coverage=float(coverage),
        central_coverage=float(central_coverage),
        border_coverage=float(border_coverage),
        brightness=float(brightness),
    )


def select_best_candidate(scores: list[CameraProbeScore]) -> int:
    if not scores:
        raise ValueError("Expected at least one camera probe score.")
    finite_indices = [index for index, score in enumerate(scores) if math.isfinite(score.score)]
    if not finite_indices:
        raise RuntimeError("All camera probe candidates have non-finite scores.")
    best_index = max(finite_indices, key=lambda index: scores[index].score)
    if scores[best_index].valid_pixels <= 0:
        raise RuntimeError("All camera probe candidates have zero valid pixels.")
    return best_index


def camera_config_payload(
    camera: PinholeCamera,
    info: AssetCameraInfo,
    score: CameraProbeScore | None = None,
    candidate_index: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": 1,
        "camera": {
            "viewmat": camera.viewmats[0].detach().cpu().tolist(),
            "intrinsics": camera.intrinsics[0].detach().cpu().tolist(),
            "width": camera.width,
            "height": camera.height,
        },
        "camera_info": {
            "target": info.target,
            "eye": info.eye,
            "bbox_min": info.bbox_min,
            "bbox_max": info.bbox_max,
            "bbox_diagonal": info.bbox_diagonal,
            "radius": info.radius,
            "focal": info.focal,
            "bbox_percentile": info.bbox_percentile,
            "radius_scale": info.radius_scale,
            "yaw_degrees": info.yaw_degrees,
            "pitch_degrees": info.pitch_degrees,
        },
    }
    if candidate_index is not None:
        payload["candidate_index"] = int(candidate_index)
    if score is not None:
        payload["score"] = score_to_dict(score)
    return payload


def load_camera_config(path: str | Path, device: torch.device | str = "cuda") -> PinholeCamera:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return camera_from_config_payload(data, device=device)


def camera_from_config_payload(payload: dict[str, object], device: torch.device | str = "cuda") -> PinholeCamera:
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"Unsupported camera config version: {payload.get('version')}")
    camera_data = payload["camera"]
    if not isinstance(camera_data, dict):
        raise ValueError("Camera config missing camera object.")
    viewmat = torch.tensor(camera_data["viewmat"], dtype=torch.float32, device=device)[None]
    intrinsics = torch.tensor(camera_data["intrinsics"], dtype=torch.float32, device=device)[None]
    return PinholeCamera(
        viewmats=viewmat,
        intrinsics=intrinsics,
        width=int(camera_data["width"]),
        height=int(camera_data["height"]),
    )


def score_to_dict(score: CameraProbeScore) -> dict[str, float | int]:
    return {
        "score": score.score,
        "valid_pixels": score.valid_pixels,
        "coverage": score.coverage,
        "central_coverage": score.central_coverage,
        "border_coverage": score.border_coverage,
        "brightness": score.brightness,
    }


def _border_mask(height: int, width: int, border_pixels: int, device: torch.device) -> torch.Tensor:
    if border_pixels <= 0:
        return torch.zeros((height, width), dtype=torch.bool, device=device)
    border = min(border_pixels, max(height, width))
    ys, xs = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return (ys < border) | (ys >= height - border) | (xs < border) | (xs >= width - border)


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.2126, 0.7152, 0.0722], dtype=rgb.dtype, device=rgb.device)
    return torch.sum(rgb * weights, dim=-1)
