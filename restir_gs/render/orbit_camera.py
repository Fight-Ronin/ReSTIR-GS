from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from restir_gs.render.camera_sequence import look_at_viewmat
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.render.transforms_loader import camera_to_config_payload


@dataclass(frozen=True)
class OrbitCameraState:
    target: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    radius: float
    focal_scale: float
    width: int
    height: int


def orbit_state_to_camera(state: OrbitCameraState, device: torch.device | str = "cuda") -> PinholeCamera:
    _check_state(state)
    target = torch.tensor(state.target, dtype=torch.float32, device=device)
    eye = orbit_state_eye(state, device=device)
    focal = float(state.width) * float(state.focal_scale)
    viewmat = look_at_viewmat(eye, target)[None]
    intrinsics = torch.tensor(
        [[[focal, 0.0, state.width * 0.5], [0.0, focal, state.height * 0.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    return PinholeCamera(viewmats=viewmat, intrinsics=intrinsics, width=state.width, height=state.height)


def orbit_state_eye(state: OrbitCameraState, device: torch.device | str = "cpu") -> torch.Tensor:
    _check_state(state)
    yaw = math.radians(float(state.yaw_degrees))
    pitch = math.radians(float(state.pitch_degrees))
    cos_pitch = math.cos(pitch)
    offset = torch.tensor(
        [
            math.sin(yaw) * cos_pitch * state.radius,
            math.sin(pitch) * state.radius,
            -math.cos(yaw) * cos_pitch * state.radius,
        ],
        dtype=torch.float32,
        device=device,
    )
    target = torch.tensor(state.target, dtype=torch.float32, device=device)
    return target + offset


def orbit_state_from_camera(
    camera: PinholeCamera,
    target: tuple[float, float, float] | torch.Tensor | None = None,
    fallback_radius: float = 1.0,
) -> OrbitCameraState:
    if fallback_radius <= 0.0:
        raise ValueError(f"Expected positive fallback_radius, got {fallback_radius}")
    eye = camera_eye(camera).detach().cpu()
    if target is None:
        forward = camera_forward(camera).detach().cpu()
        target_tensor = eye + forward * float(fallback_radius)
    else:
        target_tensor = torch.as_tensor(target, dtype=torch.float32).detach().cpu()
        if tuple(target_tensor.shape) != (3,):
            raise ValueError(f"Expected target shape [3], got {tuple(target_tensor.shape)}")

    offset = eye - target_tensor
    radius = float(torch.linalg.norm(offset).item())
    if radius <= 1e-8:
        radius = float(fallback_radius)
        offset = -camera_forward(camera).detach().cpu() * radius
        target_tensor = eye - offset

    pitch = math.degrees(math.asin(max(min(float(offset[1] / radius), 1.0), -1.0)))
    yaw = math.degrees(math.atan2(float(offset[0]), float(-offset[2])))
    fx = float(camera.intrinsics[0, 0, 0].detach().cpu())
    return OrbitCameraState(
        target=_tuple3(target_tensor),
        yaw_degrees=float(yaw),
        pitch_degrees=float(pitch),
        radius=float(radius),
        focal_scale=fx / float(camera.width),
        width=int(camera.width),
        height=int(camera.height),
    )


def orbit_state_orbit(
    state: OrbitCameraState,
    delta_yaw_degrees: float,
    delta_pitch_degrees: float,
    pitch_limit_degrees: float = 89.0,
) -> OrbitCameraState:
    _check_state(state)
    if pitch_limit_degrees <= 0.0 or pitch_limit_degrees >= 90.0:
        raise ValueError(f"Expected pitch_limit_degrees in (0,90), got {pitch_limit_degrees}")
    pitch = max(min(state.pitch_degrees + float(delta_pitch_degrees), pitch_limit_degrees), -pitch_limit_degrees)
    return OrbitCameraState(
        target=state.target,
        yaw_degrees=state.yaw_degrees + float(delta_yaw_degrees),
        pitch_degrees=pitch,
        radius=state.radius,
        focal_scale=state.focal_scale,
        width=state.width,
        height=state.height,
    )


def orbit_state_dolly(state: OrbitCameraState, scale: float, min_radius: float = 1e-3) -> OrbitCameraState:
    _check_state(state)
    if scale <= 0.0:
        raise ValueError(f"Expected positive dolly scale, got {scale}")
    if min_radius <= 0.0:
        raise ValueError(f"Expected positive min_radius, got {min_radius}")
    return OrbitCameraState(
        target=state.target,
        yaw_degrees=state.yaw_degrees,
        pitch_degrees=state.pitch_degrees,
        radius=max(state.radius * float(scale), float(min_radius)),
        focal_scale=state.focal_scale,
        width=state.width,
        height=state.height,
    )


def orbit_state_pan(
    state: OrbitCameraState,
    delta_right: float,
    delta_up: float,
    device: torch.device | str = "cpu",
) -> OrbitCameraState:
    _check_state(state)
    camera = orbit_state_to_camera(state, device=device)
    right = camera.viewmats[0, 0, :3].detach().cpu()
    up = camera.viewmats[0, 1, :3].detach().cpu()
    target = torch.tensor(state.target, dtype=torch.float32)
    target = target + right * float(delta_right) + up * float(delta_up)
    return OrbitCameraState(
        target=_tuple3(target),
        yaw_degrees=state.yaw_degrees,
        pitch_degrees=state.pitch_degrees,
        radius=state.radius,
        focal_scale=state.focal_scale,
        width=state.width,
        height=state.height,
    )


def orbit_camera_config_payload(state: OrbitCameraState, metadata: dict[str, object] | None = None) -> dict[str, object]:
    camera = orbit_state_to_camera(state, device="cpu")
    payload = camera_to_config_payload(camera, metadata=metadata)
    payload["orbit_camera_state"] = {
        "target": state.target,
        "yaw_degrees": state.yaw_degrees,
        "pitch_degrees": state.pitch_degrees,
        "radius": state.radius,
        "focal_scale": state.focal_scale,
        "width": state.width,
        "height": state.height,
    }
    return payload


def save_orbit_camera_config(
    state: OrbitCameraState,
    path: str | Path,
    metadata: dict[str, object] | None = None,
) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(orbit_camera_config_payload(state, metadata=metadata), indent=2), encoding="utf-8")


def camera_eye(camera: PinholeCamera) -> torch.Tensor:
    _check_camera(camera)
    inv = torch.linalg.inv(camera.viewmats[0])
    return inv[:3, 3]


def camera_forward(camera: PinholeCamera) -> torch.Tensor:
    _check_camera(camera)
    forward = camera.viewmats[0, 2, :3]
    return forward / torch.linalg.norm(forward).clamp_min(1e-8)


def _check_camera(camera: PinholeCamera) -> None:
    if camera.viewmats.shape != (1, 4, 4):
        raise ValueError(f"Expected viewmats shape [1,4,4], got {tuple(camera.viewmats.shape)}")
    if camera.intrinsics.shape != (1, 3, 3):
        raise ValueError(f"Expected intrinsics shape [1,3,3], got {tuple(camera.intrinsics.shape)}")
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError(f"Expected positive camera size, got {camera.width}x{camera.height}")


def _check_state(state: OrbitCameraState) -> None:
    if state.width <= 0 or state.height <= 0:
        raise ValueError(f"Expected positive image size, got {state.width}x{state.height}")
    if state.radius <= 0.0:
        raise ValueError(f"Expected positive radius, got {state.radius}")
    if state.focal_scale <= 0.0:
        raise ValueError(f"Expected positive focal_scale, got {state.focal_scale}")
    if len(state.target) != 3:
        raise ValueError(f"Expected target length 3, got {state.target}")


def _tuple3(values: torch.Tensor) -> tuple[float, float, float]:
    data = values.detach().cpu().tolist()
    return (float(data[0]), float(data[1]), float(data[2]))
