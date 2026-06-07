from __future__ import annotations

from dataclasses import replace

import torch

from restir_gs.render.orbit_camera import OrbitCameraState, orbit_state_eye, orbit_state_to_camera


def camera_move_step(state: OrbitCameraState, scale: float = 0.08, min_step: float = 1e-3) -> float:
    if scale <= 0.0:
        raise ValueError(f"Expected positive movement scale, got {scale}")
    if min_step <= 0.0:
        raise ValueError(f"Expected positive min_step, got {min_step}")
    return max(float(state.radius) * float(scale), float(min_step))


def camera_state_translate_local(
    state: OrbitCameraState,
    *,
    delta_right: float = 0.0,
    delta_up: float = 0.0,
    delta_forward: float = 0.0,
    device: torch.device | str = "cpu",
) -> OrbitCameraState:
    camera = orbit_state_to_camera(state, device=device)
    viewmat = camera.viewmats[0].detach().cpu()
    right = viewmat[0, :3]
    up = viewmat[1, :3]
    forward = viewmat[2, :3]
    delta = right * float(delta_right) + up * float(delta_up) + forward * float(delta_forward)
    target = torch.tensor(state.target, dtype=torch.float32) + delta
    return replace(
        state,
        target=(float(target[0]), float(target[1]), float(target[2])),
    )


def camera_state_look(
    state: OrbitCameraState,
    *,
    delta_yaw_degrees: float,
    delta_pitch_degrees: float,
    pitch_limit_degrees: float = 89.0,
) -> OrbitCameraState:
    if pitch_limit_degrees <= 0.0 or pitch_limit_degrees >= 90.0:
        raise ValueError(f"Expected pitch_limit_degrees in (0,90), got {pitch_limit_degrees}")
    eye = orbit_state_eye(state, device="cpu").detach().cpu()
    pitch = max(min(state.pitch_degrees + float(delta_pitch_degrees), pitch_limit_degrees), -pitch_limit_degrees)
    rotated = replace(state, target=(0.0, 0.0, 0.0), yaw_degrees=state.yaw_degrees + float(delta_yaw_degrees), pitch_degrees=pitch)
    offset = orbit_state_eye(rotated, device="cpu").detach().cpu()
    target = eye - offset
    return replace(
        state,
        target=(float(target[0]), float(target[1]), float(target[2])),
        yaw_degrees=rotated.yaw_degrees,
        pitch_degrees=rotated.pitch_degrees,
    )
