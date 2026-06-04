from __future__ import annotations

import math

import torch

from restir_gs.render.synthetic_scene import PinholeCamera


def look_at_viewmat(
    eye: torch.Tensor,
    target: torch.Tensor,
    up_hint: torch.Tensor | None = None,
) -> torch.Tensor:
    """Create a world-to-camera look-at matrix with camera +Z pointing forward."""
    if eye.shape != (3,) or target.shape != (3,):
        raise ValueError(f"Expected eye and target shape [3], got {tuple(eye.shape)} and {tuple(target.shape)}")

    dtype = eye.dtype
    device = eye.device
    if up_hint is None:
        up_hint = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)
    else:
        up_hint = up_hint.to(device=device, dtype=dtype)

    forward = target - eye
    forward = forward / torch.linalg.norm(forward).clamp_min(1e-8)
    right = torch.cross(up_hint, forward, dim=0)
    right = right / torch.linalg.norm(right).clamp_min(1e-8)
    up = torch.cross(forward, right, dim=0)

    view = torch.eye(4, dtype=dtype, device=device)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = forward
    view[0, 3] = -torch.dot(right, eye)
    view[1, 3] = -torch.dot(up, eye)
    view[2, 3] = -torch.dot(forward, eye)
    return view


def make_orbit_camera_sequence(
    width: int = 128,
    height: int = 128,
    frame_count: int = 5,
    yaw_degrees: tuple[float, float] = (-4.0, 4.0),
    target: tuple[float, float, float] = (0.0, 0.0, 2.35),
    radius: float | None = None,
    focal: float | None = None,
    device: torch.device | str = "cuda",
) -> list[PinholeCamera]:
    """Create a small yaw-orbit camera sequence around a fixed target."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Expected positive image size, got {width}x{height}")
    if frame_count <= 0:
        raise ValueError(f"Expected positive frame count, got {frame_count}")
    if focal is None:
        focal = float(width) * 1.25

    target_tensor = torch.tensor(target, dtype=torch.float32, device=device)
    if radius is None:
        radius = float(torch.linalg.norm(target_tensor).detach().cpu())

    if frame_count == 1:
        yaws = torch.tensor([sum(yaw_degrees) * 0.5], dtype=torch.float32, device=device)
    else:
        yaws = torch.linspace(float(yaw_degrees[0]), float(yaw_degrees[1]), frame_count, dtype=torch.float32, device=device)

    intrinsics = torch.tensor(
        [
            [
                [focal, 0.0, width * 0.5],
                [0.0, focal, height * 0.5],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    cameras: list[PinholeCamera] = []
    for yaw in yaws:
        yaw_rad = float(yaw.detach().cpu()) * math.pi / 180.0
        eye = target_tensor + torch.tensor(
            [math.sin(yaw_rad) * radius, 0.0, -math.cos(yaw_rad) * radius],
            dtype=torch.float32,
            device=device,
        )
        viewmat = look_at_viewmat(eye, target_tensor)[None]
        cameras.append(PinholeCamera(viewmats=viewmat, intrinsics=intrinsics.clone(), width=width, height=height))
    return cameras
