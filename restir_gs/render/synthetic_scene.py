from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticGaussians:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor


@dataclass(frozen=True)
class PinholeCamera:
    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    width: int
    height: int


def make_synthetic_gaussians(device: torch.device | str = "cuda") -> SyntheticGaussians:
    """Create a small deterministic Gaussian set in front of the camera."""
    return SyntheticGaussians(
        means=torch.tensor(
            [
                [0.00, 0.00, 2.00],
                [-0.45, -0.05, 2.25],
                [0.45, 0.05, 2.25],
                [0.00, 0.42, 2.55],
                [0.00, -0.48, 2.45],
            ],
            dtype=torch.float32,
            device=device,
        ),
        quats=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        scales=torch.tensor(
            [
                [0.16, 0.16, 0.16],
                [0.13, 0.13, 0.13],
                [0.13, 0.13, 0.13],
                [0.12, 0.12, 0.12],
                [0.12, 0.12, 0.12],
            ],
            dtype=torch.float32,
            device=device,
        ),
        opacities=torch.tensor([0.95, 0.90, 0.90, 0.85, 0.85], dtype=torch.float32, device=device),
        colors=torch.tensor(
            [
                [1.00, 0.25, 0.15],
                [0.20, 0.85, 1.00],
                [0.40, 1.00, 0.35],
                [1.00, 0.85, 0.20],
                [0.75, 0.45, 1.00],
            ],
            dtype=torch.float32,
            device=device,
        ),
    )


def make_pinhole_camera(
    width: int = 128,
    height: int = 128,
    focal: float | None = None,
    device: torch.device | str = "cuda",
) -> PinholeCamera:
    """Create a fixed identity-view pinhole camera."""
    if focal is None:
        focal = float(width) * 1.25

    viewmats = torch.eye(4, dtype=torch.float32, device=device)[None]
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
    return PinholeCamera(viewmats=viewmats, intrinsics=intrinsics, width=width, height=height)

