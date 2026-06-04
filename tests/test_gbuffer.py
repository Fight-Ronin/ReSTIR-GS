from __future__ import annotations

from types import SimpleNamespace

import torch

from restir_gs.render.gbuffer import (
    estimate_normals_from_position,
    make_pseudo_gbuffer,
    unproject_depth_to_camera,
)
from restir_gs.render.synthetic_scene import PinholeCamera


def make_intrinsics(width: int, height: int, focal: float = 2.0) -> torch.Tensor:
    return torch.tensor(
        [
            [focal, 0.0, float(width // 2)],
            [0.0, focal, float(height // 2)],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_unproject_depth_to_camera_uses_pinhole_intrinsics() -> None:
    depth = torch.full((3, 3), 2.0, dtype=torch.float32)
    intrinsics = make_intrinsics(width=3, height=3, focal=2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)

    position = unproject_depth_to_camera(depth, intrinsics, valid)

    assert torch.allclose(position[1, 1], torch.tensor([0.0, 0.0, 2.0]))
    assert torch.allclose(position[1, 2], torch.tensor([1.0, 0.0, 2.0]))
    assert torch.allclose(position[2, 1], torch.tensor([0.0, 1.0, 2.0]))


def test_estimate_normals_from_position_returns_front_facing_plane_normals() -> None:
    depth = torch.full((5, 5), 2.0, dtype=torch.float32)
    intrinsics = make_intrinsics(width=5, height=5, focal=2.0)
    valid = torch.ones_like(depth, dtype=torch.bool)
    position = unproject_depth_to_camera(depth, intrinsics, valid)

    normal, normal_mask = estimate_normals_from_position(position, valid)

    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    assert int(normal_mask.sum()) == 9
    assert torch.allclose(normal[normal_mask], expected.expand_as(normal[normal_mask]), atol=1e-6)
    assert not bool(normal_mask[0].any())
    assert not bool(normal_mask[-1].any())
    assert not bool(normal_mask[:, 0].any())
    assert not bool(normal_mask[:, -1].any())


def test_make_pseudo_gbuffer_zeroes_invalid_positions_and_normals() -> None:
    depth = torch.full((5, 5), 2.0, dtype=torch.float32)
    alpha = torch.ones((5, 5), dtype=torch.float32)
    depth[1, 1] = 0.0
    alpha[2, 2] = 0.0

    render_buffers = SimpleNamespace(
        rgb=torch.zeros((5, 5, 3), dtype=torch.float32),
        depth=depth,
        alpha=alpha,
    )
    intrinsics = make_intrinsics(width=5, height=5, focal=2.0)
    camera = PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=intrinsics[None],
        width=5,
        height=5,
    )

    gbuffer = make_pseudo_gbuffer(render_buffers, camera, alpha_threshold=1e-4)

    assert not bool(gbuffer.valid_mask[1, 1])
    assert not bool(gbuffer.valid_mask[2, 2])
    assert torch.allclose(gbuffer.position_cam[1, 1], torch.zeros(3))
    assert torch.allclose(gbuffer.position_cam[2, 2], torch.zeros(3))
    assert not bool(gbuffer.normal_mask[1, 1])
    assert not bool(gbuffer.normal_mask[2, 2])
    assert not bool(gbuffer.normal_mask[1, 2])
    assert not bool(gbuffer.normal_mask[2, 3])
    assert torch.allclose(gbuffer.normal_cam[1, 1], torch.zeros(3))
    assert torch.allclose(gbuffer.normal_cam[2, 2], torch.zeros(3))
