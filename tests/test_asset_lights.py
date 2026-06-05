from __future__ import annotations

import pytest
import torch

from restir_gs.lighting.asset_lights import (
    WorldPointLights,
    make_asset_scaled_point_lights,
    make_asset_scaled_world_lights,
    world_lights_to_camera_lights,
)
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera


def make_gbuffer(scale: float = 1.0, valid: bool = True) -> GBuffer:
    height, width = 2, 2
    rgb = torch.ones((height, width, 3), dtype=torch.float32)
    depth = torch.ones((height, width), dtype=torch.float32)
    alpha = torch.ones((height, width), dtype=torch.float32)
    position = torch.tensor(
        [
            [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.5]],
            [[-1.0, 1.0, 3.0], [1.0, 1.0, 3.5]],
        ],
        dtype=torch.float32,
    )
    position = position * float(scale)
    normal = torch.zeros((height, width, 3), dtype=torch.float32)
    normal[..., 2] = 1.0
    mask = torch.full((height, width), valid, dtype=torch.bool)
    return GBuffer(
        rgb=rgb,
        depth=depth,
        alpha=alpha,
        position_cam=position,
        normal_cam=normal,
        valid_mask=mask,
        normal_mask=mask.clone(),
    )


def test_asset_scaled_lights_are_deterministic() -> None:
    gbuffer = make_gbuffer()

    lights_a, info_a = make_asset_scaled_point_lights(gbuffer, count=4, seed=123, device="cpu")
    lights_b, info_b = make_asset_scaled_point_lights(gbuffer, count=4, seed=123, device="cpu")

    assert torch.equal(lights_a.positions_cam, lights_b.positions_cam)
    assert torch.equal(lights_a.colors, lights_b.colors)
    assert torch.equal(lights_a.intensities, lights_b.intensities)
    assert info_a == info_b


def test_asset_scaled_lights_are_finite() -> None:
    lights, info = make_asset_scaled_point_lights(make_gbuffer(), count=8, seed=1, device="cpu")

    assert torch.isfinite(lights.positions_cam).all()
    assert torch.isfinite(lights.colors).all()
    assert torch.isfinite(lights.intensities).all()
    assert lights.positions_cam.shape == (8, 3)
    assert lights.colors.shape == (8, 3)
    assert lights.intensities.shape == (8,)
    assert info["mode"] == "asset_scaled_camera_space"


def test_asset_light_intensity_grows_with_scene_scale_squared() -> None:
    small, _ = make_asset_scaled_point_lights(make_gbuffer(scale=1.0), count=4, seed=3, device="cpu")
    large, _ = make_asset_scaled_point_lights(make_gbuffer(scale=2.0), count=4, seed=3, device="cpu")

    ratio = large.intensities[0] / small.intensities[0]
    assert torch.allclose(ratio, torch.tensor(4.0), atol=1e-6)


def test_asset_scaled_lights_fail_without_valid_pixels() -> None:
    with pytest.raises(RuntimeError, match="without valid G-buffer positions"):
        make_asset_scaled_point_lights(make_gbuffer(valid=False), count=4, seed=1, device="cpu")


def test_asset_scaled_lights_reject_nonpositive_count() -> None:
    with pytest.raises(ValueError, match="positive light count"):
        make_asset_scaled_point_lights(make_gbuffer(), count=0, seed=1, device="cpu")


def test_asset_scaled_world_lights_are_deterministic() -> None:
    means = make_means()

    lights_a, info_a = make_asset_scaled_world_lights(means, count=8, seed=123, device="cpu")
    lights_b, info_b = make_asset_scaled_world_lights(means, count=8, seed=123, device="cpu")

    assert torch.equal(lights_a.positions_world, lights_b.positions_world)
    assert torch.equal(lights_a.colors, lights_b.colors)
    assert torch.equal(lights_a.intensities, lights_b.intensities)
    assert info_a == info_b
    assert info_a["light_space"] == "world"
    assert info_a["light_policy"] == "asset_scaled_spherical_shell"


def test_asset_scaled_world_lights_are_finite_and_shaped() -> None:
    lights, info = make_asset_scaled_world_lights(make_means(), count=5, seed=7, device="cpu")

    assert lights.positions_world.shape == (5, 3)
    assert lights.colors.shape == (5, 3)
    assert lights.intensities.shape == (5,)
    assert torch.isfinite(lights.positions_world).all()
    assert torch.isfinite(lights.colors).all()
    assert torch.isfinite(lights.intensities).all()
    assert info["mode"] == "asset_scaled_world_space"


def test_asset_scaled_world_light_intensity_grows_with_scene_scale_squared() -> None:
    small, _ = make_asset_scaled_world_lights(make_means(scale=1.0), count=4, seed=5, device="cpu")
    large, _ = make_asset_scaled_world_lights(make_means(scale=2.0), count=4, seed=5, device="cpu")

    ratio = large.intensities[0] / small.intensities[0]
    assert torch.allclose(ratio, torch.tensor(4.0), atol=1e-6)


def test_asset_scaled_world_lights_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one Gaussian mean"):
        make_asset_scaled_world_lights(torch.empty((0, 3)), count=4, device="cpu")
    with pytest.raises(ValueError, match="positive light count"):
        make_asset_scaled_world_lights(make_means(), count=0, device="cpu")


def test_world_lights_to_camera_lights_identity_preserves_positions() -> None:
    world_lights = make_world_lights()
    camera = make_camera(torch.eye(4, dtype=torch.float32))

    lights = world_lights_to_camera_lights(world_lights, camera)

    assert torch.equal(lights.positions_cam, world_lights.positions_world)
    assert torch.equal(lights.colors, world_lights.colors)
    assert torch.equal(lights.intensities, world_lights.intensities)


def test_world_lights_to_camera_lights_matches_homogeneous_transform() -> None:
    world_lights = make_world_lights()
    viewmat = torch.tensor(
        [
            [0.0, -1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, -3.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    camera = make_camera(viewmat)

    lights = world_lights_to_camera_lights(world_lights, camera)
    positions_h = torch.cat((world_lights.positions_world, torch.ones((2, 1))), dim=-1)
    expected = torch.einsum("ij,nj->ni", viewmat, positions_h)[..., :3]

    assert torch.allclose(lights.positions_cam, expected)


def test_world_light_indices_are_stable_across_camera_transforms() -> None:
    world_lights = make_world_lights()
    camera_a = make_camera(torch.eye(4, dtype=torch.float32))
    camera_b = make_camera(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
    )

    lights_a = world_lights_to_camera_lights(world_lights, camera_a)
    lights_b = world_lights_to_camera_lights(world_lights, camera_b)

    assert torch.equal(lights_a.colors, lights_b.colors)
    assert torch.equal(lights_a.intensities, lights_b.intensities)
    assert torch.equal(lights_a.colors[1], world_lights.colors[1])


def make_means(scale: float = 1.0) -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.5],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.5],
        ],
        dtype=torch.float32,
    ) * float(scale)


def make_world_lights() -> WorldPointLights:
    return WorldPointLights(
        positions_world=torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], dtype=torch.float32),
        colors=torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=torch.float32),
        intensities=torch.tensor([0.25, 0.5], dtype=torch.float32),
    )


def make_camera(viewmat: torch.Tensor) -> PinholeCamera:
    return PinholeCamera(
        viewmats=viewmat[None],
        intrinsics=torch.eye(3, dtype=torch.float32)[None],
        width=1,
        height=1,
    )
