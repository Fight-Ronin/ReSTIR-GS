from __future__ import annotations

import pytest
import torch

from restir_gs.lighting.asset_lights import make_asset_scaled_point_lights
from restir_gs.render.gbuffer import GBuffer


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
