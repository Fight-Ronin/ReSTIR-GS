from __future__ import annotations

import math

import torch

from restir_gs.lighting.deferred import (
    PointLights,
    evaluate_selected_light_blinn_phong,
    make_deterministic_point_lights,
    shade_deferred_blinn_phong,
    shade_deferred_lambertian,
)
from restir_gs.render.gbuffer import GBuffer


def make_gbuffer(
    rgb: torch.Tensor,
    position: torch.Tensor,
    normal: torch.Tensor,
    valid_mask: torch.Tensor,
    normal_mask: torch.Tensor,
) -> GBuffer:
    height, width, _ = rgb.shape
    return GBuffer(
        rgb=rgb,
        depth=torch.ones((height, width), dtype=rgb.dtype),
        alpha=torch.ones((height, width), dtype=rgb.dtype),
        position_cam=position,
        normal_cam=normal,
        valid_mask=valid_mask,
        normal_mask=normal_mask,
    )


def test_single_light_lambertian_matches_closed_form() -> None:
    rgb = torch.tensor([[[0.5, 0.25, 1.0]]], dtype=torch.float32)
    position = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        colors=torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float32),
        intensities=torch.tensor([4.0], dtype=torch.float32),
    )

    buffers = shade_deferred_lambertian(gbuffer, lights, ambient=0.2, distance_epsilon=0.0)

    expected_irradiance = torch.tensor([[[8.0, 4.0, 2.0]]], dtype=torch.float32)
    expected_shade = 0.2 + expected_irradiance / math.pi
    assert torch.allclose(buffers.irradiance_rgb, expected_irradiance)
    assert torch.allclose(buffers.diffuse_rgb, rgb * expected_irradiance / math.pi)
    assert torch.allclose(buffers.specular_rgb, torch.zeros_like(rgb))
    assert torch.allclose(buffers.shade_rgb, expected_shade)
    assert torch.allclose(buffers.composite_rgb, rgb * expected_shade)


def test_single_light_blinn_phong_matches_closed_form() -> None:
    rgb = torch.tensor([[[0.5, 0.25, 1.0]]], dtype=torch.float32)
    position = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        colors=torch.tensor([[2.0, 1.0, 0.5]], dtype=torch.float32),
        intensities=torch.tensor([4.0], dtype=torch.float32),
    )

    buffers = shade_deferred_blinn_phong(
        gbuffer,
        lights,
        ambient=0.2,
        specular_strength=0.25,
        shininess=8.0,
        distance_epsilon=0.0,
    )

    expected_irradiance = torch.tensor([[[8.0, 4.0, 2.0]]], dtype=torch.float32)
    expected_diffuse = rgb * expected_irradiance / math.pi
    expected_specular = torch.tensor([[[2.0, 1.0, 0.5]]], dtype=torch.float32)
    assert torch.allclose(buffers.irradiance_rgb, expected_irradiance, atol=1e-6)
    assert torch.allclose(buffers.diffuse_rgb, expected_diffuse, atol=1e-6)
    assert torch.allclose(buffers.specular_rgb, expected_specular, atol=1e-6)
    assert torch.allclose(buffers.composite_rgb, rgb * 0.2 + expected_diffuse + expected_specular, atol=1e-6)


def test_selected_light_blinn_phong_returns_per_candidate_terms() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )

    diffuse, specular = evaluate_selected_light_blinn_phong(
        gbuffer,
        lights,
        torch.tensor([[[0, 1]]], dtype=torch.long),
        specular_strength=0.5,
        shininess=4.0,
        distance_epsilon=0.0,
    )

    assert diffuse.shape == (1, 1, 2, 3)
    assert specular.shape == (1, 1, 2, 3)
    assert torch.all(diffuse >= 0.0)
    assert torch.all(specular >= 0.0)


def test_two_sided_lighting_is_invariant_to_normal_sign() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )
    normal_pos = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    normal_neg = torch.tensor([[[0.0, 0.0, -1.0]]], dtype=torch.float32)

    lit_pos = shade_deferred_lambertian(
        make_gbuffer(rgb, position, normal_pos, valid, valid),
        lights,
        ambient=0.0,
        distance_epsilon=0.0,
        two_sided=True,
    )
    lit_neg = shade_deferred_lambertian(
        make_gbuffer(rgb, position, normal_neg, valid, valid),
        lights,
        ambient=0.0,
        distance_epsilon=0.0,
        two_sided=True,
    )

    assert torch.allclose(lit_pos.irradiance_rgb, lit_neg.irradiance_rgb)
    assert torch.allclose(lit_pos.composite_rgb, lit_neg.composite_rgb)


def test_distance_epsilon_does_not_shrink_light_direction() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )

    buffers = shade_deferred_lambertian(
        make_gbuffer(rgb, position, normal, valid, valid),
        lights,
        ambient=0.0,
        distance_epsilon=3.0,
    )

    expected = torch.full((1, 1, 3), 1.0 / 7.0, dtype=torch.float32)
    assert torch.allclose(buffers.irradiance_rgb, expected)


def test_negative_distance_epsilon_fails_loudly() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.zeros((1, 1, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )

    try:
        shade_deferred_lambertian(make_gbuffer(rgb, position, normal, valid, valid), lights, distance_epsilon=-1e-4)
    except ValueError as exc:
        assert "distance_epsilon" in str(exc)
    else:
        raise AssertionError("Expected negative distance_epsilon to fail.")


def test_invalid_pixels_have_zero_lighting_and_keep_original_rgb() -> None:
    rgb = torch.tensor([[[1.0, 0.5, 0.25], [0.2, 0.4, 0.6]]], dtype=torch.float32)
    position = torch.zeros((1, 2, 3), dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid_mask = torch.tensor([[True, False]])
    normal_mask = torch.tensor([[True, True]])
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )

    buffers = shade_deferred_lambertian(
        make_gbuffer(rgb, position, normal, valid_mask, normal_mask),
        lights,
        ambient=0.2,
        distance_epsilon=0.0,
    )

    assert torch.allclose(buffers.irradiance_rgb[0, 1], torch.zeros(3))
    assert torch.allclose(buffers.diffuse_rgb[0, 1], torch.zeros(3))
    assert torch.allclose(buffers.shade_rgb[0, 1], torch.zeros(3))
    assert torch.allclose(buffers.composite_rgb[0, 1], rgb[0, 1])

    blinn = shade_deferred_blinn_phong(
        make_gbuffer(rgb, position, normal, valid_mask, normal_mask),
        lights,
        ambient=0.2,
        specular_strength=0.25,
        distance_epsilon=0.0,
    )
    assert torch.allclose(blinn.diffuse_rgb[0, 1], torch.zeros(3))
    assert torch.allclose(blinn.specular_rgb[0, 1], torch.zeros(3))
    assert torch.allclose(blinn.composite_rgb[0, 1], rgb[0, 1])


def test_blinn_phong_zero_specular_matches_lambertian() -> None:
    rgb = torch.full((1, 1, 3), 0.5, dtype=torch.float32)
    position = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    normal = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    gbuffer = make_gbuffer(rgb, position, normal, valid, valid)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )

    lambertian = shade_deferred_lambertian(gbuffer, lights, ambient=0.2, distance_epsilon=0.0)
    blinn = shade_deferred_blinn_phong(gbuffer, lights, ambient=0.2, specular_strength=0.0, distance_epsilon=0.0)

    assert torch.allclose(blinn.specular_rgb, torch.zeros_like(blinn.specular_rgb))
    assert torch.allclose(blinn.composite_rgb, lambertian.composite_rgb, atol=1e-6)


def test_blinn_phong_two_sided_lighting_is_invariant_to_normal_sign() -> None:
    rgb = torch.ones((1, 1, 3), dtype=torch.float32)
    position = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)
    lights = PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        colors=torch.ones((1, 3), dtype=torch.float32),
        intensities=torch.ones((1,), dtype=torch.float32),
    )
    normal_pos = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32)
    normal_neg = torch.tensor([[[0.0, 0.0, -1.0]]], dtype=torch.float32)

    lit_pos = shade_deferred_blinn_phong(
        make_gbuffer(rgb, position, normal_pos, valid, valid),
        lights,
        ambient=0.0,
        specular_strength=0.25,
        distance_epsilon=0.0,
        two_sided=True,
    )
    lit_neg = shade_deferred_blinn_phong(
        make_gbuffer(rgb, position, normal_neg, valid, valid),
        lights,
        ambient=0.0,
        specular_strength=0.25,
        distance_epsilon=0.0,
        two_sided=True,
    )

    assert torch.allclose(lit_pos.diffuse_rgb, lit_neg.diffuse_rgb)
    assert torch.allclose(lit_pos.specular_rgb, lit_neg.specular_rgb)
    assert torch.allclose(lit_pos.composite_rgb, lit_neg.composite_rgb)


def test_deterministic_point_lights_repeat_for_same_seed() -> None:
    lights_a = make_deterministic_point_lights(count=8, seed=2027, device="cpu")
    lights_b = make_deterministic_point_lights(count=8, seed=2027, device="cpu")

    assert torch.allclose(lights_a.positions_cam, lights_b.positions_cam)
    assert torch.allclose(lights_a.colors, lights_b.colors)
    assert torch.allclose(lights_a.intensities, lights_b.intensities)
    assert bool((lights_a.positions_cam[:, :2] >= -1.2).all())
    assert bool((lights_a.positions_cam[:, :2] <= 1.2).all())
    assert bool((lights_a.positions_cam[:, 2] >= 0.8).all())
    assert bool((lights_a.positions_cam[:, 2] <= 3.8).all())
    assert bool((lights_a.colors >= 0.4).all())
    assert bool((lights_a.colors <= 1.0).all())
    assert torch.allclose(lights_a.intensities, torch.full((8,), 3.0 / 8.0))
