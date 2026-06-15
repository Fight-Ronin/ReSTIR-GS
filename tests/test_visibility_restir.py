from __future__ import annotations

import pytest
import torch

import restir_gs.restir.proposal as proposal_module
from restir_gs.lighting.deferred import PointLights
from restir_gs.lighting.visibility import (
    ShadowMapBundle,
    evaluate_selected_light_visible_diffuse_cached,
    make_shadow_visibility_cache,
    shade_deferred_lambertian_visible,
    shade_deferred_lambertian_visible_cached,
)
from restir_gs.render.gbuffer import GBuffer
from restir_gs.render.synthetic_scene import PinholeCamera
from restir_gs.restir.initial import estimate_proposal_lighting
from restir_gs.restir.proposal import (
    CandidateSamples,
    compute_geometric_proposal_distribution,
    compute_visibility_geometric_proposal_distribution,
    compute_visibility_geometric_proposal_distribution_cached,
)
from restir_gs.restir.visibility import (
    estimate_visibility_proposal_lighting,
    estimate_visibility_proposal_lighting_cached,
    estimate_visibility_ris_initial_lighting,
    estimate_visibility_ris_initial_lighting_cached,
)


def test_all_visible_proposal_matches_existing_diffuse_estimator() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    samples = CandidateSamples(
        light_indices=torch.tensor([[[0, 1]]], dtype=torch.long),
        proposal_probs=torch.full((1, 1, 2), 0.5, dtype=torch.float32),
    )
    shadow = make_shadow_bundle(depths=[10.0, 10.0], alphas=[0.0, 0.0], depth_bias=0.0)

    visible = estimate_visibility_proposal_lighting(gbuffer, camera, lights, shadow, samples, ambient=0.2)
    unshadowed = estimate_proposal_lighting(gbuffer, lights, samples, target_mode="diffuse", ambient=0.2)

    assert torch.allclose(visible.contribution_rgb, unshadowed.contribution_rgb)
    assert torch.allclose(visible.composite_rgb, unshadowed.composite_rgb)


def test_blocked_visibility_ris_zero_target_returns_ambient_composite() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 2.0), rgb=(0.8, 0.6, 0.4))
    camera = make_identity_camera()
    lights = make_two_lights()
    candidates = torch.tensor([[[0, 1]]], dtype=torch.long)
    shadow = make_shadow_bundle(depths=[0.5, 0.5], alphas=[1.0, 1.0], depth_bias=0.0)

    buffers, reservoir = estimate_visibility_ris_initial_lighting(
        gbuffer,
        camera,
        lights,
        shadow,
        candidates,
        selection_seed=11,
        ambient=0.2,
    )

    assert torch.allclose(buffers.contribution_rgb, torch.zeros((1, 1, 3)))
    assert torch.allclose(buffers.composite_rgb, gbuffer.rgb * 0.2)
    assert bool(buffers.valid_mask.all())
    assert not bool(reservoir.valid_mask.any())
    assert torch.allclose(reservoir.W, torch.zeros((1, 1)))


def test_visibility_ris_k1_matches_visibility_proposal_mc_for_positive_target() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    candidates = torch.tensor([[[1]]], dtype=torch.long)
    samples = CandidateSamples(light_indices=candidates, proposal_probs=torch.ones((1, 1, 1), dtype=torch.float32))
    shadow = make_shadow_bundle(depths=[10.0, 10.0], alphas=[0.0, 0.0], depth_bias=0.0)

    mc = estimate_visibility_proposal_lighting(gbuffer, camera, lights, shadow, samples, ambient=0.2)
    ris, reservoir = estimate_visibility_ris_initial_lighting(
        gbuffer,
        camera,
        lights,
        shadow,
        candidates,
        proposal_probs=samples.proposal_probs,
        selection_seed=17,
        ambient=0.2,
    )

    assert torch.allclose(ris.contribution_rgb, mc.contribution_rgb)
    assert torch.allclose(ris.composite_rgb, mc.composite_rgb)
    assert torch.equal(reservoir.light_indices, torch.tensor([[1]]))
    assert torch.equal(reservoir.M, torch.tensor([[1]]))


def test_cached_visibility_paths_match_uncached_paths() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_shadow_bundle(depths=[10.0, 1.0], alphas=[0.0, 1.0], depth_bias=0.0)
    shadow.depth_maps[1, 0, 0] = 3.0
    shadow.depth_maps[1, 0, 1] = 3.0
    shadow.depth_maps[1, 1, 0] = 3.0
    cache = make_shadow_visibility_cache(gbuffer, camera, shadow, pcf_radius=1)
    samples = CandidateSamples(
        light_indices=torch.tensor([[[0, 1]]], dtype=torch.long),
        proposal_probs=torch.full((1, 1, 2), 0.5, dtype=torch.float32),
    )

    reference = shade_deferred_lambertian_visible(gbuffer, camera, lights, shadow, pcf_radius=1)
    cached_reference = shade_deferred_lambertian_visible_cached(gbuffer, lights, cache)
    proposal = compute_visibility_geometric_proposal_distribution(gbuffer, camera, lights, shadow, pcf_radius=1)
    cached_proposal = compute_visibility_geometric_proposal_distribution_cached(gbuffer, lights, cache)
    mc = estimate_visibility_proposal_lighting(gbuffer, camera, lights, shadow, samples, pcf_radius=1)
    cached_mc = estimate_visibility_proposal_lighting_cached(gbuffer, lights, cache, samples)
    ris, reservoir = estimate_visibility_ris_initial_lighting(
        gbuffer,
        camera,
        lights,
        shadow,
        samples.light_indices,
        proposal_probs=samples.proposal_probs,
        pcf_radius=1,
        selection_seed=23,
    )
    cached_ris, cached_reservoir = estimate_visibility_ris_initial_lighting_cached(
        gbuffer,
        lights,
        cache,
        samples.light_indices,
        proposal_probs=samples.proposal_probs,
        selection_seed=23,
    )
    contribution_candidates = evaluate_selected_light_visible_diffuse_cached(gbuffer, lights, cache, samples.light_indices)
    reused_mc = estimate_visibility_proposal_lighting_cached(
        gbuffer,
        lights,
        cache,
        samples,
        contribution_candidates=contribution_candidates,
    )
    reused_ris, reused_reservoir = estimate_visibility_ris_initial_lighting_cached(
        gbuffer,
        lights,
        cache,
        samples.light_indices,
        proposal_probs=samples.proposal_probs,
        selection_seed=23,
        contribution_candidates=contribution_candidates,
    )
    direct_candidate_ris, direct_candidate_reservoir = estimate_visibility_ris_initial_lighting_cached(
        gbuffer,
        lights,
        None,
        samples.light_indices,
        proposal_probs=samples.proposal_probs,
        selection_seed=23,
        contribution_candidates=contribution_candidates,
    )

    assert torch.allclose(cached_reference.diffuse_rgb, reference.diffuse_rgb)
    assert torch.allclose(cached_reference.composite_rgb, reference.composite_rgb)
    assert torch.allclose(cached_proposal, proposal)
    assert torch.allclose(cached_mc.contribution_rgb, mc.contribution_rgb)
    assert torch.allclose(cached_mc.composite_rgb, mc.composite_rgb)
    assert torch.allclose(reused_mc.contribution_rgb, cached_mc.contribution_rgb)
    assert torch.allclose(reused_mc.composite_rgb, cached_mc.composite_rgb)
    assert torch.allclose(cached_ris.contribution_rgb, ris.contribution_rgb)
    assert torch.allclose(cached_ris.composite_rgb, ris.composite_rgb)
    assert torch.allclose(reused_ris.contribution_rgb, cached_ris.contribution_rgb)
    assert torch.allclose(reused_ris.composite_rgb, cached_ris.composite_rgb)
    assert torch.allclose(direct_candidate_ris.contribution_rgb, cached_ris.contribution_rgb)
    assert torch.allclose(direct_candidate_ris.composite_rgb, cached_ris.composite_rgb)
    assert torch.equal(cached_reservoir.light_indices, reservoir.light_indices)
    assert torch.allclose(cached_reservoir.W, reservoir.W)
    assert torch.equal(reused_reservoir.light_indices, cached_reservoir.light_indices)
    assert torch.allclose(reused_reservoir.W, cached_reservoir.W)
    assert torch.equal(direct_candidate_reservoir.light_indices, cached_reservoir.light_indices)
    assert torch.allclose(direct_candidate_reservoir.W, cached_reservoir.W)


def test_cached_visibility_ris_requires_cache_without_contribution_candidates() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 2.0))
    lights = make_two_lights()
    candidates = torch.tensor([[[0, 1]]], dtype=torch.long)

    with pytest.raises(ValueError, match="visibility cache"):
        estimate_visibility_ris_initial_lighting_cached(gbuffer, lights, None, candidates)


def test_cached_visibility_geometric_proposal_uses_dense_cache_without_gather(monkeypatch) -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_shadow_bundle(depths=[10.0, 1.0], alphas=[0.0, 1.0], depth_bias=0.0)
    cache = make_shadow_visibility_cache(gbuffer, camera, shadow, pcf_radius=1)

    expected = compute_visibility_geometric_proposal_distribution(gbuffer, camera, lights, shadow, pcf_radius=1)

    def fail_gather(*_args, **_kwargs):
        raise AssertionError("dense visibility cache should not regather all light indices")

    monkeypatch.setattr(proposal_module, "gather_shadow_visibility", fail_gather)
    actual = proposal_module.compute_visibility_geometric_proposal_distribution_cached(gbuffer, lights, cache)

    assert torch.allclose(actual, expected)


def test_invalid_pixels_produce_zero_visible_proposal_contribution() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 1.0), valid=False)
    camera = make_identity_camera()
    lights = make_two_lights()
    samples = CandidateSamples(
        light_indices=torch.tensor([[[0, 1]]], dtype=torch.long),
        proposal_probs=torch.full((1, 1, 2), 0.5, dtype=torch.float32),
    )
    shadow = make_shadow_bundle(depths=[10.0, 10.0], alphas=[0.0, 0.0], depth_bias=0.0)

    visible = estimate_visibility_proposal_lighting(gbuffer, camera, lights, shadow, samples, ambient=0.2)

    assert torch.allclose(visible.contribution_rgb, torch.zeros((1, 1, 3)))
    assert torch.allclose(visible.composite_rgb, gbuffer.rgb)
    assert not bool(visible.valid_mask.any())


def test_visibility_geometric_proposal_matches_geometric_when_all_lights_visible() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_shadow_bundle(depths=[10.0, 10.0], alphas=[0.0, 0.0], depth_bias=0.0)

    geometric = compute_geometric_proposal_distribution(gbuffer, lights)
    visible = compute_visibility_geometric_proposal_distribution(gbuffer, camera, lights, shadow)

    assert torch.allclose(visible, geometric)


def test_visibility_geometric_proposal_removes_blocked_light_mass() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_shadow_bundle(depths=[10.0, 0.5], alphas=[0.0, 1.0], depth_bias=0.0)

    visible = compute_visibility_geometric_proposal_distribution(gbuffer, camera, lights, shadow)

    assert torch.allclose(visible, torch.tensor([[[1.0, 0.0]]], dtype=torch.float32))


def test_visibility_geometric_proposal_uses_soft_pcf_visibility() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 2.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_shadow_bundle(depths=[10.0, 1.0], alphas=[0.0, 1.0], depth_bias=0.0)
    shadow.depth_maps[1, 0, 0] = 3.0
    shadow.depth_maps[1, 0, 1] = 3.0
    shadow.depth_maps[1, 1, 0] = 3.0

    geometric = compute_geometric_proposal_distribution(gbuffer, lights)
    visible = compute_visibility_geometric_proposal_distribution(gbuffer, camera, lights, shadow, pcf_radius=1)
    expected_weights = geometric * torch.tensor([[[1.0, 3.0 / 9.0]]], dtype=torch.float32)
    expected = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    assert torch.allclose(visible, expected)


def test_visibility_geometric_proposal_falls_back_when_all_lights_blocked() -> None:
    gbuffer = make_gbuffer(position=(0.0, 0.0, 1.0))
    camera = make_identity_camera()
    lights = make_two_lights()
    shadow = make_shadow_bundle(depths=[0.5, 0.5], alphas=[1.0, 1.0], depth_bias=0.0)

    geometric = compute_geometric_proposal_distribution(gbuffer, lights)
    visible = compute_visibility_geometric_proposal_distribution(gbuffer, camera, lights, shadow)

    assert torch.allclose(visible, geometric)


def make_gbuffer(
    position: tuple[float, float, float],
    rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    valid: bool = True,
) -> GBuffer:
    valid_mask = torch.tensor([[valid]], dtype=torch.bool)
    return GBuffer(
        rgb=torch.tensor([[rgb]], dtype=torch.float32),
        depth=torch.ones((1, 1), dtype=torch.float32),
        alpha=torch.ones((1, 1), dtype=torch.float32),
        position_cam=torch.tensor([[position]], dtype=torch.float32),
        normal_cam=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32),
        valid_mask=valid_mask,
        normal_mask=valid_mask.clone(),
    )


def make_identity_camera() -> PinholeCamera:
    return PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        width=3,
        height=3,
    )


def make_shadow_bundle(depths: list[float], alphas: list[float], depth_bias: float) -> ShadowMapBundle:
    camera = make_identity_camera()
    return ShadowMapBundle(
        light_indices=torch.arange(len(depths), dtype=torch.long),
        light_cameras=[camera for _ in depths],
        depth_maps=torch.stack([torch.full((3, 3), depth, dtype=torch.float32) for depth in depths], dim=0),
        alpha_maps=torch.stack([torch.full((3, 3), alpha, dtype=torch.float32) for alpha in alphas], dim=0),
        scene_radius=1.0,
        depth_bias=depth_bias,
    )


def make_two_lights() -> PointLights:
    return PointLights(
        positions_cam=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
        intensities=torch.ones((2,), dtype=torch.float32),
    )
