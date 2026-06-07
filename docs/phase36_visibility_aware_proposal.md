# Phase 36: Visibility-Aware Proposal Refinement

Phase 36 is a narrow algorithm refinement for the optional visibility target path. The default diffuse aligned renderer is unchanged.

## Motivation

The visibility target estimates:

```text
visible_contribution(light) = lambertian_diffuse(light) * shadow_visibility(light)
```

Before this phase, the visibility renderer still sampled lights from the old geometric proposal:

```text
q_geo(l | x) proportional to intensity_l * luminance(color_l) * abs(dot(N, wi)) / dist2
```

That proposal does not know whether a light is shadowed. Phase 36 makes the visibility target use a visibility-aware proposal:

```text
q_vis(l | x) proportional to q_geo_unnormalized(l | x) * shadow_visibility(x, l)
```

In implementation, this is computed by multiplying the normalized geometric distribution by binary shadow visibility and renormalizing. If all visible mass is zero for a pixel, the proposal falls back to the base geometric distribution so sampling remains well-defined.

## Interface

The renderer chooses the proposal from the target mode:

```text
target_mode=diffuse    -> proposal=geometric
target_mode=visibility -> proposal=visibility_geometric
```

There is no public proposal switch in the active renderer. This is intentional: the visibility-aware proposal is an algorithm refinement of the visibility target path, not a new ablation knob.

The default active renderer remains:

```text
target_mode=diffuse
proposal=geometric
```

## Run

Run a short visibility renderer pass with the refined proposal:

```powershell
scripts\run_visibility_validation_windows.bat
```

Or call the renderer directly:

```powershell
python scripts/demo_26_aligned_restir_renderer.py --asset-set testing --target-mode visibility --num-lights 16 --frame-indices 45,46,47 --width 128 --height 128 --device cuda --output-dir outputs/aligned_restir_visibility
```

## Correctness Anchors

- If every light is visible, `visibility_geometric` matches `geometric`.
- If one light is shadowed, its proposal mass becomes zero after renormalization.
- If every light has zero visible mass, the proposal falls back to geometric.
- Diffuse target mode keeps the geometric proposal.

## Boundary

This phase changes proposal quality for the optional visibility target only. It does not change temporal reuse, spatial reuse, shadow-map quality, visibility bias, RIS target math, Blinn-Phong, or the default active validation path.
