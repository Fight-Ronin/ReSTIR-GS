# Phase 38: Soft Visibility Shadow Filtering

Phase 38 refines the preferred visibility renderer target without changing the
RIS estimator, temporal reuse policy, proposal family, or dataset workflow.

## Motivation

The active renderer now uses:

```text
visible_contribution = lambertian_diffuse * shadow_visibility
```

Before this phase, `shadow_visibility` was a nearest-neighbor binary shadow-map
comparison. That was useful for wiring correctness, but it can produce hard
aliasing and self-shadow sensitivity because the shadow maps are rendered from
3DGS expected depth rather than exact mesh depth.

## Change

The shared visibility evaluator now supports percentage-closer filtering:

```text
pcf_radius = 0 -> legacy hard single-texel visibility
pcf_radius = 1 -> 3x3 averaged hard comparisons, visibility in [0, 1]
```

This is applied through the shared visibility path used by:

- all-lights visibility reference,
- selected-light visibility contribution,
- visibility-geometric proposal,
- visibility RIS target,
- interactive viewer visibility inspection.

The active Windows renderer runner defaults to:

```text
RESTIRGS_RESTIR_VISIBILITY_SHADOW_PCF_RADIUS=1
```

Set it to `0` to reproduce the hard shadow-map behavior:

```powershell
$env:RESTIRGS_RESTIR_VISIBILITY_SHADOW_PCF_RADIUS="0"
scripts\run_aligned_restir_renderer_windows.bat
```

## Scope

This phase does not add a new estimator, shadow algorithm, temporal policy,
visibility sweep, denoiser, PCSS, or physical light transport model. It only
softens the current expected-depth shadow-map proxy in the active target path.
