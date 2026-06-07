# Phase 34: Visibility-Aware Initial RIS Smoke

Phase 34 keeps the active aligned renderer unchanged and adds a small optional smoke for sampling a visibility-aware direct-light target.

The target is:

```text
visible_contribution(light) = lambertian_diffuse(light) * shadow_visibility(light)
```

The shadow visibility comes from the Phase 33 expected-depth shadow-map proxy. This is still an approximate reference, not physically exact visibility.

## What This Tests

The smoke answers one narrow question:

```text
Can uniform/geometric MC and initial RIS estimate the visibility-aware direct-light reference?
```

It does not change:

- the active aligned ReSTIR renderer,
- temporal reservoir reuse,
- the geometric proposal distribution,
- Blinn-Phong target handling,
- spatial reuse,
- visibility-aware temporal sampling.

## Estimators

For proposal MC, sampled contributions are evaluated as:

```text
estimate = mean(f_visible(light_i) / q(light_i))
```

For initial RIS, candidate weights are:

```text
target_i = luminance(f_visible(light_i))
w_i = target_i / q(light_i)
W = sum(w_i) / (K * target_selected)
estimate = f_visible(selected) * W
```

When all visibility values are one, the estimator reduces to the existing diffuse MC/RIS behavior.

## Run

```powershell
scripts\run_aligned_visibility_ris_smoke_windows.bat
```

Useful overrides:

```powershell
$env:RESTIRGS_VISIBILITY_RIS_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VISIBILITY_RIS_EXTRA_ARGS="--candidate-count 8 --num-lights 16"
scripts\run_aligned_visibility_ris_smoke_windows.bat
```

## Outputs

The script writes under `outputs/aligned_visibility_ris/`:

```text
visibility_ris_rows.csv
visibility_ris_summary.json
visibility_ris_unshadowed_reference.png
visibility_ris_shadowed_reference.png
visibility_ris_uniform_mc.png
visibility_ris_geometric_mc.png
visibility_ris_uniform_ris.png
visibility_ris_geometric_ris.png
visibility_ris_geometric_ris_abs_error.png
visibility_ris_contact.png
```

Rows compare `uniform/geometric` proposals and `mc/ris` estimators against the shadowed all-lights reference for:

```text
visible_contribution_rgb
visible_composite_rgb
```

## Interpretation

Passing this phase means the visibility-aware target path is wired correctly and produces finite metrics. It does not require RIS to beat MC.

If this smoke is stable, the next decision is whether to expose visibility as an optional target mode in a controlled renderer path. That should happen only after the smoke remains robust across more than one aligned asset.
