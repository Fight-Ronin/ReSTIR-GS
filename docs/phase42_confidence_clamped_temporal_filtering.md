# Phase 42: Confidence-Clamped Temporal Filtering

Phase 42 adds a cheap temporal filtering layer on top of the active visibility RIS renderer. It does not change the visibility target, proposal distribution, RIS weights, world-space lights, shadow maps, G-buffer, or temporal reprojection gates.

Phase 43 tested a more aggressive soft pre-gate filter, but active validation did not meet acceptance criteria, so this conservative Phase 42 behavior remains the active temporal filter.

The renderer now reports three estimators:

```text
initial_ris
temporal_ris
temporal_filtered_ris
```

`temporal_ris` is retained as the reservoir-combine debug path. The preferred active temporal output is `temporal_filtered_ris`.

## Filter Equation

For each current pixel:

```text
current = current-frame initial RIS contribution
history = reprojected previous filtered contribution
alpha   = temporal_filter_blend_max * compatibility_confidence

filtered = lerp(current, clamped(history around current), alpha)
```

Missing or rejected history gives `alpha=0`, so the filtered output exactly equals current-frame initial RIS.

## Compatibility Confidence

The filter reuses the existing temporal lookup diagnostics:

```text
depth_conf  = clamp(1 - relative_depth_error / depth_tolerance, 0, 1)
normal_conf = normal_abs_dot
rgb_conf    = clamp(1 - rgb_distance / temporal_rgb_threshold, 0, 1)
motion_conf = clamp(1 - motion_pixels_norm / temporal_max_motion_pixels, 0, 1)
confidence  = min(depth_conf, normal_conf, rgb_conf, motion_conf)
alpha       = lookup.valid_mask * temporal_filter_blend_max * confidence
```

If an optional threshold is disabled with `none`, its confidence term becomes `1`.

## History Clamp

History is clamped around the current estimate before blending:

```text
clamp_radius = temporal_filter_clamp_scale * mean(abs(current)) + temporal_filter_clamp_min
history_clamped = clamp(history, current - clamp_radius, current + clamp_radius)
```

The composite is recomputed from the current-frame pseudo albedo:

```text
filtered_composite = current_gbuffer.rgb * ambient + filtered_contribution
```

Invalid current pixels preserve the original initial composite and receive zero filtered contribution.

## Active Defaults

The active Windows renderer runner uses:

```text
target_mode = visibility
proposal = visibility_geometric
num_lights = 16
resolution = 128x128
frames = 45,46,47
temporal_history_m_cap = 1
temporal_filter_blend_max = 0.15
temporal_filter_clamp_scale = 0.50
temporal_filter_clamp_min = 1e-5
```

The active renderer CSV should now contain `72` rows for the four-asset testing set:

```text
4 assets * 3 frames * 3 estimators * 2 reference quantities = 72 rows
```

## Outputs

The renderer writes the existing reference, initial, temporal reservoir, gate, reuse, and error previews plus:

```text
final_temporal_filtered_ris.png
final_temporal_filtered_abs_error.png
final_temporal_filter_alpha.png
```

Rows include:

```text
temporal_filter_confidence_mean
temporal_filter_alpha_mean
temporal_filter_alpha_max
temporal_filter_history_delta_mean
temporal_filter_clamp_delta_mean
```

## Interpretation

This is a real-time variance-reduction layer, not a heavier ReSTIR sampling change. The current-frame RIS estimate remains the fresh signal. The previous filtered contribution is a bounded history signal. The reservoir-combine temporal output remains available for debugging, but it is no longer the main active temporal image.
