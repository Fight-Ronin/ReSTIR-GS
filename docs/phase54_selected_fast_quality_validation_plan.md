# Phase 54: Selected-Fast Quality Validation Plan

## Final Status

This quality harness remains the main validation artifact for selected-fast.
Follow-up proposal experiments were measured and then removed from the codepath
because they did not produce a stable cross-asset policy. Final project scope
and cleanup decisions are summarized in `docs/phase55_project_wrap_up.md`.

## Goal

Phase 53 made the selected-fast visibility path fast enough to be a serious
real-time candidate:

```text
dxgl_apple, 256x256, K=8, medians:

mode            lights  fps    frame_ms  selected_vis_ms
cache           32      25.06   39.91      0.00
selected_fast   32      28.44   35.17      7.05

cache          128       7.68  130.23      0.00
selected_fast  128      21.61   46.27      8.63
```

The next question is not kernel speed. The next question is whether the
selected-fast display path has acceptable image quality and temporal stability.

```text
dense-cache default: visibility-geometric proposal + dense all-light visibility
selected-fast:       geometric proposal + selected candidate visibility
```

Those are different proposal semantics, so matching shadow visibility math is
not enough. Phase 54 should validate quality before any renderer policy change.

## Non-Goals

- Do not promote selected-fast to the default renderer.
- Do not optimize another kernel.
- Do not tune temporal thresholds or filter alpha.
- Do not change proposal semantics while measuring selected-fast.
- Do not use ray tracing.

## Validation Questions

1. Does selected-fast produce comparable error against an all-light visible
   diffuse reference?
2. Does selected-fast introduce obvious occlusion noise, missing shadows, or
   high-power-light flicker?
3. Does temporal filtering stabilize selected-fast rather than hiding large
   frame-to-frame proposal noise?
4. Does selected-fast keep its measured frame-time advantage while writing the
   same quality outputs?

## Experiment Design

Use one script-level A/B harness, kept separate from the default renderer.

Suggested script:

```text
scripts/eval_selected_fast_quality.py
```

The script should run both paths on the same asset, frame window, light seed,
shadow bundle, and camera sequence:

```text
asset_id = dxgl_apple
frames = 45..53
resolution = 256x256
num_lights = 32 and 128
candidate_count = 8
visibility_shadow_resolution = 128
visibility_shadow_pcf_radius = 1
```

## Reference Definition

The reference for quality metrics should be all-light visible diffuse computed
from the dense visibility cache:

```text
reference.diffuse_rgb
reference.composite_rgb
reference.valid_mask
```

This is the same reference quantity used by the existing aligned ReSTIR
evaluation path. It keeps the comparison about estimator/proposal quality, not
about a new lighting model.

## Compared Outputs

For each frame, record metrics for both paths:

```text
dense_cache.initial.contribution_rgb
dense_cache.temporal.contribution_rgb
dense_cache.temporal_filtered.contribution_rgb

selected_fast.initial.contribution_rgb
selected_fast.temporal.contribution_rgb
selected_fast.temporal_filtered.contribution_rgb
```

Also save visual previews:

```text
reference composite
dense-cache temporal filtered composite
selected-fast temporal filtered composite
dense-cache abs error
selected-fast abs error
selected-fast minus dense-cache abs difference
reuse mask
temporal filter alpha
```

## Metrics

Reuse `restir_gs.metrics.compute_rgb_error_metrics` for contribution-space
metrics against `reference.diffuse_rgb` and valid pixels:

```text
mae
rmse
bias_r
bias_g
bias_b
mean_abs_bias
```

Add frame-to-frame stability metrics for temporal filtered composite output:

```text
mean_abs_frame_delta
valid_pixel_frame_delta
```

These are not perfect perceptual metrics, but they are enough for the next
decision gate.

## Output Files

For each run:

```text
outputs/selected_fast_quality/<asset_id>/
  selected_fast_quality_rows.csv
  selected_fast_quality_summary.json
  contact.png
  frame_<index>_reference.png
  frame_<index>_dense_temporal_filtered.png
  frame_<index>_selected_fast_temporal_filtered.png
  frame_<index>_dense_abs_error.png
  frame_<index>_selected_fast_abs_error.png
  frame_<index>_selected_vs_dense_abs_diff.png
```

The contact sheet should be the main human-inspection artifact.

## Current Harness Status

The first A/B harness is implemented:

```text
scripts/eval_selected_fast_quality.py
```

Tiny smoke run:

```text
asset_id = dxgl_apple
frames = 49,50
resolution = 64x64
num_lights = 4
candidate_count = 2
visibility_shadow_resolution = 32
```

Outputs:

```text
outputs/selected_fast_quality_smoke/selected_fast_quality_rows.csv
outputs/selected_fast_quality_smoke/selected_fast_quality_summary.json
outputs/selected_fast_quality_smoke/dxgl_apple/4l_k2/contact.png
```

The smoke verifies that the harness writes finite metrics and preview images.
It is intentionally too small to make a quality decision.

## Initial 256x256 Matrix Result

Formal first-pass matrix:

```text
asset_id = dxgl_apple
frames = 45..53
resolution = 256x256
candidate_count = 8
visibility_shadow_resolution = 128
visibility_shadow_pcf_radius = 1
```

Temporal-filtered contribution metrics, averaged over 9 frames:

```text
lights  path           contribution_mae  contribution_rmse  frame_gpu_ms
32      dense_cache    0.001641          0.003824           127.15
32      selected_fast  0.012421          0.020384            73.67

128     dense_cache    0.003513          0.005839           189.86
128     selected_fast  0.014004          0.021593            61.42
```

Per-frame selected-fast MAE is consistently higher than dense-cache MAE:

```text
32 lights:  roughly 6.5x to 13.9x dense-cache MAE
128 lights: roughly 3.4x to 4.5x dense-cache MAE
```

Frame-to-frame temporal-filtered composite deltas are only slightly higher for
selected-fast:

```text
32 lights:  dense 0.0386, selected-fast 0.0423
128 lights: dense 0.0391, selected-fast 0.0432
```

Interpretation:

- Selected-fast keeps a clear frame-time advantage in the quality harness.
- Main views look visually close at thumbnail scale.
- Error and selected-vs-dense difference maps show broad surface noise, not only
  localized shadow-edge error.
- The likely next bottleneck is proposal quality/variance, not selected
  visibility kernel speed.

This result is not strong enough to promote selected-fast as the default
renderer policy. It is strong enough to continue with proposal-quality
experiments.

## Implementation Plan

1. Add the A/B script.
   - Reuse asset loading, light creation, and shadow bundle setup from
     `scripts/bench_realtime_display_fps.py`.
   - Reuse image conversion/contact-sheet patterns from
     `scripts/demo_26_aligned_restir_renderer.py`.
   - Reuse `compute_rgb_error_metrics` for CSV/JSON metrics.

2. Keep both paths explicit.
   - Dense-cache path uses the current visibility display/evaluation behavior.
   - Selected-fast path calls the experimental selected visibility renderer with
     `selected_visibility_impl="fast"`.
   - Do not hide the mode choice behind global renderer defaults.

3. Validate finite outputs.
   - Check all output tensors used for metrics are finite.
   - Check CSV rows are non-empty.
   - Check saved images exist.

4. Run the small matrix.
   - `dxgl_apple`, 32 lights.
   - `dxgl_apple`, 128 lights.

5. Decide whether to broaden.
   - If results are clean, run the same script over the active aligned asset
     subset.
   - If results show visible noise, keep selected-fast experimental and inspect
     proposal semantics before optimizing further.

## Success Criteria

Selected-fast can move toward renderer-policy discussion only if:

- Full tests still pass.
- The A/B script writes finite CSV/JSON and preview images.
- Selected-fast temporal filtered MAE/RMSE is comparable to dense-cache temporal
  filtered output on the 32-light and 128-light runs.
- Contact sheets show no obvious missing-shadow artifacts or temporal instability.
- 128-light selected-fast keeps a clear frame-time advantage.

If selected-fast is faster but visibly noisier, the correct next step is not a
kernel optimization. The correct next step is proposal quality: visibility-aware
selected proposal, hybrid cache/selected policy, or another estimator-side
variance reduction.

## Risks

- Dense-cache ReSTIR and selected-fast do not sample from the same proposal, so
  metric differences are expected. The goal is acceptability, not bitwise
  equality.
- A small asset/frame set may miss hard occlusion cases.
- Temporal filtering can hide single-frame noise in preview images, so metrics
  must include both initial and temporal filtered outputs.
- The all-light reference is still rasterized shadow-map visibility, not a
  physically exact reference.
