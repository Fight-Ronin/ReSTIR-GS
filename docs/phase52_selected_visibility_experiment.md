# Phase 52: Selected-Only Visibility Experiment

## Final Status

This phase remains as the correctness and first benchmark record for selected
candidate visibility. The selected-only path stays experimental; final project
scope and cleanup decisions are summarized in
`docs/phase55_project_wrap_up.md`.

## Goal

Phase 51 moved proposal sampling back to CUDA and reduced the 128-light display
frame from roughly `342 ms` to `130 ms`. The new bottleneck is the full
per-frame visibility cache:

```text
128 lights, 256x256, K=8:
  frame_gpu_ms median             130.2
  visibility_cache_gpu_ms median  104.0
  proposal_gpu_ms median           11.6
```

The next question is whether real-time display needs full `[H, W, N]`
visibility every frame. Phase 52 is a contained experiment to measure an
alternative:

```text
geometric proposal over all lights
-> sample K candidates
-> evaluate shadow visibility only for those selected candidates
-> initial RIS / temporal reuse
```

This changes the proposal policy, so it must stay experimental until measured.

## Non-Goals

- Do not replace the default visibility-geometric renderer path.
- Do not remove the dense visibility cache.
- Do not claim identical variance or image quality.
- Do not introduce ray tracing.

## Implementation Plan

1. Add a selected-only shadow visibility helper.
   - Input shape remains `[H, W, K]` light indices.
   - It should match `evaluate_shadow_visibility` numerically for the same
     selected light indices.
   - It should avoid projecting all `N` lights when only `K` are requested.

2. Add an experimental benchmark mode.
   - Keep the existing benchmark default unchanged.
   - Add a flag for selected-only visibility.
   - Report the proposal mode in CSV/JSON.
   - Report selected visibility timing separately from dense visibility cache
     timing.

3. Compare the same `dxgl_apple` settings:

```text
python scripts/bench_realtime_display_fps.py --asset-ids dxgl_apple --width 256 --height 256 --num-lights 128 --candidate-count 8 --warmup-iters 1 --repeat-iters 3 --device cuda

python scripts/bench_realtime_display_fps.py --asset-ids dxgl_apple --width 256 --height 256 --num-lights 128 --candidate-count 8 --warmup-iters 1 --repeat-iters 3 --experimental-selected-visibility --device cuda
```

## Success Criteria

- Selected-only visibility helper matches the current dense visibility result
  for sampled candidate indices.
- Existing tests still pass.
- The experimental benchmark writes finite CSV/JSON rows.
- The experiment shows whether frame time is limited by full visibility cache
  construction or by selected visibility evaluation and temporal logic.

## Current Result

The first naive implementation is correct enough to run the benchmark, but it
should not replace the default path.

```text
dxgl_apple, 256x256, K=8, medians:

mode          lights  fps    frame_ms  dense_cache_ms  selected_vis_ms
cache         32      25.06   39.91     18.94           0.00
selected      32      12.47   80.21      0.02          48.46
cache        128       7.68  130.23    103.98           0.00
selected     128      11.96   83.64      0.03          42.39
```

Interpretation:

- The selected-only experiment removes the full dense cache bottleneck.
- At 128 lights, this is already faster overall: `130.23 ms -> 83.64 ms`.
- At 32 lights, it is worse: `39.91 ms -> 80.21 ms`.
- The naive selected helper spends most of the saved time on per-pixel selected
  view/projection gathers, so it proves the direction but not the implementation.

The next useful step is a CUDA/vectorized selected shadow-visibility kernel
that evaluates `[H, W, K]` candidates directly, then reruns this exact A/B
benchmark before changing renderer defaults.

## Risks

- The proposal becomes geometric, not visibility-geometric. Occluded high-power
  lights may be sampled more often, increasing variance.
- Temporal reuse still needs visibility for previous-frame selected reservoirs.
- Selected-only projection may have different memory access patterns and may
  not scale linearly with `K` until optimized further.
