# Phase 53: Selected Visibility Kernel Optimization Plan

## Final Status

The candidate-flat Torch implementation was sufficient for the selected-fast
experiment, so no custom CUDA kernel was kept as a required next step. Final
project scope and cleanup decisions are summarized in
`docs/phase55_project_wrap_up.md`.

## Goal

Phase 52 showed that full dense visibility cache construction is the dominant
cost at 128 lights, but the first selected-only implementation is still too
slow:

```text
dxgl_apple, 256x256, K=8, medians:

mode          lights  fps    frame_ms  dense_cache_ms  selected_vis_ms
cache         32      25.06   39.91     18.94           0.00
selected      32      12.47   80.21      0.02          48.46
cache        128       7.68  130.23    103.98           0.00
selected     128      11.96   83.64      0.03          42.39
```

Phase 53 should optimize selected shadow visibility for `[H, W, K]` candidates
without changing the default renderer. The purpose is to answer one narrow
question:

```text
Can selected-only visibility become fast enough to be the high-light-count
real-time path?
```

## Current Correctness Audit

- The default visibility renderer still builds and uses a dense visibility
  cache.
- The selected-only path is only reachable through the benchmark
  `--experimental-selected-visibility` flag.
- The selected shadow helper matches the existing dense visibility path for the
  same candidate indices, including PCF and invalid selected indices.
- RIS can now accept precomputed contribution candidates without a visibility
  cache, but it still raises if both the cache and precomputed candidates are
  missing.
- Full regression currently passes: `217 passed`.

## Non-Goals

- Do not introduce ray tracing.
- Do not replace the default visibility-geometric renderer path in this phase.
- Do not remove the dense visibility cache.
- Do not change proposal semantics and kernel optimization in the same step.
- Do not fuse RIS, proposal sampling, temporal reuse, and visibility into one
  large kernel yet.

## Kernel Contract

The first optimized primitive should compute selected shadow visibility only:

```text
selected_shadow_visibility(
  world_positions: [H, W, 3] float32 CUDA,
  valid_mask: [H, W] bool CUDA,
  light_indices: [H, W, K] int64 CUDA,
  viewmats: [N, 4, 4] float32 CUDA,
  intrinsics: [N, 3, 3] float32 CUDA,
  shadow_depth: [N, S, S] float32 CUDA,
  shadow_alpha: [N, S, S] float32 CUDA,
  depth_bias: float,
  alpha_threshold: float,
  pcf_radius: int
) -> visibility: [H, W, K] float32 CUDA
```

The optimized result must match `evaluate_shadow_visibility_selected_dense`
within a small fp32 tolerance. `evaluate_selected_light_diffuse` can remain as
the diffuse contribution path until selected shadow visibility is no longer the
bottleneck.

## Implementation Stages

### Stage 0: Protect the Reference

Before optimizing, keep the current selected helper as the reference
implementation.

- Add a CUDA-only correctness test that compares the new optimized path against
  `evaluate_shadow_visibility_selected_dense`.
- Cover hard shadows, PCF radius `1`, alpha-threshold soft blockers,
  out-of-bounds projections, behind-light points, invalid candidate indices, and
  `K=1` / `K=8`.
- Keep CPU behavior on the current reference path.

### Stage 1: Candidate-Flat Torch Prototype

Try a minimal vectorized torch implementation before adding a custom extension.

- Flatten candidates to `[H * W * K]`.
- Gather selected light camera rows once.
- Project selected candidates without the current Python loop over candidate
  slots.
- Reuse the existing opacity-aware depth compare semantics.
- Benchmark this prototype against the current selected helper.

This stage may still materialize large temporary tensors, so it is a probe, not
the final design.

### Stage 2: Custom CUDA Kernel Only If Needed

If Stage 1 does not reduce selected visibility enough, add a small custom CUDA
kernel with one output element per `(pixel, candidate)` and an inner loop over
PCF taps.

Kernel behavior:

- Return zero for invalid pixels, invalid candidate indices, out-of-bounds
  shadow coordinates, and non-positive light-space depth.
- Use the same projection convention as the dense path:
  `u = x * fx / z + cx`, `v = y * fy / z + cy`, then nearest texel via
  `round`.
- Use the same opacity-aware visibility:
  depth pass returns `1`; failed depth returns `1 - normalized_alpha`.
- Average all PCF taps uniformly.
- Support `pcf_radius=0` and `pcf_radius=1` first, since `1` is the benchmark
  default.

Because the repo currently has no standalone `.cu/.cpp` extension files, keep
the extension isolated behind a Python wrapper and preserve fallback behavior.
Do not make the whole project depend on successful kernel compilation.

## Integration Plan

1. Add `evaluate_shadow_visibility_selected_dense_fast(...)`.
   - Same inputs and output shape as the reference selected helper.
   - Dispatches to optimized CUDA only when tensors are CUDA float32 and the
     kernel is available.
   - Falls back to `evaluate_shadow_visibility_selected_dense` otherwise.

2. Add a benchmark switch.
   - Keep `--experimental-selected-visibility` behavior unchanged by default.
   - Add a second switch such as
     `--selected-visibility-impl reference|fast`.
   - Report `selected_visibility_impl` in JSON/CSV.

3. Update selected visible diffuse.
   - Use the fast shadow visibility helper only inside the experimental
     selected-only benchmark path.
   - Do not route the default renderer through it yet.

## Verification Plan

Correctness checks:

```text
conda run -n restirgs python -m pytest tests\test_visibility_lighting.py -q
conda run -n restirgs python -m pytest tests\test_visibility_restir.py tests\test_restir_renderer.py -q
conda run -n restirgs python -m pytest -q
```

Benchmark checks:

```text
python scripts/bench_realtime_display_fps.py --asset-ids dxgl_apple --width 256 --height 256 --num-lights 32 --candidate-count 8 --warmup-iters 1 --repeat-iters 3 --experimental-selected-visibility --selected-visibility-impl fast --device cuda

python scripts/bench_realtime_display_fps.py --asset-ids dxgl_apple --width 256 --height 256 --num-lights 128 --candidate-count 8 --warmup-iters 1 --repeat-iters 3 --experimental-selected-visibility --selected-visibility-impl fast --device cuda
```

Compare against the Phase 52 reference selected-only outputs:

- `selected_candidate_visibility_gpu_ms`
- `initial_ris_gpu_ms`
- `temporal_ris_gpu_ms`
- `frame_gpu_ms`
- `estimated_gpu_fps`

## Success Criteria

- Correctness tests match the reference selected helper within fp32 tolerance.
- No change to default visibility-cache renderer outputs or tests.
- At 128 lights, selected candidate visibility drops from roughly `42 ms` to
  below `10 ms`.
- At 128 lights, end-to-end selected-only frame time drops below `60 ms`.
- At 32 lights, selected-only remains clearly labeled experimental unless it is
  within `10%` of the dense-cache path.

## Stage 1 Result

The candidate-flat Torch prototype was enough to make the selected-only path
competitive. It removes the Python loop over candidate slots and evaluates all
`[H, W, K]` selected candidates in one projected tensor path.

```text
dxgl_apple, 256x256, K=8, medians:

mode            lights  fps    frame_ms  selected_vis_ms  initial_ris_ms  temporal_ris_ms
cache           32      25.06   39.91      0.00            3.52            4.63
selected_ref    32      12.47   80.21     48.46           51.46           17.31
selected_fast   32      28.44   35.17      7.05           10.19           10.33

cache          128       7.68  130.23      0.00            3.15            4.18
selected_ref   128      11.96   83.64     42.39           45.82           16.85
selected_fast  128      21.61   46.27      8.63           11.70           12.66
```

Interpretation:

- Stage 1 meets the main performance target: 128-light selected visibility is
  below `10 ms`, and end-to-end selected-only frame time is below `60 ms`.
- The 32-light selected-fast path is also slightly faster than the dense-cache
  path in this benchmark.
- `temporal_ris_gpu_ms` is now the largest selected-only visibility-related
  cost after proposal construction, because temporal reuse evaluates current and
  previous reservoir candidates.
- This still should remain experimental until image quality/variance is checked
  against the visibility-geometric dense-cache path.

## Decision After Phase 53

If the fast selected path wins at high light counts but loses at low light
counts, use an adaptive policy later:

```text
small N: dense visibility cache
large N: selected-only visibility
```

If the fast selected path still loses after kernel optimization, keep the dense
cache path and shift optimization effort to dense cache construction or shadow
bundle generation instead.

## Risks

- The selected-only experiment uses geometric proposal, not
  visibility-geometric proposal, so variance and image quality can differ.
- Temporal reuse evaluates visibility for current and previous reservoir
  candidates; end-to-end speedup depends on both initial and temporal paths.
- A custom extension may introduce build/cache fragility on Windows.
- Fusing too much too early can hide correctness regressions inside performance
  work.
