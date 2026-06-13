# Phase 50: ReSTIR Visibility Optimization Plan

## Purpose

Phase 50 turns the active visibility ReSTIR path from ad-hoc profiling into a
measured optimization workflow.

The active renderer should continue to focus on visibility-aware direct
lighting with ReSTIR-style candidate sampling and temporal filtering. This
phase is not a retreat to G-buffer-only rendering, and it is not a new lighting
model.

## Current Verified State

The first small optimization is correctness-preserving contribution reuse in
the visibility target path.

Before this cleanup, the same sampled candidate lights could evaluate selected
visible diffuse contribution twice when both the proposal MC baseline and
initial RIS were requested. The renderer now evaluates those sampled candidate
contributions once and passes the tensor into both cached estimators.

Verified properties:

```text
cached estimator public APIs still work without precomputed contributions
visibility renderer calls the sampled selected-light evaluator once
proposal MC baseline and initial RIS reuse the same sampled contribution tensor
non-timing output rows match the previous smoke exactly
```

Recent checks:

```text
pytest tests\test_visibility_restir.py tests\test_restir_renderer.py
pytest
tiny dxgl_apple visibility smoke: 12 rows, all_numeric_finite = true
pre/post non-timing CSV comparison: 636 fields, 0 mismatches, max delta 0
```

This verifies semantics, not a stable speedup.

## Timing Caveat

Single `demo_26_aligned_restir_renderer.py` runs are useful smoke tests but are
not reliable proof of a small optimization. The timing fields moved together
across unrelated stages during repeated runs, especially when other tests were
running or CUDA extension/cache work was nearby.

Do not claim performance wins from one full-script run. Use it only to confirm
that the real aligned asset path still runs and writes finite rows.

## Next Stage

The next stage is to build a narrow CUDA microbenchmark for the visibility
ReSTIR hot path before taking the second optimization step.

Benchmark these operations separately with warmup, synchronization, repeated
CUDA events, and stable scene inputs:

```text
make_shadow_visibility_cache
compute_visibility_geometric_proposal_distribution_cached
evaluate_selected_light_visible_diffuse_cached for sampled candidates
estimate_visibility_ris_initial_lighting_cached
temporal combine contribution evaluation in visibility mode
```

Use fixed inputs first:

```text
asset_id = dxgl_apple
frames = 49, 50
target_mode = visibility
width = 128
height = 128
num_lights = 8
candidate_count = 4
visibility_shadow_resolution = 64
```

Then repeat at a more representative active setting after the benchmark harness
is stable.

## Success Criteria

The microbenchmark is useful only if it reports stable per-stage medians and
keeps renderer semantics pinned.

Required output:

```text
benchmark config: asset, frame, resolution, light count, candidate count
per-stage median / p90 / min / max GPU ms
full smoke row count and finite status
optional non-timing row equality against a saved baseline
```

Required correctness checks:

```text
unit tests still pass
full pytest still passes
tiny real-asset smoke remains finite
non-timing metrics remain unchanged for correctness-preserving optimizations
```

## Candidate Second Optimization

Choose the next implementation only after the microbenchmark identifies the
dominant local cost.

Likely branches:

```text
If proposal dominates:
  optimize all-light visibility-geometric proposal construction.
  Avoid unnecessary allocation or gather work where possible.

If sampled candidate contribution dominates:
  fuse or batch selected diffuse contribution and cached visibility gather for
  sampled candidates.

If all-lights reference dominates:
  keep it clearly evaluation-only, and avoid pulling reference cost into display
  or interactive paths.

If temporal combine dominates:
  focus on the visibility contribution evaluator used during temporal reservoir
  combine, not temporal threshold tuning.
```

## Non-goals

```text
No ray tracing rewrite.
No G-buffer-only cleanup.
No Lambertian/Blinn-Phong objective debate in this phase.
No temporal alpha or threshold tuning.
No claim of speedup without a stable benchmark.
No CUDA kernel fusion until the Python/Torch hot path is measured clearly.
```

