# Phase 51: Real-Time Display FPS Optimization Plan

Status: this phase is historical. The deferred selected-only visibility idea was
implemented and measured as an explicit experiment in phases 52-55. The final
default renderer policy remains dense-cache visibility-geometric; selected-fast
is retained only for profiling and quality validation.

## Context

The active project direction is the rasterized G-buffer/ReSTIR path, not a
ray-tracing renderer. The display target remains visibility-aware direct
lighting over registered 3D Gaussian Splatting assets:

```text
gsplat RGB + expected-depth render
-> pseudo G-buffer
-> world-space lights
-> shadow-map visibility proxy
-> visibility-geometric proposal
-> initial RIS
-> previous-frame temporal reuse
-> confidence-clamped temporal-filtered display
```

The project value is not to make a physically complete light transport engine.
It is to understand whether a practical rasterized 3DGS renderer can expose
enough visibility/G-buffer semantics for ReSTIR-style reuse while remaining
interactive.

## Current Measurement

`scripts/bench_realtime_display_fps.py` measures the display-only renderer
path without the all-lights reference evaluator.

Baseline on `dxgl_apple`, `256x256`, `K=8`, visibility target:

```text
32 lights:
  median frame_gpu_ms          54.1
  median estimated_gpu_fps     18.5
  median visibility_cache_ms   18.6
  median proposal_ms           20.0

128 lights:
  median frame_gpu_ms         341.8
  median estimated_gpu_fps      2.9
  median visibility_cache_ms  109.3
  median proposal_ms          168.2
```

The bottleneck is therefore not the RGB splat render. It is the per-frame
all-light visibility/proposal path.

## First Optimization Step

Proceed with the smallest semantics-preserving proposal optimization:

1. Keep the active proposal policy as `visibility_geometric`.
2. Keep the full `[H, W, N]` proposal distribution for now.
3. Avoid unnecessary dense visibility gathers when the cache is already dense.
4. Sample candidates on GPU when the proposal lives on CUDA.
5. Split timing into proposal distribution and proposal sampling sub-stages.

This deliberately does not introduce a new proposal policy. It should only make
the existing policy cheaper and easier to profile.

## Expected Benefit

The most suspicious current cost is:

```text
proposal_probs.detach().cpu()
torch.multinomial(flat_probs, ...)
copy sampled indices back to CUDA
```

For realtime display, this is a poor fit because every frame moves a large
`[H, W, N]` distribution through the CPU only to sample `K` candidates.

The first fast path should keep sampling on CUDA with a per-pixel CDF sampler:

```text
proposal_probs -> cumsum over lights -> random thresholds -> searchsorted
```

The selected candidates and gathered proposal probabilities should have the
same shapes and estimator contract as today.

## Correctness Criteria

Required checks:

```text
python -m pytest tests/test_proposal.py tests/test_visibility_restir.py tests/test_restir_renderer.py -q
python scripts/bench_realtime_display_fps.py --asset-ids dxgl_apple --width 64 --height 64 --num-lights 4 --candidate-count 2 --warmup-iters 0 --repeat-iters 1 --visibility-shadow-resolution 32 --output-dir outputs/realtime_display_fps_smoke --device cuda
```

Benchmark checks:

```text
python scripts/bench_realtime_display_fps.py --asset-ids dxgl_apple --width 256 --height 256 --num-lights 32  --candidate-count 8 --warmup-iters 1 --repeat-iters 3 --output-dir outputs/realtime_display_fps/dxgl_apple_256_visibility_32l_k8 --device cuda
python scripts/bench_realtime_display_fps.py --asset-ids dxgl_apple --width 256 --height 256 --num-lights 128 --candidate-count 8 --warmup-iters 1 --repeat-iters 3 --output-dir outputs/realtime_display_fps/dxgl_apple_256_visibility_128l_k8 --device cuda
```

Compare median `proposal_gpu_ms`, `proposal_distribution_gpu_ms`,
`proposal_sampling_gpu_ms`, and `frame_gpu_ms` against the pre-optimization
baseline above.

## Risks

- CUDA CDF sampling will not be bit-identical to CPU `torch.multinomial`.
  The distribution contract should match, but exact candidate indices may
  change.
- The full `[H, W, N]` proposal tensor still exists. This step does not solve
  all-light memory bandwidth by itself.
- The visibility cache remains expensive at high light counts.
- CUDA event timing can hide some host-side behavior, so wall timing should
  still be checked.

## Deferred Ideas

These are not part of the first step:

- approximate top-k light subsets;
- unshadowed geometric proposal with visibility only on sampled candidates;
- selected-only shadow projection path;
- lower-resolution visibility/proposal grids;
- temporal reuse of proposal distributions.

Those are likely useful, but each changes either semantics, variance, or the
display/evaluation contract. They should be considered only after the exact
fast path is measured.
