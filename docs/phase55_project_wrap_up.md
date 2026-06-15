# Phase 55: Project Wrap-Up

## Scope

This project focused on improving the rasterized ReSTIR-GS visibility path for
real-time RGB rendering. The final scope is deliberately narrow:

- Keep dense shadow-map visibility and G-buffer/rasterization semantics.
- Keep selected-fast candidate visibility as an experimental speed path.
- Do not continue ray tracing, Lambertian/Blinn-Phong material expansion, or
  broad renderer policy work.
- Do not promote selected-fast or proposal variants to the default renderer.

## What Remains Useful

### Selected-fast candidate visibility

The selected-fast path is useful as an experimental high-light-count display
path. It evaluates visibility only for sampled candidates instead of building a
dense all-light visibility cache.

It remains valuable for profiling because it makes the tradeoff explicit:

```text
dense-cache:    stronger visibility-geometric proposal, expensive all-light cache
selected-fast:  cheaper selected candidate visibility, higher proposal variance
```

### Quality harness

`scripts/eval_selected_fast_quality.py` remains useful. It compares selected-fast
against the dense-cache all-light visible diffuse reference and writes CSV, JSON,
and contact-sheet artifacts.

This is the right place to validate any future rasterization-side visibility
experiment before touching renderer defaults.

## What Was Removed

The following experimental proposal knobs were removed after validation:

- Uniform proposal mixture.
- Top-k visibility-conditioned proposal mixture.
- Initial-candidate visibility reuse for top-k proposal candidates.

The reason is not that they never helped. The reason is that they did not
produce a stable cross-asset policy.

## Final 5-Asset Check

Fixed setup:

```text
resolution = 256x256
frames = 45..53
lights = 128
shadow_resolution = 128
baseline = selected-fast pure K=32
candidate = selected-fast K=8 + top32 visibility proposal mix 0.8 + reuse
```

Result:

```text
asset                 pure_mae  top_mae   top/pure  pure_fps  top_fps
dxgl_apple            0.007630  0.005602  0.73      12.67     10.80
dxgl_cash_register    0.012825  0.015988  1.25      13.83     11.87
dxgl_drill            0.001391  0.002549  1.83      13.08     14.91
dxgl_fire_extinguisher 0.007829 0.013048  1.67      16.60     14.51
dxgl_potted_plant     0.015192  0.015457  1.02      15.36     13.57
```

Interpretation:

- Top-k proposal helped strongly on `dxgl_apple`.
- It was roughly neutral on `dxgl_potted_plant`.
- It lost quality on `dxgl_cash_register`, `dxgl_drill`, and
  `dxgl_fire_extinguisher`.
- Therefore fixed `K=8/top32/mix0.8` is not a good default or recommended
  follow-up policy.

## Final Recommendation

Stop here for this project phase.

The useful artifact is not a new default renderer. The useful artifact is the
measured conclusion:

```text
Selected-only visibility can recover real-time performance at high light counts,
but robust quality requires better proposal semantics than the tested cheap
mixtures. The tested top-k visibility proposal is promising on some assets but
not stable enough to keep in the codepath.
```

The renderer should stay conservative:

- Default visibility path remains dense-cache visibility-geometric.
- Selected-fast remains an explicit experiment.
- Future work should start from the quality harness and use a small asset matrix
  before adding new renderer knobs.

