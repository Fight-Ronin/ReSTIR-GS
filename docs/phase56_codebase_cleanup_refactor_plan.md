# Phase 56: Codebase Cleanup And Refactor Plan

## Purpose

This plan is a cautious cleanup follow-up to the project wrap-up in
`docs/phase55_project_wrap_up.md`.

The goal is not to change renderer behavior. The goal is to keep the repository
easy to understand after the selected-fast visibility experiments:

- remove only clearly stale code;
- keep diagnostic code that still has tests, scripts, or viewer usage;
- avoid speculative refactors unless future work needs the extension point;
- keep the default renderer policy conservative.

## Current Assessment

The current active path is still:

```text
registered aligned assets
-> gsplat RGB + expected-depth render
-> pseudo G-buffer
-> world-space lights
-> dense shadow-map visibility cache
-> visibility-geometric proposal
-> initial RIS
-> previous-frame temporal reuse/filter
```

The selected-fast visibility path is intentionally separate:

```text
scripts/bench_realtime_display_fps.py --experimental-selected-visibility
scripts/eval_selected_fast_quality.py
```

It should remain an explicit experiment unless a future asset-matrix validation
shows a robust quality/performance policy.

## Applied Decisions

The cleanup decisions for this pass are:

- Keep `gs_gen/tools/download_tnt_single.py`.
- Delete compatibility wrappers and script aliases.
- Keep Lambertian and Blinn-Phong material diagnostic views.

## Cleanup Classification

### Safe cleanup

These are small, behavior-preserving changes with clear evidence.

1. Remove `shade_deferred_lambertian_unshadowed`. Completed.

   It is a thin compatibility wrapper around `shade_deferred_lambertian`, and
   current scans did not find active call sites.

2. Align README viewer wording with active docs. Completed.

   The viewer keys can still list Lambertian and Blinn-Phong, but they should be
   described as diagnostics rather than active optimization targets.

3. Keep stale-symbol checks in the cleanup workflow.

   The useful search is:

   ```powershell
   rg -n "selected_proposal|mix_proposal|selected_candidate_visibility_reuse|visibility_topk|uniform_mix|evaluate_selected_light_visible_diffuse_from_visibility|evaluate_initial_visible_candidates" restir_gs scripts tests
   ```

### Guarded cleanup

These items may be removable, but they should not be deleted casually.

1. `gs_gen/tools/download_tnt_single.py`

   This is a standalone helper for old external video acquisition. It is not
   part of the active renderer or core `gs_gen` command surface.

   Decision:

   ```text
   keep
   ```

2. Compatibility wrappers and script aliases.

   `scripts/demo_22_interactive_viewer.py` was a tiny compatibility entrypoint
   for `interactive.launcher`. It has been removed; use
   `python -m interactive.launcher` or the Windows viewer runner directly.

3. Material diagnostic views.

   Lambertian and Blinn-Phong are not active research targets. However, they are
   still used by the viewer, smoke matrix, sampling benchmark, and tests. Do not
   remove them unless we also remove those surfaces intentionally.

   Decision:

   ```text
   keep
   ```

## Refactor Candidates

These are not recommended for the current wrap-up commit. They become useful
only if development continues.

### Split `restir_gs/lighting/visibility.py`

Status: completed as a compatibility-preserving module split.

Current issue:

`visibility.py` combines shadow-map construction, dense visibility cache,
selected-only visibility, and visible-lighting wrappers.

Suggested split:

```text
restir_gs/lighting/shadow_maps.py
  ShadowMapBundle
  make_shadow_map_bundle
  make_light_camera

restir_gs/lighting/shadow_visibility.py
  ShadowVisibilityCache
  make_shadow_visibility_cache
  gather_shadow_visibility
  evaluate_shadow_visibility
  evaluate_shadow_visibility_selected_dense
  evaluate_shadow_visibility_selected_dense_fast

restir_gs/lighting/visible_lighting.py
  evaluate_selected_light_visible_diffuse*
  shade_deferred_lambertian_visible*

restir_gs/lighting/visibility.py
  compatibility facade for existing imports
```

Refactor trigger:

This split was done because selected-fast visibility made the old single module
too broad for follow-up work. Existing `restir_gs.lighting.visibility` imports
are preserved through the facade.

### Split `restir_gs/restir/renderer.py`

Status: completed as a compatibility-preserving support-module split.

Current issue:

`renderer.py` contains result dataclasses, timing, display/evaluation renderer
core, temporal filtering, metric-row generation, and summary helpers.

Suggested split:

```text
restir_gs/restir/renderer.py
  render_restir_frame
  evaluate_restir_display_frame_from_gbuffer
  evaluate_restir_frame_from_gbuffer
  compatibility re-exports for existing renderer imports

restir_gs/restir/types.py
  RestirRenderSettings
  RestirFrameTimings
  result dataclasses

restir_gs/restir/temporal_filter.py
  TemporalFilterStats
  empty_temporal_filter_stats
  apply_confidence_clamped_temporal_filter

restir_gs/restir/metrics.py
  make_restir_metric_rows
  summarize_restir_rows
  summarize_restir_timing_rows
```

Refactor trigger:

This split was done after the lighting split because the renderer module had
become the last broad mixed-responsibility file on the active ReSTIR path. The
active frame pipeline remains in `renderer.py`; support types, temporal filter
helpers, and metric summaries now live in focused modules. Existing
`restir_gs.restir.renderer` imports are preserved.

### Keep selected-fast outside renderer defaults

The selected-fast render path currently lives in
`scripts/bench_realtime_display_fps.py`. That is acceptable for a closeout
experiment.

Do not move it into `RestirRenderSettings` unless we decide selected-fast is a
maintained renderer mode. Moving it now would make an experimental quality-risky
path look like a supported policy.

## Recommended Sequence

### Step 1: Small cleanup patch

Scope:

```text
remove unused shade_deferred_lambertian_unshadowed
update README viewer key wording to say diagnostics
remove scripts/demo_22_interactive_viewer.py
keep gs_gen/tools/download_tnt_single.py
```

Verification:

```powershell
python -m py_compile restir_gs/lighting/visibility.py
python -m pytest tests/test_lighting.py tests/test_visibility_lighting.py -q
rg -n "shade_deferred_lambertian_unshadowed" restir_gs scripts tests interactive
rg -n "demo_22_interactive_viewer" README.md docs/current_architecture.md scripts tests
```

### Step 2: No-op architecture boundary check

Scope:

```text
confirm current_architecture.md matches README and scripts/README.md
confirm selected-fast is documented as experimental only
confirm material diagnostics are not described as active objectives
```

Verification:

```powershell
rg -n "selected-fast|experimental-selected-visibility|diagnostic" README.md docs scripts/README.md
```

### Step 3: Renderer support split

The lighting visibility split and renderer support split have been completed.
This kept the active frame pipeline stable while moving support code to:

```text
restir_gs/restir/types.py
restir_gs/restir/temporal_filter.py
restir_gs/restir/metrics.py
```

Do not combine this with another broad module split in the same cleanup pass.

Verification:

```powershell
python -m pytest tests/test_visibility_lighting.py tests/test_visibility_restir.py tests/test_restir_renderer.py -q
python scripts/bench_realtime_display_fps.py --help
python scripts/eval_selected_fast_quality.py --help
```

## Non-Goals

- No renderer default change.
- No new proposal policy.
- No ray tracing work.
- No Lambertian/Blinn-Phong material expansion.
- No broad test rewrite.
- No additional large module split in this cleanup pass.

## Stop Criteria

Stop cleanup when:

- active docs agree on the default renderer policy;
- stale selected-proposal symbols are absent from code paths;
- selected-fast is clearly marked as experimental;
- remaining diagnostic code has an explicit owner: viewer, smoke matrix,
  benchmark, or tests.
