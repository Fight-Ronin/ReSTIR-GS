# Phase 35: Visibility Target Smoke Matrix And Renderer Opt-In

Phase 35 keeps the default aligned renderer on the diffuse target, but adds a controlled opt-in path for the Phase 33/34 visibility-aware direct-light target.

The target remains:

```text
visible_contribution(light) = lambertian_diffuse(light) * shadow_visibility(light)
```

Visibility is computed by the expected-depth shadow-map proxy from Phase 33. This is a research/debug target, not a physically exact shadow model.

## What Changed

- Added a fixed multi-asset visibility smoke matrix over the registered aligned testing set.
- Added `target_mode="visibility"` as an opt-in renderer setting.
- Temporal combine can now use a custom contribution evaluator, so historical reservoirs are re-evaluated at the current pixel using the current visibility target.
- The default renderer path remains `target_mode="diffuse"`.

## Run Visibility Matrix

```powershell
scripts\run_aligned_visibility_smoke_matrix_windows.bat
```

To run the full optional visibility validation bundle:

```powershell
scripts\run_visibility_validation_windows.bat
```

This runs the matrix and then a short `target_mode="visibility"` renderer pass into `outputs/aligned_restir_visibility/`.

Useful overrides:

```powershell
$env:RESTIRGS_VISIBILITY_ASSET_SET="testing"
$env:RESTIRGS_VISIBILITY_MATRIX_EXTRA_ARGS="--width 128 --height 128 --candidate-count 8"
scripts\run_aligned_visibility_smoke_matrix_windows.bat
```

Outputs:

```text
outputs/aligned_visibility_matrix/visibility_smoke_matrix_rows.csv
outputs/aligned_visibility_matrix/visibility_smoke_matrix_summary.json
outputs/aligned_visibility_matrix/visibility_smoke_matrix_contact.png
outputs/aligned_visibility_matrix/<asset_id>/*.png
```

## Run Renderer With Visibility Target

The standard renderer runner already accepts extra args:

```powershell
$env:RESTIRGS_RESTIR_EXTRA_ARGS="--target-mode visibility --num-lights 16 --frame-indices 45,46,47"
scripts\run_aligned_restir_renderer_windows.bat
```

This writes to the normal renderer output folder:

```text
outputs/aligned_restir/
```

Use a short frame window first because visibility mode builds shadow maps for the selected lights.

## Interpretation

Passing this phase means:

- visibility target rows are finite across the aligned testing set,
- the target is non-degenerate enough to produce visible differences,
- the renderer can use visibility as an opt-in target without changing the default diffuse baseline.

It does not mean visibility should become the default target. That decision should wait until the visibility renderer path is inspected visually and compared against the existing diffuse renderer across several assets.
