# Phase 37: Visibility Target As Preferred Active Renderer

Phase 37 promotes the visibility-aware target from an optional diagnostic into the preferred active renderer path. This is a runner and documentation policy change; the reusable renderer API still supports the diffuse compatibility baseline.

Current active runner defaults were refined after this phase: Phase 42 adds conservative `temporal_filtered_ris`, uses `temporal_history_m_cap=1`, and produces `72` renderer rows for the four-asset testing set. Phase 43 tested but did not retain an aggressive pre-gate filter. The visibility target/proposal policy below remains the active target policy.

## Active Renderer Policy

The active Windows renderer runner now defaults to:

```text
target_mode=visibility
proposal=visibility_geometric
visibility_shadow_pcf_radius=1
temporal_history_m_cap=1
num_lights=16
render_size=128x128
frame_indices=45,46,47
output_dir=outputs/aligned_restir
```

The visibility target is:

```text
visible_contribution = lambertian_diffuse * shadow_visibility
```

The proposal follows the target:

```text
target_mode=diffuse    -> proposal=geometric
target_mode=visibility -> proposal=visibility_geometric
```

## Commands

Run the active workflow:

```powershell
scripts\run_active_validation_windows.bat
```

Run only the active renderer:

```powershell
scripts\run_aligned_restir_renderer_windows.bat
```

Run the retained diffuse baseline:

```powershell
$env:RESTIRGS_RESTIR_TARGET_MODE="diffuse"
$env:RESTIRGS_RESTIR_NUM_LIGHTS="128"
$env:RESTIRGS_RESTIR_WIDTH="256"
$env:RESTIRGS_RESTIR_HEIGHT="256"
$env:RESTIRGS_RESTIR_FRAME_INDICES="manifest"
$env:RESTIRGS_RESTIR_OUTPUT_DIR="outputs\aligned_restir_diffuse"
scripts\run_aligned_restir_renderer_windows.bat
```

## Runner Overrides

`scripts\run_aligned_restir_renderer_windows.bat` supports:

```text
RESTIRGS_RESTIR_TARGET_MODE
RESTIRGS_RESTIR_NUM_LIGHTS
RESTIRGS_RESTIR_WIDTH
RESTIRGS_RESTIR_HEIGHT
RESTIRGS_RESTIR_FRAME_INDICES
RESTIRGS_RESTIR_OUTPUT_DIR
RESTIRGS_RESTIR_VISIBILITY_SHADOW_PCF_RADIUS
RESTIRGS_RESTIR_TEMPORAL_HISTORY_M_CAP
RESTIRGS_RESTIR_TEMPORAL_FILTER_BLEND_MAX
RESTIRGS_RESTIR_TEMPORAL_FILTER_CLAMP_SCALE
RESTIRGS_RESTIR_TEMPORAL_FILTER_CLAMP_MIN
RESTIRGS_RESTIR_EXTRA_ARGS
```

Set `RESTIRGS_RESTIR_FRAME_INDICES=manifest` to use the manifest temporal window.
Set `RESTIRGS_RESTIR_VISIBILITY_SHADOW_PCF_RADIUS=0` to reproduce the legacy hard shadow-map visibility target.

## Expected Validation

The active renderer summary should record:

```text
target_mode=visibility
proposal=visibility_geometric
visibility_shadow_pcf_radius=1
temporal_history_m_cap=1
temporal_filtered_ris rows present
row_count=72
all_numeric_finite=true
```

The diffuse compatibility summary should record:

```text
target_mode=diffuse
proposal=geometric
all_numeric_finite=true
```

## Boundary

This phase does not add a new estimator, shadow algorithm, temporal policy, spatial reuse, or ablation sweep. It only changes which already-validated target is treated as the preferred active renderer output.
