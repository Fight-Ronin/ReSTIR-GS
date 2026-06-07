# Current Milestone Snapshot

This document records the current stable project state after the visibility target was promoted to the preferred active renderer path. It is a status snapshot, not a new algorithm phase.

## Stable Active Path

The default active renderer path is:

```text
aligned asset registry
-> dataset adapter
-> load_gaussian_asset
-> aligned camera render with RGB + expected depth + alpha
-> pseudo G-buffer
-> scene-stable world lights
-> visibility-aware direct-lighting target
-> visibility-geometric proposal
-> initial RIS
-> compatibility-gated previous-frame temporal reuse
```

Default renderer settings remain conservative:

- Target: visibility-aware Lambertian contribution.
- Proposal: visibility-geometric proposal.
- Lights: world-space, asset-scaled, stable across aligned frames.
- Temporal reuse: previous frame only, gated by depth, world normal, RGB, and motion compatibility.
- Fallback rule: rejected or missing history must exactly match current-frame initial RIS.

## Diffuse Compatibility Path

The older diffuse Lambertian renderer remains available as a compatibility/debug baseline:

```powershell
$env:RESTIRGS_RESTIR_TARGET_MODE="diffuse"
$env:RESTIRGS_RESTIR_NUM_LIGHTS="128"
$env:RESTIRGS_RESTIR_WIDTH="256"
$env:RESTIRGS_RESTIR_HEIGHT="256"
$env:RESTIRGS_RESTIR_FRAME_INDICES="manifest"
$env:RESTIRGS_RESTIR_OUTPUT_DIR="outputs\aligned_restir_diffuse"
scripts\run_aligned_restir_renderer_windows.bat
```

It should record `target_mode=diffuse` and `proposal=geometric`.

Relevant outputs:

```text
outputs/aligned_visibility/
outputs/aligned_visibility_ris/
outputs/aligned_visibility_matrix/
outputs/aligned_restir_visibility/
```

## Interactive Inspection

The interactive viewer is now registry-aware and can inspect any registered aligned asset. Its main modes are:

```text
1: RGB
2: G-buffer
3: lighting
4: single-frame ReSTIR inspection
5: optional visibility target inspection
```

The visibility mode builds a separate small scene-stable light set and shadow-map bundle for inspection only. Saved visibility previews are written under:

```text
outputs/interactive_viewer/current_visibility_reference.png
outputs/interactive_viewer/current_visibility_ris.png
outputs/interactive_viewer/current_visibility_error.png
```

## Current Validation Commands

Use this command for the default active path:

```powershell
scripts\run_active_validation_windows.bat
```

Use this command for deeper visibility diagnostics:

```powershell
scripts\run_visibility_validation_windows.bat
```

Use this command for interactive inspection:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
scripts\run_interactive_viewer_windows.bat
```

For a non-interactive visibility viewer smoke:

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

## Current Readout

- The four-asset DXGL aligned testing set loads, renders, and produces finite smoke metrics.
- The visibility target is now the preferred active renderer target.
- The retained diffuse aligned renderer is the compatibility/debug baseline.
- The visibility target uses a `visibility_geometric` proposal that multiplies geometric proposal mass by shadow visibility.
- The interactive viewer visibility panel follows the same visibility-geometric proposal policy.
- The viewer is a debugging and human-inspection tool, not a benchmark.
- There is no need for more broad ablation before the next algorithmic change.

## Sensible Next Work

Good next steps should be narrow and tied to what inspection shows:

- If visibility-aware target remains stable, refine shadow quality or temporal usefulness inside the visibility renderer.
- If viewer usability becomes the bottleneck, add asset/frame cycling controls rather than new evaluation logic.
- If presentation or reporting is needed, generate a compact contact-sheet report from the current outputs.
- If algorithm work resumes, keep the active diffuse renderer unchanged while introducing one opt-in variant at a time.
