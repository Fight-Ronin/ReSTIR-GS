# Phase 48: Interactive Viewer Timing And Lazy Compute

## Purpose

Phase 48 adds timing instrumentation and view-scoped lazy compute to the interactive viewer without changing active renderer behavior.

The goal is to separate viewer latency into major GPU stages, then avoid computing buffers that the current viewer mode does not display.

## Run A Timing Smoke

```powershell
$env:RESTIRGS_VIEWER_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VIEWER_WIDTH="256"
$env:RESTIRGS_VIEWER_HEIGHT="256"
$env:RESTIRGS_VIEWER_EXTRA_ARGS="--save-and-exit --save-visibility"
scripts\run_interactive_viewer_windows.bat
```

This display smoke saves the visibility RIS image without computing an all-lights reference. Use `--save-visibility-reference` when reference/error images are needed.

Outputs:

```text
outputs/interactive_viewer/interactive_viewer_save_summary.json
outputs/interactive_viewer/current_camera.json
outputs/interactive_viewer/current_*.png
```

The summary and camera metadata include:

```text
render_rgbd_gpu_ms
gbuffer_gpu_ms
world_lights_gpu_ms
diffuse_restir_gpu_ms
blinn_phong_gpu_ms
proposal_confidence_gpu_ms
visibility_gpu_ms
total_gpu_ms
total_wall_ms
```

The saved metadata also records `computed_views`, so a run can tell whether it rendered only RGB/G-buffer layers, Blinn-Phong, or the optional visibility outputs.

## Lazy Compute Policy

The interactive viewer now renders only the buffers required by the selected view:

```text
RGB / alpha / depth / normal -> gsplat RGB+ED + pseudo G-buffer only
Lambertian                 -> base buffers + diffuse RIS/reference path
Blinn-Phong                -> base buffers + Blinn-Phong shader
save-and-exit              -> Blinn-Phong save render, plus visibility RIS only with --save-visibility
save-and-exit reference    -> visibility reference/error only with --save-visibility-reference
```

Switching view modes triggers a targeted rerender for that mode. Saving from the UI performs a separate Blinn-Phong save render so normal RGB/G-buffer inspection does not carry save-output cost.

## Current Observation

The 256x256 save-and-exit visibility smoke shows the largest viewer costs in:

```text
visibility_gpu_ms
blinn_phong_gpu_ms
```

`gsplat` RGB+D rendering and pseudo G-buffer generation are much smaller. Lazy compute removes unrelated RIS/visibility work from ordinary view navigation, and the display/reference split avoids all-lights reference cost during ordinary visibility display saves. Progressive preview remains a possible future interaction improvement.

## Non-goals

- No active renderer behavior change.
- No RIS/proposal/visibility math change.
- No progressive preview yet.
- No CUDA kernel fusion yet.
