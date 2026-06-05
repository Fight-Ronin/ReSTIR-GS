# Phase 22: DXGL Aligned G-buffer Validation

Phase 22 validates the real aligned DXGL Apple geometry path before returning to deferred lighting or ReSTIR.

The inputs are:

```text
DXGL Apple RGB / mask / depth_16bit / normals / transforms.json
DXGL Apple pretrained splat
Phase 21 raw-to-splat camera normalization
```

The pipeline is:

```text
load normalized camera -> gsplat RGB+ED/alpha -> pseudo G-buffer
compare against aligned modalities
```

## Running

```powershell
scripts\run_dxgl_gbuffer_validation_windows.bat
```

Outputs:

```text
outputs/aligned_gbuffer/dxgl_apple_gbuffer_contact.png
outputs/aligned_gbuffer/dxgl_apple_gbuffer_summary.json
outputs/aligned_gbuffer/dxgl_apple_gbuffer_frame_<index>_*.png
```

## Metrics

The demo records:

```text
RGB MAE/RMSE/PSNR under DXGL mask
alpha-mask IoU/precision/recall
render expected-depth vs DXGL depth_16bit diagnostic
pseudo-normal display vs DXGL normal display diagnostic
```

Depth assumes `depth_16bit / 10000` is raw camera z-depth, then scales it by the Phase 21 scene scale to compare against normalized splat-space expected depth. This is a diagnostic assumption and is recorded in the summary.

Normal comparison is display-space only. The pseudo normals are camera-space screen-gradient normals; DXGL normal images may not have identical semantic space. The contact sheet is the primary normal sanity check.

## Gate

This phase is a gate before lighting/ReSTIR:

```text
alpha roughly matches mask
RGB render remains aligned
depth has sane magnitude and spatial structure
pseudo normals are non-empty and visually plausible
```

If these hold, the next phase can safely run deferred lighting and proposal baselines on the aligned DXGL asset. If not, fix G-buffer geometry first.
