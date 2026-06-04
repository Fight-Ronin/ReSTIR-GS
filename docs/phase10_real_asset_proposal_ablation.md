# Phase 10: Real-Asset Proposal Ablation

This phase runs the Phase 6 single-frame proposal ablation on a real 3DGS `.ply` asset using the Phase 9 selected camera.

## Workflow

Probe a stable camera first:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
scripts\run_camera_probe_windows.bat
```

Run the real-asset ablation from that selected view:

```powershell
$env:RESTIRGS_CAMERA_CONFIG="outputs\camera_probe_selected_camera.json"
scripts\run_real_asset_proposal_ablation_windows.bat
```

The ablation does not fall back to auto-camera. This keeps real-scene evaluation tied to a stable replayable camera config.

## Evaluation

The demo renders the selected camera, builds the pseudo G-buffer, creates asset-scaled camera-space point lights, then compares:

```text
uniform MC
uniform RIS
geometric MC
geometric RIS
```

against the all-lights deferred reference for both diffuse RGB and composite RGB.

Default sweep:

```text
K values             = 1, 2, 4, 8, 16, 32
seed count           = 8
candidate seed base  = 9100
selection seed base  = 10100
light count          = 128
max gaussians        = 200000
```

## Outputs

```text
outputs/real_asset_proposal_ablation.csv
outputs/real_asset_proposal_ablation_summary.json
```

The JSON records scene stats, selected camera metadata, valid lighting pixel count, asset-light metadata, sweep settings, row count, and grouped mean/std MAE/RMSE summaries.

## Limitations

This is a single-frame proposal-quality measurement. It does not include temporal reuse, spatial reuse, visibility, shadows, COLMAP camera loading, new proposal distributions, or plots.
