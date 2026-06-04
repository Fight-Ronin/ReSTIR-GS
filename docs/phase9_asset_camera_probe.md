# Phase 9: Real-Asset Camera Probe

This phase adds a repeatable view-selection pass for real 3DGS `.ply` assets. It stabilizes the camera before running real-scene lighting or proposal comparisons.

## Probe Grid

The probe starts from the Phase 8 robust asset bbox target and generates look-at cameras with camera `+Z` forward.

Default values:

```text
bbox percentile = 0.98
yaw degrees     = -30, -15, 0, 15, 30
pitch degrees   = -10, 0, 10
radius scales   = 0.9, 1.1, 1.3
candidate count = 45
```

The selected camera is saved as exact `viewmat` and `intrinsics` matrices so later runs replay the same view rather than recomputing it from probe parameters.

## Scoring

Each candidate renders RGB, expected depth, and alpha at `128x128`. Valid pixels are:

```text
alpha > 1e-4 and depth is finite and depth > 0
```

The score is:

```text
coverage = valid_pixels / total_pixels
central_coverage = valid pixels in center 50% crop / crop pixels
border_coverage = valid pixels on 8px border / border pixels
brightness = mean Rec.709 luminance over valid RGB pixels, or 0
score = coverage + 0.75 * central_coverage + 0.15 * brightness - 0.5 * border_coverage
```

This favors views that fill the image center and penalizes views that look clipped against the image boundary.

## Outputs

Run:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
scripts\run_camera_probe_windows.bat
```

The probe writes:

```text
outputs/camera_probe_contact.png
outputs/camera_probe_selected_rgb.png
outputs/camera_probe_selected_depth.png
outputs/camera_probe_selected_alpha.png
outputs/camera_probe_summary.json
outputs/camera_probe_selected_camera.json
```

Use the selected camera with the Phase 8 baseline:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
$env:RESTIRGS_CAMERA_CONFIG="outputs\camera_probe_selected_camera.json"
scripts\run_ply_asset_windows.bat
```

`camera_probe_summary.json` records every candidate's yaw, pitch, radius scale, camera position, and score. `camera_probe_selected_camera.json` is the replay artifact consumed by the PLY baseline.

## Limitations

This is still a heuristic view-selection pass. It does not read COLMAP cameras, optimize viewpoint quality, inspect semantic content, or solve visibility/shadowing. It exists to make real-asset baselines repeatable before adding real-scene ablations.
