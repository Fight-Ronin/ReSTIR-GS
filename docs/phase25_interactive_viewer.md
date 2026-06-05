# Phase 25: Interactive Aligned 3DGS Viewer

Phase 25 adds a lightweight local viewer for compatible 3DGS splats. The default validation target is the active aligned DXGL Apple asset, but the viewer entrypoint also accepts any GraphDECO/Nerfstudio-style 3DGS `.ply` supported by the generic loader. The viewer is a research/debugging instrument: it lets us orbit, pan, and dolly around a splat, inspect render buffers, and save a replayable camera config for later scripts.

It does not add RIS, spatial reuse, temporal reuse, visibility, new proposal distributions, or new estimators.

## Orbit Camera

The reusable camera state lives in `restir_gs/render/orbit_camera.py`:

```text
OrbitCameraState(
  target,
  yaw_degrees,
  pitch_degrees,
  radius,
  focal_scale,
  width,
  height
)
```

Conversion to `PinholeCamera` uses the existing `look_at_viewmat` convention:

```text
world-to-camera view matrix
camera +Z forward
image +Y down through the pinhole intrinsics
```

The orbit utilities and viewer render loop are dataset-agnostic. DXGL Apple is only the current default validation asset.

## Viewer Flow

The default DXGL viewer command is:

```powershell
scripts\run_interactive_viewer_windows.bat
```

Internally it runs:

```text
load Gaussian splat
-> load DXGL aligned cameras or one generic PLY camera
-> initialize orbit state from the selected camera
-> render RGB+expected-depth+alpha
-> build pseudo G-buffer
-> create asset-scaled camera-space point lights
-> shade Lambertian and Blinn-Phong
-> display a 2x3 matplotlib panel grid
```

Default settings:

```text
frame_index = 49
resolution = 256x256
device = cuda
lights = 128
ambient = 0.2
specular_strength = 0.15
shininess = 24
```

For a generic compatible 3DGS PLY:

```powershell
$env:RESTIRGS_VIEWER_PLY="C:\path\to\splat.ply"
scripts\run_interactive_viewer_windows.bat
```

The runner calls the Visual Studio x64 setup and configures the `gsplat` JIT environment. Direct `python ... --device cuda` is only appropriate from an x64 Native Tools shell or after manually calling `vcvars64.bat`.

The underlying Python entrypoint is:

```powershell
python scripts/demo_22_interactive_viewer.py --ply C:\path\to\splat.ply --device cuda
```

This uses a conservative auto-camera from the Gaussian mean bbox. To replay an existing camera:

```powershell
python scripts/demo_22_interactive_viewer.py --ply C:\path\to\splat.ply --camera-config outputs\interactive_viewer\current_camera.json --device cuda
```

With the runner:

```powershell
$env:RESTIRGS_VIEWER_PLY="C:\path\to\splat.ply"
$env:RESTIRGS_VIEWER_CAMERA_CONFIG="outputs\interactive_viewer\current_camera.json"
scripts\run_interactive_viewer_windows.bat
```

## Controls

```text
Left drag                 orbit yaw/pitch
Middle drag               pan target
Shift + left drag         pan target
Mouse wheel               dolly in/out
[ / ]                     previous / next DXGL frame, reset to that aligned camera
1                         RGB overview mode
2                         G-buffer mode
3                         lighting mode
r                         reset to current DXGL frame camera
s                         save current camera and preview images
q                         quit
```

## Saved Outputs

Pressing `s` writes:

```text
outputs/interactive_viewer/current_camera.json
outputs/interactive_viewer/current_rgb.png
outputs/interactive_viewer/current_alpha.png
outputs/interactive_viewer/current_normal.png
outputs/interactive_viewer/current_blinn_phong.png
```

`current_camera.json` uses the same minimal camera payload shape as existing camera config loaders, with an extra `orbit_camera_state` field for debugging. Existing loaders ignore that extra field.

For non-interactive validation:

```powershell
python scripts/demo_22_interactive_viewer.py --save-and-exit
```

This renders the default view once, saves the same outputs, and exits without opening a window.

## Limitations

- The viewer is debug-usable, not a real-time renderer.
- Lights are regenerated as camera-space asset-scaled lights for the current view, matching the current deferred lighting demos.
- The saved camera is useful for replaying aligned render/debug scripts, but it is not a benchmark row.
- The viewer should not be used to draw sampling conclusions; Phase 24 remains the active aligned sampling readout.
