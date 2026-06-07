# Phase 25: Interactive Aligned 3DGS Viewer

Phase 25 adds a lightweight local viewer for compatible 3DGS splats. The default validation target is the active aligned DXGL Apple asset, but the viewer entrypoint also accepts any GraphDECO/Nerfstudio-style 3DGS `.ply` supported by the generic loader. The viewer is a research/debugging instrument: it lets us free-move a camera around a splat, inspect render layers, and save a replayable camera config for later scripts.

It does not add spatial reuse, temporal reuse, new proposal distributions, or benchmark rows. The viewer can optionally inspect the current initial RIS and visibility-aware direct-light target for debugging.

## Interactive Camera

The reusable camera state is still serialized through `restir_gs/render/orbit_camera.py`:

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

`interactive/camera.py` adds free-camera movement helpers on top of that state: local forward/back, left/right, up/down translation, plus a free-look helper for future clients. The current matplotlib client keeps left-drag orbit for stable object inspection. The final renderer input remains a normal `PinholeCamera`, so the backend renderer API does not change.

## Viewer Flow

The default DXGL viewer command is:

```powershell
scripts\run_interactive_viewer_windows.bat
```

Internally it runs:

```text
load Gaussian splat
-> load DXGL aligned cameras or one generic PLY camera
-> initialize interactive camera state from the selected camera
-> render the active layer requirements
-> build pseudo G-buffer
-> create asset-scaled camera-space point lights only for lighting layers
-> display a single matplotlib viewport
```

RGB, alpha, depth, and normal share the base G-buffer render. Lambertian and Blinn-Phong are triggered with `5` and `6`; the session only computes them when the selected layer needs that backend path.

Default settings:

```text
frame_index = 49
resolution = 768x768
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

To trade quality for faster interaction, override the runner resolution:

```powershell
$env:RESTIRGS_VIEWER_WIDTH="512"
$env:RESTIRGS_VIEWER_HEIGHT="512"
scripts\run_interactive_viewer_windows.bat
```

The runner calls the Visual Studio x64 setup and configures the `gsplat` JIT environment. Direct `python ... --device cuda` is only appropriate from an x64 Native Tools shell or after manually calling `vcvars64.bat`.

The underlying Python entrypoint is:

```powershell
python -m interactive.launcher --ply C:\path\to\splat.ply --device cuda
```

This uses a conservative auto-camera from the Gaussian mean bbox. To replay an existing camera:

```powershell
python -m interactive.launcher --ply C:\path\to\splat.ply --camera-config outputs\interactive_viewer\current_camera.json --device cuda
```

With the runner:

```powershell
$env:RESTIRGS_VIEWER_PLY="C:\path\to\splat.ply"
$env:RESTIRGS_VIEWER_CAMERA_CONFIG="outputs\interactive_viewer\current_camera.json"
scripts\run_interactive_viewer_windows.bat
```

## Browser WebUI Prototype

The browser prototype uses the same `InteractiveSession` and backend renderer, but serves a static browser UI through FastAPI:

```powershell
scripts\run_interactive_web_viewer_windows.bat
```

It listens on `http://127.0.0.1:8765` by default. Override `RESTIRGS_WEB_HOST` or `RESTIRGS_WEB_PORT` when needed. Direct generic PLY mode mirrors the matplotlib entrypoint:

The WebUI runner defaults to `1024x1024` so the viewport fills more of a desktop browser. Override `RESTIRGS_VIEWER_WIDTH` and `RESTIRGS_VIEWER_HEIGHT` to trade quality for faster interaction.

```powershell
python -m interactive.web_server --ply C:\path\to\splat.ply --device cuda
```

After the browser connects, the WebUI observes the viewport size and asks the server to re-render at that canvas size, clamped to `64..2048` pixels per axis. This keeps the backend render buffer aligned with the visible WebUI stage instead of letterboxing a fixed square render.

For browser display only, RGB, Lambertian, and Blinn-Phong views are sent as RGBA PNGs and composited over the WebUI stage background. Alpha, depth, and normal remain opaque diagnostic buffers. Saved outputs and the matplotlib inspector keep the backend render images unchanged.

## Controls

```text
W / S                     move forward / backward
A / D                     move left / right
Shift / Ctrl              move up / down
Left drag                 orbit yaw/pitch
Middle drag               pan camera target
Shift + left drag         pan camera target
Mouse wheel               dolly focus distance
[ / ]                     previous / next DXGL frame, reset to that aligned camera
1                         RGB layer
2                         Alpha layer
3                         Depth layer
4                         Normal layer
5                         Lambertian layer
6                         Blinn-Phong layer
r                         reset to current DXGL frame camera
Ctrl + S                  save current camera and preview images
q                         quit
```

## Saved Outputs

Pressing `Ctrl + S` writes:

```text
outputs/interactive_viewer/current_camera.json
outputs/interactive_viewer/current_rgb.png
outputs/interactive_viewer/current_alpha.png
outputs/interactive_viewer/current_normal.png
outputs/interactive_viewer/current_blinn_phong.png
```

When saving with `--save-and-exit --save-visibility`, the viewer also writes the display-side visibility RIS image:

```text
outputs/interactive_viewer/current_visibility_ris.png
```

This path does not compute an all-lights visibility reference. For explicit debug/evaluation output, use `--save-and-exit --save-visibility-reference`; that additionally writes:

```text
outputs/interactive_viewer/current_visibility_reference.png
outputs/interactive_viewer/current_visibility_error.png
```

`current_camera.json` uses the same minimal camera payload shape as existing camera config loaders, with an extra `orbit_camera_state` field for debugging. Existing loaders ignore that extra field.

For non-interactive validation:

```powershell
python -m interactive.launcher --save-and-exit
```

This renders the default view once, saves the same outputs, and exits without opening a window.

## Limitations

- The viewer is debug-usable, not a real-time renderer.
- Lights are regenerated as camera-space asset-scaled lights for the current view, matching the current deferred lighting demos.
- Visibility mode uses a separate scene-stable world-light set and expected-depth shadow-map proxy. It is for inspection, not a benchmark.
- The saved camera is useful for replaying aligned render/debug scripts, but it is not a benchmark row.
- The viewer should not be used to draw sampling conclusions; Phase 24 remains the active aligned sampling readout.
