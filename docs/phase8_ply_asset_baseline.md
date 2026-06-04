# Phase 8: Real 3DGS PLY Asset Baseline

This phase moves the single-frame rendering and proposal baseline from five synthetic Gaussians to a real 3D Gaussian Splatting `.ply` asset.

## Supported PLY Schema

The loader expects a `vertex` element with:

```text
x, y, z
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
```

Color is read from the first available source:

```text
f_dc_0, f_dc_1, f_dc_2
red, green, blue
r, g, b
```

`f_rest_*` fields and normals are intentionally ignored in this phase.

## Field Conversion

The GraphDECO-style fields are converted as:

```text
means = [x, y, z]
scales = exp(scale_*)
opacities = sigmoid(opacity)
quats = normalize([rot_0, rot_1, rot_2, rot_3])
rgb = clamp(f_dc * 0.2820947918 + 0.5, 0, 1)
```

For RGB fields, values above `1` are treated as `0..255` and divided by `255`.

## Auto Camera

The demo builds a look-at camera from a robust Gaussian mean bounding box. By default it uses the central `98%` of loaded means to avoid a small number of outlier Gaussians pulling the camera very far away:

```text
target = central-percentile bbox center
eye = target + [0, 0, -radius]
radius = 1.4 * central-percentile bbox diagonal
focal = width * 1.25
```

The camera follows the project convention: world-to-camera view matrix, camera looks along `+Z`.

The initial full-bbox rule was useful as a smoke test, but real 3DGS assets often contain sparse outliers. The runner exposes the camera controls through the Python demo:

```powershell
python scripts\demo_07_ply_asset_baseline.py --ply path\to\point_cloud.ply --camera-bbox-percentile 0.98 --camera-radius-scale 1.4
```

Use `--camera-bbox-percentile 1.0` to recover the original full-bbox behavior.

## Asset-scaled Lights

The Phase 3 synthetic light generator uses a fixed small camera-space volume. For real assets, the Phase 8 demo instead places deterministic lights around the current rendered G-buffer extent and scales intensity by the squared scene scale:

```text
light xy center = valid G-buffer position center
light z range   = valid G-buffer depth range expanded by a small margin
intensity_l     = 3 * scene_scale^2 / num_lights
```

This keeps the single-frame lighting baseline numerically meaningful when the camera-space asset depth is tens of units instead of the synthetic scene's two or three units.

## Outputs

Run with:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
scripts\run_ply_asset_windows.bat
```

The demo writes:

```text
outputs/ply_rgb.png
outputs/ply_depth.png
outputs/ply_alpha.png
outputs/ply_normal.png
outputs/ply_deferred_composite.png
outputs/ply_geometric_mc_composite.png
outputs/ply_geometric_ris_composite.png
outputs/ply_asset_summary.json
```

The JSON records scene counts, color source, auto-camera parameters, valid pixel count, and finite error metrics against the all-lights reference.

## Limitations

This phase does not support view-dependent SH color, visibility, shadows, temporal reuse, spatial reuse, training, export tooling, or Nerfstudio integration. Large `.ply` files should stay outside the repository and be passed through `RESTIRGS_PLY`.
