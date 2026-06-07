# Phase 33: Visibility-Aware Direct Lighting Smoke

Phase 33 adds a small shadow-map visibility proxy for direct lighting. It is a smoke test for visibility semantics, not a replacement for the active aligned ReSTIR renderer.

## What It Does

For a small set of scene-stable world-space point lights, the smoke script renders low-resolution expected-depth shadow maps from the light positions. Current camera G-buffer positions are projected into those light cameras and tested with:

```text
visible = in_bounds
       & light_z > 0
       & (shadow_alpha <= alpha_threshold
          or light_z <= shadow_depth + depth_bias)
```

The visible Lambertian reference is:

```text
visible_diffuse = unshadowed_diffuse * visibility
```

Existing unshadowed Lambertian, Blinn-Phong, RIS, proposal, temporal reuse, and active validation defaults are unchanged.

## Run

```powershell
scripts\run_aligned_visibility_smoke_windows.bat
```

Optional controls:

```powershell
$env:RESTIRGS_VISIBILITY_ASSET_ID="dxgl_apple"
$env:RESTIRGS_VISIBILITY_EXTRA_ARGS="--frame-index 49 --num-lights 16 --shadow-resolution 128"
scripts\run_aligned_visibility_smoke_windows.bat
```

## Outputs

Outputs are written to:

```text
outputs/aligned_visibility/
```

The key artifacts are:

```text
visibility_unshadowed_reference.png
visibility_shadowed_reference.png
visibility_difference.png
visibility_debug_light_mask.png
visibility_contact.png
visibility_smoke_summary.json
```

The summary records valid pixels, visibility min/mean/max, whether the debug visibility mask is nontrivial, shadow settings, and shadowed-vs-unshadowed RGB error.

## Limitations

This is a nearest-neighbor expected-depth shadow proxy. It can self-shadow and it is not physically exact. It does not use PCF, soft shadows, visibility-aware proposals, visibility-aware temporal reuse, or any production shadow renderer. Its purpose is to verify that the project has a coherent visibility signal before adding visibility to RIS.
