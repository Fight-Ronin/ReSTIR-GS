# Phase 23: Dataset-Agnostic Blinn-Phong Deferred Lighting

Phase 23 adds a generic Blinn-Phong deferred lighting shader and a generic Gaussian asset loading surface. DXGL Apple is only the current aligned validation asset.

## Generic Gaussian Asset Loading

Use:

```python
load_gaussian_asset(path, device="cuda", max_gaussians=None, schema="auto")
```

V1 supports compatible 3DGS PLY files with:

```text
x/y/z
opacity
scale_0..2
rot_0..3
f_dc_0..2 or RGB fallback
```

The loader records whether `f_rest_*` exists, but rendering still uses DC color as pseudo albedo. Dataset-specific camera normalization remains outside this loader.

## Blinn-Phong Shader

Use:

```python
shade_deferred_blinn_phong(
    gbuffer,
    lights,
    ambient=0.2,
    specular_strength=0.15,
    shininess=24,
)
```

The shader computes camera-space point lighting:

```text
V = normalize(-position_cam)
L = light_pos_cam - position_cam
wi = normalize(L)
H = normalize(wi + V)
diffuse = albedo * light * abs(dot(N, wi)) / dist2 / pi
specular = specular_strength * light * abs(dot(N, H))^shininess / dist2
composite = albedo * ambient + diffuse + specular
```

Normals are still pseudo screen-space normals, so specular defaults are intentionally conservative. `LightingBuffers.shade_rgb` remains the diffuse shading term for compatibility with the Lambertian path. The complete Blinn-Phong result is `composite_rgb`, and the specular term is exposed separately as `specular_rgb`.

## Optional RIS Target Probe

The default RIS/proposal estimators still use the diffuse target. Phase 23 also adds an opt-in target path for controlled experiments:

```python
estimate_ris_initial_lighting(..., target_mode="blinn_phong")
estimate_proposal_lighting(..., target_mode="blinn_phong")
```

This evaluates candidate contribution as `diffuse + specular`, then uses Rec.709 luminance of that contribution as the RIS target. The older `estimate_ris_initial_diffuse` and `estimate_proposal_diffuse` wrappers are unchanged and remain the source of truth for previous benchmark phases.

## DXGL Validation Demo

Run:

```powershell
scripts\run_dxgl_blinn_phong_lighting_windows.bat
```

Outputs:

```text
outputs/aligned_lighting/dxgl_blinn_phong_lighting_contact.png
outputs/aligned_lighting/dxgl_blinn_phong_lighting_summary.json
outputs/aligned_lighting/dxgl_lighting_frame_<index>_*.png
```

This demo validates visual behavior on the aligned DXGL Apple asset. The shader itself is dataset-agnostic and only requires `GBuffer + PointLights`.

## Scope

Blinn-Phong is an all-lights visual/reference baseline by default. The optional RIS target path is intended for small, controlled probes before changing any benchmark defaults.
