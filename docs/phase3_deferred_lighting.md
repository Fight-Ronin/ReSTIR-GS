# Phase 3: Naive Deferred Point Lighting

This phase adds a deterministic all-lights baseline on top of the Phase 2 pseudo G-buffer. It is the reference path for later RIS and ReSTIR experiments.

## Inputs

The lighting pass consumes:

```text
pseudo albedo = gbuffer.rgb
position      = gbuffer.position_cam
normal        = gbuffer.normal_cam
valid pixels  = gbuffer.valid_mask & gbuffer.normal_mask
```

This is not a physically recovered material model. The RGB buffer is treated as pseudo albedo so the lighting path can be tested before adding real splat assets, inverse-rendered materials, or shadows.

## Point-light Equation

All lights are evaluated in camera space:

```text
L = light_pos_cam - position_cam
dist2 = dot(L, L) + eps
wi = normalize(L) with eps only as a zero-length guard
cos = abs(dot(normal_cam, wi))
irradiance += light_color * intensity * cos / dist2
shade = ambient + irradiance / pi
composite = pseudo_albedo * shade
```

Invalid pixels receive zero irradiance and diffuse lighting. Their composite output keeps the original RGB so silhouettes/background remain stable.

## Two-sided Normal Policy

Phase 2 normals are screen-space pseudo normals, flipped to `normal.z >= 0`. For this first lighting baseline, the diffuse term uses:

```text
abs(dot(N, wi))
```

This two-sided policy avoids making the whole baseline depend on an arbitrary pseudo-normal sign. Later phases can revisit one-sided or face-forward shading once real asset conventions are clearer.

## Deterministic Lights

`make_deterministic_point_lights(count=128, seed=2027)` creates reproducible camera-space lights:

```text
x,y in [-1.2, 1.2]
z   in [0.8, 3.8]
rgb in [0.4, 1.0]
intensity = 3.0 / count
```

The random tensors are generated on CPU with a fixed seed, then moved to the requested device. This makes tests and future ablations repeatable.

## Outputs

The demo writes:

```text
outputs/deferred_base_rgb.png
outputs/deferred_irradiance.png
outputs/deferred_diffuse.png
outputs/deferred_composite.png
outputs/deferred_normal.png
```

## Known Limits

- This is an all-lights reference, not RIS or ReSTIR.
- There is no visibility term, shadowing, temporal reuse, or spatial reuse.
- Lighting is camera-space for the synthetic identity-camera scene.
- Pseudo normals and pseudo albedo are sufficient for a vertical slice, but they are not true relightable 3DGS attributes.
