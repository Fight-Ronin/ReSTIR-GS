# Phase 27: Scene-Stable World-Space Lights

Phase 26 proved that aligned temporal reprojection and reservoir carry can run on DXGL Apple frames. It did not make temporal reuse fully meaningful as a quality experiment, because that demo generated point lights from each frame's camera-space G-buffer bounds.

With per-frame camera-space lights, the same `light_index` can refer to a different physical light in adjacent frames. Temporal reservoir reuse assumes the opposite: a carried sample id must remain in the same sample domain.

## Policy

Phase 27 introduces scene-stable world-space point lights:

```text
WorldPointLights(position_world, color, intensity)
```

The aligned temporal demo now generates one deterministic world-space light set from the loaded Gaussian means before the frame loop. Each frame converts that set into the existing camera-space `PointLights` API:

```text
position_cam = viewmat * position_world
```

The deferred shader, geometric proposal, RIS estimator, and temporal combine code continue to consume camera-space `PointLights`.

## Light Generation

`make_asset_scaled_world_lights` uses the robust Gaussian-mean bbox:

```text
center = 0.5 * (bbox_min + bbox_max)
radius = norm(0.5 * (bbox_max - bbox_min)) * radius_scale
```

Default settings:

```text
bbox_percentile = 0.98
radius_scale = 1.25
colors in [0.4, 1.0]
intensity_per_light = 3.0 * radius^2 / count
```

Positions are sampled deterministically on a spherical shell around the bbox center using a CPU `torch.Generator`, then moved to the requested device.

## Temporal Meaning

Reservoirs still store only `light_indices`, `W`, and `M`. After Phase 27:

```text
same light index across frames = same world-space light
```

This makes the Phase 26 combine equation semantically valid:

```text
candidate_weight = luminance(f_current(pixel, light_index)) * W * M
```

where `f_current` is evaluated at the current pixel after transforming the world light into the current camera frame.

## Limitations

This phase does not improve lighting realism, add visibility, change RIS targets, or prove temporal reuse improves error. It only fixes the sample-domain identity needed before temporal quality can be interpreted.

Existing single-frame demos keep their camera-space asset-scaled light helper so older baselines remain reproducible.
