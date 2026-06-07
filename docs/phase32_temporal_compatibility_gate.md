# Phase 32: Conservative Temporal History Compatibility Gate

Phase 32 tightens the active aligned ReSTIR renderer's temporal history acceptance. The renderer still uses the same diffuse target, geometric proposal, scene-stable world lights, initial RIS, and previous-frame temporal reservoir reuse. The change is only the history gate.

## Motivation

Phase 27 made light indices stable across frames by using world-space lights. That fixes the sample-domain identity for lights, but a reprojected previous pixel can still represent the wrong surface. Depth agreement alone is not always enough for pseudo G-buffers, especially near silhouettes, thin structures, and expected-depth blends.

The active temporal lookup now separates:

```text
pre_gate_mask = old depth-based reprojection acceptance
valid_mask    = pre_gate_mask plus compatibility gates
```

If history is rejected, temporal output falls back exactly to the current initial RIS output.

## Gate Policy

For each current pixel, the previous reservoir is accepted only when:

```text
current pixel is valid
previous projected pixel is in bounds
previous G-buffer pixel is valid
relative depth error <= depth_tolerance
world-space normal dot >= temporal_normal_threshold
mean abs RGB distance <= temporal_rgb_threshold
motion magnitude <= temporal_max_motion_pixels
```

The active defaults are:

```text
depth_tolerance = 0.05
temporal_normal_threshold = 0.85
temporal_rgb_threshold = 0.20
temporal_max_motion_pixels = 32.0
```

Optional gates can be disabled from the renderer CLI with `none`, for example:

```powershell
$env:RESTIRGS_RESTIR_EXTRA_ARGS="--temporal-normal-threshold none"
scripts\run_aligned_restir_renderer_windows.bat
```

## Metrics

Renderer rows now include both old and final temporal acceptance counts:

```text
pre_gate_pixels
pre_gate_fraction
reuse_pixels
reuse_fraction
mean_relative_depth_error
mean_temporal_normal_dot
mean_temporal_rgb_distance
mean_motion_pixels
```

This makes it visible whether temporal reuse is being rejected by conservative compatibility checks or by basic reprojection/depth validity.

## Limitations

The gate is deliberately conservative and deterministic. It is not a learned gate, not an oracle, and not a new estimator. It does not add visibility, shadows, denoising, spatial reuse, or a Blinn-Phong temporal target. Temporal RIS is still not required to beat current-frame initial RIS for correctness; the goal is cleaner sample semantics.
