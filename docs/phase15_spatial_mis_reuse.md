# Phase 15: Defensive Spatial MIS Candidate Reuse

## Goal

Phase 14 showed that same-frame spatial reuse has recoverable signal, but the current reservoir combine harms too many pixels. The best oracle keep-best variant improves over initial RIS, while every actual Phase 13 spatial variant still loses to initial RIS.

Phase 15 tests a more conservative estimator:

```text
reuse neighbor proposal candidates, not compressed neighbor reservoir weight
```

This is experimental but kept because it showed a concrete real-asset error reduction on the selected playroom view. It does not add temporal reuse, shadows, visibility, new proposal distributions, or learned gates.

## Estimator

For a current pixel `x`, the all-light diffuse target is:

```text
I_x = sum_l f_x(l)
```

Phase 6 already computes a per-pixel geometric proposal:

```text
q_x(l) ∝ intensity_l * luminance(color_l) * abs(dot(N_x, wi_xl)) / dist_xl^2
```

For each current pixel, Phase 15 gathers the center pixel and valid 3x3 neighbor proposal distributions:

```text
q_s(l),  s in S_x
```

Each source gets a compatibility weight `beta_s`. The center source has a defensive floor:

```text
beta_center >= center_floor
sum_s beta_s = 1
```

Neighbor raw compatibility is:

```text
compat = exp(
  -normal_penalty * (1 - dot(N_x, N_s))
  -depth_penalty * relative_depth(x, s)
  -rgb_penalty * rgb_distance(x, s)
)
```

Neighbors still need to pass the same basic local-validity checks:

```text
normal_dot >= normal_threshold
relative_depth <= depth_tolerance
rgb_distance <= rgb_threshold, when rgb_threshold is set
```

The current-pixel mixture proposal is:

```text
Q_x(l) = sum_s beta_s q_s(l)
```

The defensive floor ensures:

```text
Q_x(l) >= beta_center * q_x(l)
```

This is the main harm-mitigation mechanism: neighbor samples cannot dominate the current pixel as easily as Phase 12's `W_neighbor * M_neighbor` reservoir reuse.

## MIS MC

For `K` samples from each source proposal:

```text
l_{s,k} ~ q_s
```

The MIS estimate is:

```text
I_hat_x = sum_s beta_s / K * sum_k f_x(l_{s,k}) / Q_x(l_{s,k})
```

All light contributions are re-evaluated at the current pixel.

## Default Variants

The real-asset demo evaluates:

```text
geometry_floor_0_50
geometry_floor_0_75
geometry_floor_0_90
rgb_floor_0_50
rgb_floor_0_75
rgb_floor_0_90
```

The RGB variants use `rgb_penalty=8.0`; they do not use a hard RGB threshold by default.

## Outputs

Run:

```powershell
$env:RESTIRGS_PLY="C:\path\to\point_cloud.ply"
$env:RESTIRGS_CAMERA_CONFIG="outputs\camera_probe_selected_camera.json"
scripts\run_spatial_mis_reuse_windows.bat
```

Outputs:

```text
outputs/spatial_mis_ablation.csv
outputs/spatial_mis_ablation_summary.json
outputs/spatial_mis_best_composite.png
outputs/spatial_mis_best_abs_error.png
outputs/spatial_mis_initial_abs_error.png
```

The CSV/JSON record per-variant diffuse RGB error, reuse fraction, accepted-neighbor count, average center/neighbor proposal weights, and improve/harm fractions relative to initial RIS.

## Interpretation

If a defensive MIS variant beats initial RIS diffuse MAE, then same-frame spatial candidate reuse is viable once neighbor samples are proposal-corrected.

If all defensive MIS variants still lose to initial RIS, but harm fraction drops, then spatial reuse has signal but needs a better confidence model or more samples.

If both MAE and harm remain poor, the current pseudo G-buffer likely does not provide enough same-frame local compatibility signal, and the next research direction should shift toward temporal reuse, visibility, or a better target/proposal.
