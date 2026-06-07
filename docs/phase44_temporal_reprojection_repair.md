# Phase 44: Temporal Reprojection Neighborhood Repair

Phase 44 refines temporal correspondence without changing the visibility target, proposal, RIS weights, temporal filter, world lights, shadow maps, or G-buffer.

## Idea

The old temporal lookup used one rounded previous-frame pixel:

```text
current pixel -> projected previous location -> nearest previous pixel
```

That is fragile near silhouettes, splat noise, and subpixel camera motion. Phase 44 keeps the same strict compatibility gates, but searches a tiny previous-frame neighborhood and chooses the best compatible pixel:

```text
current pixel -> projected previous location -> 3x3 previous-pixel candidates -> best strict-compatible candidate
```

## Selection

The active default is:

```text
temporal_reprojection_search_radius = 1
```

Candidate scoring uses existing diagnostics:

```text
score = depth_term + normal_term + rgb_term + motion_term + tiny_offset_penalty
```

Only candidates that pass the existing hard gates can become accepted temporal history:

```text
relative_depth_error <= depth_tolerance
normal_abs_dot >= temporal_normal_threshold
rgb_distance <= temporal_rgb_threshold
motion_pixels <= temporal_max_motion_pixels
```

If no compatible candidate exists, the lookup remains invalid and downstream temporal outputs fall back exactly as before.

## Compatibility

Set:

```text
--temporal-reprojection-search-radius 0
```

to recover the old nearest-neighbor behavior. The active Windows runner also exposes:

```text
RESTIRGS_RESTIR_TEMPORAL_REPROJECTION_SEARCH_RADIUS
```

## Interpretation

This refinement tries to increase strict reuse coverage by repairing correspondence, not by letting lower-quality history through. It is deliberately different from the rejected Phase 43 soft pre-gate filter trial.

## Active Readout

On the four-asset active validation set, the first accepted run recorded:

```text
reuse_fraction_mean:  0.0246 -> 0.0487
pre_gate_mean:        0.1430 -> 0.1789
initial_ris MAE:      0.00278594295862907
filtered_ris MAE:     0.0027849741660854
```

The aggregate filtered output improved, with one small per-asset regression and three per-asset improvements. This is a useful refinement, not a final temporal solution.
