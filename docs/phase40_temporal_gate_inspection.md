# Phase 40: Temporal Gate Inspection

Phase 40 does not change the renderer algorithm. It exposes why accepted
temporal history is much smaller than the old depth-only pre-gate population.

The active temporal acceptance is still:

```text
valid_history =
    pre_gate
    and normal_gate_pass
    and rgb_gate_pass
    and motion_gate_pass
```

where:

```text
pre_gate = current valid
        and previous projection in bounds
        and previous G-buffer valid
        and relative depth error <= depth_tolerance

normal_gate_pass = pre_gate and abs(normal_dot) >= temporal_normal_threshold
rgb_gate_pass    = pre_gate and rgb_distance <= temporal_rgb_threshold
motion_gate_pass = pre_gate and motion_pixels <= temporal_max_motion_pixels
```

## New Renderer Row Fields

The aligned ReSTIR renderer now writes these per-row diagnostics:

```text
normal_gate_pass_pixels
normal_gate_pass_fraction
normal_gate_pass_pre_gate_fraction
rgb_gate_pass_pixels
rgb_gate_pass_fraction
rgb_gate_pass_pre_gate_fraction
motion_gate_pass_pixels
motion_gate_pass_fraction
motion_gate_pass_pre_gate_fraction
mean_pre_gate_normal_dot
mean_pre_gate_rgb_distance
mean_pre_gate_motion_pixels
```

Existing fields remain unchanged:

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

The `*_pre_gate_fraction` fields are the most useful first readout because they
answer: "among pixels that passed reprojection and depth, which compatibility
gate rejected history?"

## New Final-Frame Debug Images

For each asset, the renderer also saves:

```text
final_pre_gate_mask.png
final_normal_reject_mask.png
final_rgb_reject_mask.png
final_motion_reject_mask.png
final_reuse_mask.png
```

The reject masks are sequential:

```text
normal_reject = pre_gate and not normal_gate_pass
rgb_reject    = pre_gate and normal_gate_pass and not rgb_gate_pass
motion_reject = pre_gate and normal_gate_pass and rgb_gate_pass and not motion_gate_pass
```

This partitions the final-frame compatibility rejections into readable causes.

## Interpretation

- Low `pre_gate_fraction`: reprojection, depth, frame spacing, or G-buffer
  validity is the bottleneck.
- Low `normal_gate_pass_pre_gate_fraction`: pseudo normal tangent planes are
  unstable across aligned frames or the normal threshold is too strict.
- Low `rgb_gate_pass_pre_gate_fraction`: pseudo albedo/RGB differs too much
  under view changes or visibility changes.
- Low `motion_gate_pass_pre_gate_fraction`: projected motion is too large for
  the current motion cap or frame spacing.
- High gate pass rates but low improvement: history is accepted cleanly, but the
  reservoir sample itself is not helping enough.

This phase is an observability step. It does not change target mode, proposal,
RIS weights, world lights, visibility, or the temporal combine equation.
