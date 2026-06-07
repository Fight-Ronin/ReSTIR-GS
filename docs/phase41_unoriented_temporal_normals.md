# Phase 41: Unoriented Temporal Normal Compatibility

Phase 41 changes one temporal compatibility detail: the normal gate compares
unoriented pseudo normals.

The old gate was:

```text
dot(N_current_world, N_previous_world) >= temporal_normal_threshold
```

The active gate is now:

```text
abs(dot(N_current_world, N_previous_world)) >= temporal_normal_threshold
```

## Why

The current G-buffer normal is a screen-space pseudo normal estimated from
expected-depth positions. It is useful for lighting and local geometry checks,
but its orientation is not a stable physical surface orientation across views.

For temporal history acceptance, the question is whether the current and
previous pixels look like the same local tangent plane. A reversed pseudo-normal
sign should not reject history by itself. A perpendicular normal should still be
rejected.

## Diagnostics

The renderer keeps both values:

```text
mean_temporal_normal_dot
mean_temporal_normal_abs_dot
mean_pre_gate_normal_dot
mean_pre_gate_normal_abs_dot
```

The signed dot helps diagnose orientation flips. The abs dot is the actual normal
compatibility value used by the gate.

## Scope

This does not change:

- target mode,
- proposal distribution,
- RIS weights,
- visibility,
- world-space light identity,
- RGB or motion compatibility gates,
- temporal history M cap.

It is a semantic cleanup for pseudo normal reuse, not a threshold sweep.
