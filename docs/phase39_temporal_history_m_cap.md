# Phase 39: Temporal History M Cap

Phase 39 keeps the active visibility renderer, visibility-geometric proposal,
and temporal compatibility gates unchanged. It only limits how much accepted
history can dominate the current initial reservoir.

## Motivation

The active renderer showed valid but low temporal reuse. When history was
accepted, the carried reservoir `M` could be larger than the current frame's
initial candidate count `K`. That gives history more influence than the current
pixel's fresh samples.

Rather than adding soft compatibility curves or new gates, this phase uses a
small defensive cap:

```text
current_M_eff = current_M
history_M_eff = min(previous_M, temporal_history_m_cap)
```

The temporal combine weights remain:

```text
candidate_weight = target_current(light) * reservoir.W * M_eff
W = weight_sum / (combined_M_eff * selected_target)
```

## Active Default

In the renderer, `temporal_history_m_cap=None` means:

```text
effective cap = candidate_count
```

For the Phase 39 renderer default this was:

```text
candidate_count = 8
temporal_history_m_cap = 8
```

The current active Windows runner was later tightened by Phase 43 to pass
`temporal_history_m_cap=1` while using aggressive confidence-clamped temporal
filtering as the preferred temporal output.

Low-level `combine_temporal_reservoirs(..., history_m_cap=None)` still preserves
the uncapped behavior for direct compatibility tests and older callers.

## Scope

This phase does not change:

- visibility target,
- visibility-geometric proposal,
- RIS target,
- temporal reprojection or compatibility gates,
- spatial reuse,
- datasets or benchmarks.

It is a conservative temporal refinement: history can help, but accepted history
cannot carry more effective mass than the current initial reservoir by default.
