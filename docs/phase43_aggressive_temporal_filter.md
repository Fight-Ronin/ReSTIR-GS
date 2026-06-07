# Phase 43: Aggressive Soft Pre-Gate Temporal Filter Trial

Phase 43 tested a more aggressive temporal filter, then intentionally fell back to the conservative Phase 42 active path because the trial did not meet acceptance criteria.

## Trial

The trial kept `temporal_ris` strict but allowed `temporal_filtered_ris` to use pre-gate history:

```text
temporal_ris candidate = lookup.valid_mask
filter candidate = lookup.pre_gate_mask & current_valid
```

Normal, RGB, and motion diagnostics were used as soft confidence terms instead of hard filter rejects:

```text
confidence = min(depth_conf, normal_abs_dot, rgb_conf, motion_conf)
alpha = filter_candidate * temporal_filter_blend_max * confidence
```

The active trial defaults were:

```text
temporal_filter_blend_max = 0.35
temporal_filter_clamp_scale = 0.75
temporal_filter_clamp_min = 1e-5
```

## Result

The trial passed tests and produced finite active validation rows, but failed the quality acceptance criteria:

```text
initial_ris contribution MAE            = 0.00278594295862907
temporal_filtered_ris contribution MAE  = 0.002793073220649
filter alpha mean                       = 0.00486555437479789
```

The aggregate filtered MAE was worse than initial RIS, and the alpha mean did not reach the intended `0.01+` useful-history range. Per-asset deltas also showed visible risk concentration on `dxgl_apple` and `dxgl_fire_extinguisher`.

## Fallback

The active path remains the conservative Phase 42 filter:

```text
filter candidate = lookup.valid_mask
temporal_filter_blend_max = 0.15
temporal_filter_clamp_scale = 0.50
temporal_filter_clamp_min = 1e-5
temporal_history_m_cap = 1
```

This keeps the verified safe behavior while preserving the Phase 43 conclusion: pre-gate history has coverage, but its quality is not high enough to use directly through alpha/clamp tuning. A future temporal improvement should improve correspondence or visibility/motion confidence rather than simply increasing filter aggressiveness.
