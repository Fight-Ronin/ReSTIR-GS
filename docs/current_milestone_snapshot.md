# Current Milestone Snapshot

## Status

The current baseline is an aligned, registry-driven ReSTIR-GS renderer over the DXGL testing asset set.

Active command:

```powershell
scripts\run_active_baseline_demo_windows.bat
```

Expected high-level result:

```text
aligned smoke matrix rows: 76
active renderer rows: 72
target_mode: visibility
proposal: visibility_geometric
all numeric metrics finite: true
timing_summary present: true
active demo snapshot present: true
```

## Active Algorithm

```text
3DGS render -> pseudo G-buffer
world-space lights -> camera-space lights
visibility-aware direct lighting reference
visibility-geometric proposal
initial RIS
strict previous-frame temporal reservoir combine
confidence-clamped previous filtered contribution
```

The preferred displayed temporal output is `temporal_filtered_ris`.
Performance readout is now attached to the active renderer output through CUDA-event timing fields. `frame_wall_ms` is retained only as wall-clock context.

## Display / Evaluation Split

The renderer now has two clear surfaces:

```text
display path     -> ReSTIR display buffers, no all-lights reference
evaluation path  -> display buffers + all-lights reference + metrics/error maps
```

The interactive viewer uses display-oriented output by default. `--save-visibility` writes the visibility RIS display image only; `--save-visibility-reference` explicitly computes and writes reference/error images.

## Current Temporal Policy

- Previous frame only.
- World-space light identity is stable across frames.
- Reprojection uses nearest projection plus a local 3x3 candidate repair.
- Acceptance uses depth, unoriented normal compatibility, RGB distance, and motion.
- `temporal_history_m_cap=1` is the active default.
- The image-space temporal filter is conservative: `blend_max=0.15`, `clamp_scale=0.50`.

## What Is Not Active

- No standalone ablation sweeps.
- No Voxel51 or single-view PLY benchmark path.
- No spatial MIS path.
- No temporal live carry inside the interactive viewer.
- No new visibility or target sweeps.

## Next Sensible Work

The codebase is clean enough to treat the current active path as the handoff baseline. Future work should be a deliberate larger change, such as GPU performance engineering for visibility proposal/RIS evaluation, better visibility semantics, broader aligned assets, or a production-quality viewer path. More small parameter tuning on the current four assets is not recommended.
