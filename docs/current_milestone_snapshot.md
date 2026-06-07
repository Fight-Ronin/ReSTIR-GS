# Current Milestone Snapshot

## Status

The current baseline is an aligned, registry-driven ReSTIR-GS renderer over the DXGL testing asset set.

Active command:

```powershell
scripts\run_active_validation_windows.bat
```

Expected high-level result:

```text
aligned smoke matrix rows: 76
active renderer rows: 72
target_mode: visibility
proposal: visibility_geometric
all numeric metrics finite: true
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

The codebase is clean enough to treat the current active path as the baseline. Future algorithm work should be a deliberate larger change, such as better visibility/correspondence or a new proposal, not more small parameter tuning on the current four assets.
