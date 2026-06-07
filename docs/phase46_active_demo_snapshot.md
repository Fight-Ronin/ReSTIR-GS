# Phase 46: Active Renderer Demo Snapshot

## Purpose

Phase 46 does not change the renderer algorithm. It packages the current active renderer output into a compact inspection artifact:

```text
active renderer CSV/JSON/PNGs
-> demo contact sheet
-> compact performance + quality summary
```

This gives us a stable handoff point for human inspection and performance discussion without adding another ablation.

## Inputs

The snapshot consumes existing active renderer outputs:

```text
outputs/aligned_restir/restir_renderer_rows.csv
outputs/aligned_restir/restir_renderer_summary.json
outputs/aligned_restir/<asset_id>/final_*.png
```

Run active validation first if those files are missing:

```powershell
scripts\run_active_validation_windows.bat
```

## Run

```powershell
scripts\run_active_demo_snapshot_windows.bat
```

Outputs:

```text
outputs/active_demo/active_renderer_snapshot_contact.png
outputs/active_demo/active_renderer_snapshot_summary.json
```

## What It Reports

The summary records:

- active policy: target mode, proposal, preferred output, light count, temporal settings
- validation flags: finite metrics, row count, timing availability
- contribution-error summary by estimator
- GPU timing stage rank from Phase 45 timing fields
- per-asset timing inherited from the active renderer summary

The contact sheet shows, per asset:

```text
reference
initial RIS
temporal filtered RIS
filtered error
filter alpha
reuse mask
```

## Non-goals

- No algorithm changes.
- No extra renders.
- No new ablation or benchmark sweep.
- No replacement for the interactive viewer.
