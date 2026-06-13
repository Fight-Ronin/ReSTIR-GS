# Local Data

This directory is the common local data root for ReSTIR-GS.

Expected layout:

```text
data/
  dxgl/       DXGL Polyhaven datasets, zips, splats, and intake summaries
  gs_gen/     local gs_gen source captures, frame sequences, and scratch inputs
```

`configs/aligned_assets.json` points registered DXGL assets at `data/dxgl/...`.
Generated metrics, render outputs, trained workspaces, and snapshots still belong under
`outputs/`.

The dataset contents are intentionally ignored by git.
