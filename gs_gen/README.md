# gs_gen

Standalone helper for producing local room Gaussian Splatting assets before they are connected to ReSTIR-GS.

This directory intentionally does not modify `configs/aligned_assets.json` or the active renderer path. It only plans, probes, validates, and stages generated assets.

## Input Modes

Use one source:

```powershell
python -m gs_gen plan --asset-id my_room --images data\room_capture\my_room\images
python -m gs_gen plan --asset-id my_room --video data\room_capture\my_room\walkthrough.mp4
```

Optional source check:

```powershell
python -m gs_gen probe-source --images data\room_capture\my_room\images
python -m gs_gen probe-source --video data\room_capture\my_room\walkthrough.mp4
```

## Planned External Commands

The generated commands follow this path:

```text
ns-process-data images|video
ns-train splatfacto
ns-export gaussian-splat
python -m gs_gen validate
python -m gs_gen stage
```

`gs_gen` does not install, clone, or run external training tools in this MVP.

## Output Layout

```text
outputs/gsgen/<asset_id>/
  processed/
  train/
  export/
  staged/
    transforms.json
    splat.ply
    asset_info.json
    images/              optional, with --copy-images
```

## Validation

```powershell
python -m gs_gen validate --dataset-root outputs\gsgen\my_room\processed --splat outputs\gsgen\my_room\export\splat.ply
```

Validation checks:

- `transforms.json` exists and has frames.
- Global image size and focal metadata exist.
- Frame image paths resolve.
- Exported PLY has compatible 3DGS fields.

## Staging

```powershell
python -m gs_gen stage --asset-id my_room --dataset-root outputs\gsgen\my_room\processed --splat outputs\gsgen\my_room\export\splat.ply --copy-images
```

Staging writes a stable asset folder, but does not register it with ReSTIR-GS yet.
