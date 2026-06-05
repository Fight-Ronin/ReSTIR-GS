# Phase 21: DXGL Camera Normalization Fix

Phase 20 proved the DXGL Apple pretrained splat can load and render, but the render was much smaller than the aligned reference image. The measured scene extents explain why:

```text
raw points3D.ply bbox diagonal      ~= 3.795
pretrained apple.ply bbox diagonal ~= 0.947
scale                              ~= 0.25
```

The `transforms.json` cameras are in raw dataset coordinates, while the pretrained splat is in a normalized splatfacto/nerfstudio scene space.

## Fix

Phase 21 infers a uniform similarity from the raw point cloud to the pretrained splat. The splat PLY header says `Vertical Axis: z`, while the raw DXGL Apple point cloud is visibly Y-up. The default fix therefore uses a right-handed +90 degree X rotation:

```text
R = raw_y_to_z_up
splat_world = scale * R * (raw_world - raw_center) + splat_center
```

Camera translation and rotation are transformed:

```text
R_camera_splat = R * R_camera_raw
t_splat = scale * R * (t_raw - raw_center) + splat_center
```

Intrinsics are unchanged. The existing OpenGL camera-to-world to project world-to-camera conversion still runs after this normalization.

## Running

The normalized path is now the default:

```powershell
scripts\run_dxgl_splat_fidelity_windows.bat
```

To reproduce the Phase 20 raw-camera behavior:

```powershell
python scripts/demo_18_dxgl_splat_fidelity.py --camera-normalization none
```

To reproduce the scale/center-only Phase 21 diagnostic before the rotation fix:

```powershell
python scripts/demo_18_dxgl_splat_fidelity.py --normalization-rotation identity
```

The summary records:

```text
camera_normalization.mode
camera_normalization.normalization.raw_center
camera_normalization.normalization.target_center
camera_normalization.normalization.scale
camera_normalization.normalization.raw_to_target_rotation
camera_normalization.normalization.raw_bbox_diagonal
camera_normalization.normalization.target_bbox_diagonal
```

## Interpretation

If this fix works, the render alpha area should become comparable to the reference mask area, and the contact sheet should show the apple at approximately the same size and location as the DXGL RGB reference.

If size improves but orientation is wrong, the next bug is likely rotation/convention. If size and orientation improve but color remains different, then fidelity debugging can move to appearance, SH degree, exposure, or background treatment.
