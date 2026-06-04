# Phase 2: Synthetic Pseudo G-buffer

This phase converts the Phase 1 `gsplat` RGB/expected-depth/alpha output into a small pseudo G-buffer for later deferred lighting and ReSTIR experiments.

## Expected-depth Input

`gsplat.rasterization(render_mode="RGB+ED")` returns RGB plus expected z-depth:

```text
ED = sum_i(w_i * z_i) / sum_i(w_i)
```

This is different from `RGB+D`, where depth is the accumulated weighted depth `sum_i(w_i * z_i)`. Expected depth is the better first buffer for unprojection because it behaves like one representative camera-space z value per pixel.

Pixels are considered valid when:

```text
alpha > 1e-4 and depth is finite and depth > 0
```

## Camera-space Position

The synthetic camera currently uses an identity view matrix and a pinhole intrinsic matrix. Identity view means camera space and world space coincide for the Phase 1/2 scene. The camera looks along `+Z`.

For each valid pixel `(u, v)` with expected z-depth `z`, unprojection is:

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth
```

Invalid pixels get zero position and are excluded from downstream normal estimation.

## Screen-space Normals

Normals are estimated from central differences of camera-space position:

```text
dx = position[y, x + 1] - position[y, x - 1]
dy = position[y + 1, x] - position[y - 1, x]
normal = normalize(cross(dx, dy))
```

With image `y` increasing downward and the camera looking along `+Z`, a fronto-parallel constant-depth plane produces `[0, 0, 1]`. Normals are flipped when needed so `normal.z >= 0`.

A pixel receives a valid normal only when the center, left, right, up, and down pixels are all valid and the cross product has nonzero length. Boundary pixels have invalid normals.

## Known Limits

- These are pseudo normals from a screen-space depth surface, not true Gaussian surface normals.
- Expected depth blends overlapping Gaussians, so position and normals are representative buffers rather than exact scene geometry.
- This phase intentionally avoids `.ply` loading, material parameters, direct lighting, and ReSTIR sampling.
