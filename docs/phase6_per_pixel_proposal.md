# Phase 6: Per-pixel Geometric Proposal

This phase adds a non-uniform light proposal before temporal or spatial ReSTIR. It keeps the scene synthetic and compares proposal quality against the Phase 3 all-lights reference.

## Proposal Distribution

For each valid G-buffer pixel `x` and light `l`, the geometric proposal weight is:

```text
w_l(x) = intensity_l * luminance(color_l) * abs(dot(N_x, wi_l)) / dist_l^2
q_l(x) = w_l(x) / sum_j(w_j(x))
```

`wi_l` points from the camera-space surface position to the light. The current phase uses the same two-sided normal policy as Phase 3/4 because normals are screen-space pseudo normals.

This proposal is not an oracle. It does not use pseudo albedo and does not directly normalize the final diffuse contribution. It is a practical proxy based on light power, distance, and the local normal.

## Invalid And Zero-weight Pixels

Invalid pixels, pixels without valid normals, and pixels whose geometric weights sum to zero use a uniform proposal fallback:

```text
q_l(x) = 1 / num_lights
```

This keeps sampled proposal probabilities finite. Estimators still use the G-buffer lighting mask, so invalid pixels receive zero estimated diffuse lighting and preserve the original RGB in the composite.

## Estimators

The proposal Monte Carlo estimator uses:

```text
estimated_diffuse = mean_i(f_i / q_i)
```

where `f_i` is the selected light's diffuse RGB contribution.

The RIS estimator uses:

```text
p_hat_i = luminance(f_i)
w_i = p_hat_i / q_i
W = sum_i(w_i) / (K * p_hat_selected)
estimated_diffuse = f_selected * W
```

With `K=1` and positive target, RIS matches the proposal Monte Carlo estimator.

## Outputs

The Phase 6 demo writes:

```text
outputs/proposal_ablation.csv
outputs/proposal_ablation_summary.json
```

Rows include:

```text
proposal, estimator, k, seed_index, candidate_seed, selection_seed,
reference_quantity, mae, rmse, bias_r, bias_g, bias_b, mean_abs_bias
```

The default sweep compares:

```text
proposal = uniform, geometric
estimator = mc, ris
K = [1, 2, 4, 8, 16, 32]
seed_count = 8
```

## Limitations

This phase only studies initial sampling quality. It does not add temporal reuse, spatial reuse, visibility, shadow rays, real `.ply` assets, or RTXDI integration.

The full `[H,W,N]` proposal tensor is intentionally stored for this synthetic `128x128x128` milestone. Alias tables, tiling, or streaming proposal evaluation are later scalability work.
