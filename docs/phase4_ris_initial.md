# Phase 4: Initial RIS Reservoir Sampling

This phase adds the first ReSTIR-family estimator over the Phase 3 deterministic point lights. It compares sampled estimators against the all-lights deferred lighting reference.

## Quantity Estimated

The estimator targets diffuse RGB contribution:

```text
f(light) = pseudo_albedo * irradiance(light) / pi
```

Ambient is not sampled. It is added after the estimate:

```text
composite = rgb * ambient + estimated_diffuse
```

Invalid pixels preserve the original RGB in the composite.

## Uniform Proposals

Candidates are sampled uniformly with replacement:

```text
q(light) = 1 / num_lights
```

The uniform `K`-sample estimator is:

```text
estimated_diffuse = num_lights / K * sum_j f(sample_j)
```

Candidate indices are generated with a CPU `torch.Generator` and then moved to the requested device. This keeps CPU tests and CUDA demos reproducible.

## Initial RIS Reservoir

For each sampled candidate:

```text
p_hat_i = luminance(f_i)
w_i = p_hat_i / q_i = p_hat_i * num_lights
```

The reservoir selects one candidate with probability proportional to `w_i`. Its normalization is:

```text
W = sum_i(w_i) / (K * p_hat_selected)
estimated_diffuse = f_selected * W
```

Luminance uses Rec.709:

```text
luminance(rgb) = 0.2126*r + 0.7152*g + 0.0722*b
```

Zero-target pixels produce zero estimated diffuse and preserve original RGB in the composite.

## Outputs

The demo writes:

```text
outputs/ris_reference_composite.png
outputs/ris_uniform1_composite.png
outputs/ris_uniformK_composite.png
outputs/ris_initial_composite.png
outputs/ris_uniformK_abs_error.png
outputs/ris_initial_abs_error.png
```

It also prints mean absolute error against the all-lights diffuse reference for uniform 1-sample, uniform K-sample, and RIS K-sample.

## Known Limits

- This is initial RIS only, not temporal or spatial ReSTIR.
- There is still no visibility term or shadowing.
- The proposal distribution is uniform over lights, not light-power or position aware.
- The target is based on pseudo G-buffer lighting, not true relightable 3DGS materials.
