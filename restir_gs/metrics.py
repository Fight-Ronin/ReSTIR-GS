from __future__ import annotations

import torch


def compute_rgb_error_metrics(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, float]:
    """Compute RGB error metrics over valid pixels."""
    if estimate.shape != reference.shape:
        raise ValueError(f"Expected estimate/reference shapes to match, got {tuple(estimate.shape)} and {tuple(reference.shape)}")
    if estimate.ndim != 3 or estimate.shape[-1] != 3:
        raise ValueError(f"Expected RGB tensors with shape [H,W,3], got {tuple(estimate.shape)}")
    if valid_mask.shape != estimate.shape[:2]:
        raise ValueError(f"Expected valid mask shape {tuple(estimate.shape[:2])}, got {tuple(valid_mask.shape)}")

    valid = valid_mask.to(device=estimate.device, dtype=torch.bool)
    if not bool(valid.any()):
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "bias_r": 0.0,
            "bias_g": 0.0,
            "bias_b": 0.0,
            "mean_abs_bias": 0.0,
        }

    error = estimate - reference
    selected = error[valid]
    abs_error = selected.abs()
    squared_error = selected * selected
    bias = selected.mean(dim=0)
    return {
        "mae": float(abs_error.mean().detach().cpu()),
        "rmse": float(torch.sqrt(squared_error.mean()).detach().cpu()),
        "bias_r": float(bias[0].detach().cpu()),
        "bias_g": float(bias[1].detach().cpu()),
        "bias_b": float(bias[2].detach().cpu()),
        "mean_abs_bias": float(bias.abs().mean().detach().cpu()),
    }
