from __future__ import annotations

import torch


def masked_rgb_metrics(estimate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    if estimate.shape != reference.shape:
        raise ValueError(f"Estimate/reference shape mismatch: {tuple(estimate.shape)} vs {tuple(reference.shape)}")
    if mask.shape != estimate.shape[:2]:
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match RGB shape {tuple(estimate.shape)}")
    valid_count = int(mask.sum().item())
    if valid_count <= 0:
        return {"valid_pixels": 0, "mae": 0.0, "rmse": 0.0, "psnr": 0.0}
    diff = estimate.detach().cpu()[mask] - reference.detach().cpu()[mask]
    mse = torch.mean(diff.square())
    mae = torch.mean(diff.abs())
    rmse = torch.sqrt(mse)
    psnr = 20.0 * torch.log10(torch.tensor(1.0) / torch.clamp(rmse, min=1e-8))
    return {"valid_pixels": valid_count, "mae": float(mae), "rmse": float(rmse), "psnr": float(psnr)}


def binary_mask_metrics(estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, float | int]:
    if estimate.shape != reference.shape:
        raise ValueError(f"Mask shape mismatch: {tuple(estimate.shape)} vs {tuple(reference.shape)}")
    estimate = estimate.detach().cpu().to(torch.bool)
    reference = reference.detach().cpu().to(torch.bool)
    intersection = int((estimate & reference).sum().item())
    union = int((estimate | reference).sum().item())
    estimate_count = int(estimate.sum().item())
    reference_count = int(reference.sum().item())
    iou = intersection / float(union) if union > 0 else 0.0
    precision = intersection / float(estimate_count) if estimate_count > 0 else 0.0
    recall = intersection / float(reference_count) if reference_count > 0 else 0.0
    return {
        "estimate_pixels": estimate_count,
        "reference_pixels": reference_count,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def depth_metrics(estimate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    if estimate.shape != reference.shape or mask.shape != estimate.shape:
        raise ValueError("Expected estimate, reference, and mask to have matching [H,W] shapes.")
    valid = mask.detach().cpu().to(torch.bool) & torch.isfinite(estimate.detach().cpu()) & torch.isfinite(reference.detach().cpu())
    valid = valid & (estimate.detach().cpu() > 0.0) & (reference.detach().cpu() > 0.0)
    valid_count = int(valid.sum().item())
    if valid_count <= 0:
        return {"valid_pixels": 0, "mae": 0.0, "rmse": 0.0, "abs_rel": 0.0}
    diff = estimate.detach().cpu()[valid] - reference.detach().cpu()[valid]
    abs_diff = torch.abs(diff)
    rmse = torch.sqrt(torch.mean(diff.square()))
    abs_rel = torch.mean(abs_diff / torch.clamp(reference.detach().cpu()[valid], min=1e-8))
    return {"valid_pixels": valid_count, "mae": float(abs_diff.mean()), "rmse": float(rmse), "abs_rel": float(abs_rel)}


def normal_display_metrics(estimate_display_rgb: torch.Tensor, reference_rgb: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    # Diagnostic only: this compares displayed normal colors, not guaranteed equal semantic spaces.
    return masked_rgb_metrics(estimate_display_rgb, reference_rgb, mask)
