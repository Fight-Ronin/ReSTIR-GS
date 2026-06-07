from __future__ import annotations

import json
from pathlib import Path

import torch

from restir_gs.render.synthetic_scene import PinholeCamera


def load_camera_config(path: str | Path, device: torch.device | str = "cuda") -> PinholeCamera:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return camera_from_config_payload(data, device=device)


def camera_from_config_payload(payload: dict[str, object], device: torch.device | str = "cuda") -> PinholeCamera:
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"Unsupported camera config version: {payload.get('version')}")
    camera_data = payload["camera"]
    if not isinstance(camera_data, dict):
        raise ValueError("Camera config missing camera object.")
    viewmat = torch.tensor(camera_data["viewmat"], dtype=torch.float32, device=device)[None]
    intrinsics = torch.tensor(camera_data["intrinsics"], dtype=torch.float32, device=device)[None]
    return PinholeCamera(
        viewmats=viewmat,
        intrinsics=intrinsics,
        width=int(camera_data["width"]),
        height=int(camera_data["height"]),
    )
