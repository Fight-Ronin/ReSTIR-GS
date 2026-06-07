from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from gs_gen.paths import DEFAULT_WORKSPACE, validate_asset_id
from gs_gen.source_probe import SourceInput


@dataclass(frozen=True)
class GsGenConfig:
    asset_id: str
    source: SourceInput
    workspace: Path = DEFAULT_WORKSPACE


def load_config(path: Path) -> GsGenConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected config object in {path}")
    return config_from_dict(data)


def config_from_dict(data: dict[str, Any]) -> GsGenConfig:
    asset_id = validate_asset_id(str(data.get("asset_id", "")))
    source_data = data.get("source")
    if not isinstance(source_data, dict):
        raise ValueError("config source must be an object")
    kind = str(source_data.get("kind", ""))
    source_path = source_data.get("path")
    if kind not in ("images", "video"):
        raise ValueError(f"source.kind must be 'images' or 'video', got {kind!r}")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("source.path must be a non-empty string")
    workspace = Path(str(data.get("workspace", DEFAULT_WORKSPACE)))
    return GsGenConfig(asset_id=asset_id, source=SourceInput(kind=kind, path=Path(source_path)), workspace=workspace)
