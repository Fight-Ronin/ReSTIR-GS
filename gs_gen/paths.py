from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_WORKSPACE = Path("outputs/gsgen")


@dataclass(frozen=True)
class GsGenPaths:
    asset_id: str
    workspace: Path = DEFAULT_WORKSPACE

    @property
    def asset_root(self) -> Path:
        return self.workspace / self.asset_id

    @property
    def processed_dir(self) -> Path:
        return self.asset_root / "processed"

    @property
    def train_dir(self) -> Path:
        return self.asset_root / "train"

    @property
    def export_dir(self) -> Path:
        return self.asset_root / "export"

    @property
    def staged_dir(self) -> Path:
        return self.asset_root / "staged"

    @property
    def default_splat_path(self) -> Path:
        return self.export_dir / "splat.ply"


def validate_asset_id(asset_id: str) -> str:
    value = asset_id.strip()
    if not value:
        raise ValueError("asset_id must be non-empty")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(char not in allowed for char in value):
        raise ValueError(f"asset_id may only contain letters, numbers, '_' and '-', got {asset_id!r}")
    return value
