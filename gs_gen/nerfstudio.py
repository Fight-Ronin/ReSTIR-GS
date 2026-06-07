from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gs_gen.paths import GsGenPaths
from gs_gen.source_probe import SourceInput


@dataclass(frozen=True)
class NerfstudioPlan:
    asset_id: str
    source: SourceInput
    paths: GsGenPaths
    commands: list[str]


def build_nerfstudio_plan(asset_id: str, source: SourceInput, workspace: Path) -> NerfstudioPlan:
    if source.kind not in ("images", "video"):
        raise ValueError(f"Unsupported source kind {source.kind!r}; expected 'images' or 'video'")
    paths = GsGenPaths(asset_id=asset_id, workspace=workspace)
    commands = [
        format_command(["ns-process-data", source.kind, "--data", source.path, "--output-dir", paths.processed_dir]),
        format_command(["ns-train", "splatfacto", "--data", paths.processed_dir, "--output-dir", paths.train_dir]),
        format_command(["ns-export", "gaussian-splat", "--load-config", "<trained-config.yml>", "--output-dir", paths.export_dir]),
        format_command(["python", "-m", "gs_gen", "validate", "--dataset-root", paths.processed_dir, "--splat", paths.default_splat_path]),
        format_command(["python", "-m", "gs_gen", "stage", "--asset-id", asset_id, "--dataset-root", paths.processed_dir, "--splat", paths.default_splat_path]),
    ]
    return NerfstudioPlan(asset_id=asset_id, source=source, paths=paths, commands=commands)


def format_command(parts: list[object]) -> str:
    return " ".join(_quote_arg(str(part)) for part in parts)


def _quote_arg(value: str) -> str:
    if value.startswith("<") and value.endswith(">"):
        return value
    needs_quotes = any(char.isspace() for char in value) or any(char in value for char in ['"', "&", "(", ")"])
    if not needs_quotes:
        return value
    return '"' + value.replace('"', '\\"') + '"'
