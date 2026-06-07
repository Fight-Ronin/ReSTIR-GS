from __future__ import annotations

import json
from pathlib import Path

import pytest

from gs_gen.config import config_from_dict, load_config


def test_config_loads_images_source(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "asset_id": "my_room",
                "workspace": "outputs/custom",
                "source": {"kind": "images", "path": "data/images"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.asset_id == "my_room"
    assert config.source.kind == "images"
    assert config.workspace == Path("outputs/custom")


def test_config_rejects_invalid_source_kind() -> None:
    with pytest.raises(ValueError, match="source.kind"):
        config_from_dict({"asset_id": "room", "source": {"kind": "audio", "path": "x"}})
