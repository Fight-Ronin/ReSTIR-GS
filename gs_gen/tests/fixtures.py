from __future__ import annotations

import json
from pathlib import Path


def write_dataset(tmp_path: Path, write_image: bool = True) -> Path:
    root = tmp_path / "processed"
    (root / "images").mkdir(parents=True)
    if write_image:
        (root / "images" / "frame_000.png").write_bytes(b"placeholder")
    transforms = {
        "fl_x": 10.0,
        "w": 20,
        "h": 22,
        "frames": [
            {
                "file_path": "images/frame_000.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (root / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")
    return root


def graphdeco_properties() -> list[tuple[str, str]]:
    return [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("float", "opacity"),
        ("float", "scale_0"),
        ("float", "scale_1"),
        ("float", "scale_2"),
        ("float", "rot_0"),
        ("float", "rot_1"),
        ("float", "rot_2"),
        ("float", "rot_3"),
        ("float", "f_dc_0"),
        ("float", "f_dc_1"),
        ("float", "f_dc_2"),
    ]


def write_ply(path: Path, properties: list[tuple[str, str]]) -> None:
    lines = ["ply", "format ascii 1.0", "element vertex 1"]
    lines.extend(f"property {kind} {name}" for kind, name in properties)
    lines.append("end_header")
    lines.append(" ".join("0" for _ in properties))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
