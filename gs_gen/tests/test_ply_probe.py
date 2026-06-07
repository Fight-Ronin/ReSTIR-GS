from __future__ import annotations

from pathlib import Path

from gs_gen.ply_probe import validate_3dgs_ply
from gs_gen.tests.fixtures import graphdeco_properties, write_ply


def test_validate_3dgs_ply_accepts_graphdeco_shape(tmp_path: Path) -> None:
    path = tmp_path / "splat.ply"
    write_ply(path, graphdeco_properties())

    result = validate_3dgs_ply(path)

    assert result["valid"] is True
    assert result["vertex_count"] == 1
    assert result["color_fields_present"] is True


def test_validate_3dgs_ply_rejects_plain_point_cloud(tmp_path: Path) -> None:
    path = tmp_path / "points.ply"
    write_ply(path, [("float", "x"), ("float", "y"), ("float", "z")])

    result = validate_3dgs_ply(path)

    assert result["valid"] is False
    assert any("missing required 3DGS fields" in error for error in result["errors"])


def test_validate_3dgs_ply_reports_malformed_vertex_count(tmp_path: Path) -> None:
    path = tmp_path / "bad_count.ply"
    path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex nope",
                "property float x",
                "end_header",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_3dgs_ply(path)

    assert result["valid"] is False
    assert "invalid vertex count: nope" in result["errors"]


def test_validate_3dgs_ply_requires_format_line(tmp_path: Path) -> None:
    path = tmp_path / "missing_format.ply"
    lines = ["ply", "element vertex 1"]
    lines.extend(f"property {kind} {name}" for kind, name in graphdeco_properties())
    lines.append("end_header")
    lines.append(" ".join("0" for _ in graphdeco_properties()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_3dgs_ply(path)

    assert result["valid"] is False
    assert "missing PLY format line" in result["errors"]
