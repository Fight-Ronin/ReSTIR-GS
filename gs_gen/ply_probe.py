from __future__ import annotations

from pathlib import Path


REQUIRED_3DGS_FIELDS = {
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
}


def validate_3dgs_ply(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"valid": False, "path": str(path), "errors": ["missing splat PLY"], "vertex_count": 0}

    header = read_ply_header(path)
    properties = set(header["properties"])
    errors: list[str] = []
    if not header["saw_end_header"]:
        errors.append("missing PLY end_header")
    if not header["format"]:
        errors.append("missing PLY format line")
    if header["header_errors"]:
        errors.extend(header["header_errors"])
    if int(header["vertex_count"]) <= 0:
        errors.append("vertex count must be positive")
    missing = sorted(REQUIRED_3DGS_FIELDS - properties)
    if missing:
        errors.append(f"missing required 3DGS fields: {missing}")
    has_color = (
        {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(properties)
        or {"red", "green", "blue"}.issubset(properties)
        or {"r", "g", "b"}.issubset(properties)
    )
    if not has_color:
        errors.append("missing color fields f_dc_0..2, red/green/blue, or r/g/b")
    return {
        "valid": not errors,
        "path": str(path),
        "format": header["format"],
        "vertex_count": header["vertex_count"],
        "property_count": len(header["properties"]),
        "color_fields_present": has_color,
        "errors": errors,
    }


def read_ply_header(path: Path) -> dict[str, object]:
    properties: list[str] = []
    vertex_count = 0
    ply_format = ""
    header_errors: list[str] = []
    in_vertex = False
    saw_end_header = False
    with path.open("rb") as handle:
        first = handle.readline().decode("utf-8", errors="replace").strip()
        if first != "ply":
            return {
                "format": "",
                "vertex_count": 0,
                "properties": [],
                "saw_end_header": False,
                "header_errors": ["missing PLY magic header"],
            }
        for raw_line in handle:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith("format "):
                ply_format = line
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    try:
                        vertex_count = int(parts[2])
                    except ValueError:
                        header_errors.append(f"invalid vertex count: {parts[2]}")
            elif in_vertex and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    properties.append(parts[-1])
            elif line == "end_header":
                saw_end_header = True
                break
    return {
        "format": ply_format,
        "vertex_count": vertex_count,
        "properties": properties,
        "saw_end_header": saw_end_header,
        "header_errors": header_errors,
    }
