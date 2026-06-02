"""Validate a SMPL-to-OpenSim retarget YAML against an OpenSim model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import yaml


def _tag_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _named_elements(root: ET.Element, element_name: str) -> set[str]:
    names: set[str] = set()
    for elem in root.iter():
        if _tag_name(elem.tag) == element_name:
            name = elem.attrib.get("name")
            if name:
                names.add(name)
    return names


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _flatten_coordinate_groups(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_coordinate_groups(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for item in value.values():
            out.extend(_flatten_coordinate_groups(item))
        return out
    return []


def validate(config_path: Path) -> int:
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    references = config.get("references", {})
    model_path = _resolve(config_path.parent, references["opensim_model"])
    tree = ET.parse(model_path)
    root = tree.getroot()

    markers = _named_elements(root, "Marker")
    bodies = _named_elements(root, "Body")
    coordinates = _named_elements(root, "Coordinate")

    errors: list[str] = []
    retarget_points = config.get("retarget_points", [])
    expected_count = config.get("quality_checks", {}).get("expected_retarget_point_count")
    if expected_count is not None and len(retarget_points) != expected_count:
        errors.append(f"retarget_points count is {len(retarget_points)}, expected {expected_count}")

    for point in retarget_points:
        key = point.get("key", "<missing-key>")
        target_type = point.get("target_type")
        if target_type == "opensim_marker":
            marker = point.get("opensim_marker")
            if marker not in markers:
                errors.append(f"{key}: missing opensim_marker {marker!r}")
        elif target_type == "average_opensim_markers":
            for marker in point.get("opensim_markers", []):
                if marker not in markers:
                    errors.append(f"{key}: missing average marker {marker!r}")
        elif target_type == "opensim_body_origin":
            body = point.get("opensim_body")
            if body not in bodies:
                errors.append(f"{key}: missing opensim_body {body!r}")
        else:
            errors.append(f"{key}: unsupported target_type {target_type!r}")

        for marker in point.get("auxiliary_opensim_markers", []):
            if marker not in markers:
                errors.append(f"{key}: missing auxiliary marker {marker!r}")

    for coord in _flatten_coordinate_groups(config.get("optimizable_coordinates", {})):
        if coord not in coordinates:
            errors.append(f"missing optimizable coordinate {coord!r}")

    for coord in _flatten_coordinate_groups(config.get("fixed_coordinates", {})):
        if coord not in coordinates:
            errors.append(f"missing fixed coordinate {coord!r}")

    print(f"config: {config_path}")
    print(f"opensim_model: {model_path}")
    print(f"retarget_points: {len(retarget_points)}")
    print(f"opensim markers found: {len(markers)}")
    print(f"opensim bodies found: {len(bodies)}")
    print(f"opensim coordinates found: {len(coordinates)}")

    if errors:
        print("validation: failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print("validation: passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/retarget/smpl_to_mimicmsk_opensim.yaml",
        help="Path to the SMPL-to-OpenSim retarget YAML.",
    )
    args = parser.parse_args()
    return validate(Path(args.config))


if __name__ == "__main__":
    sys.exit(main())
