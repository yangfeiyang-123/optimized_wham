from pathlib import Path

import numpy as np

from lib.smpl_optimization.metrics import temporal_derivative_summary


DEFAULT_COORDINATE_RANGES = {
    "pelvis_tilt": (-90.0, 90.0),
    "pelvis_list": (-90.0, 90.0),
    "pelvis_rotation": (-180.0, 180.0),
    "hip_flexion_l": (-120.0, 120.0),
    "hip_flexion_r": (-120.0, 120.0),
    "hip_adduction_l": (-90.0, 90.0),
    "hip_adduction_r": (-90.0, 90.0),
    "hip_rotation_l": (-90.0, 90.0),
    "hip_rotation_r": (-90.0, 90.0),
    "knee_angle_l": (-120.0, 160.0),
    "knee_angle_r": (-120.0, 160.0),
    "ankle_angle_l": (-90.0, 90.0),
    "ankle_angle_r": (-90.0, 90.0),
    "subtalar_angle_l": (-90.0, 90.0),
    "subtalar_angle_r": (-90.0, 90.0),
    "mtp_angle_l": (-90.0, 90.0),
    "mtp_angle_r": (-90.0, 90.0),
    "lumbar_extension": (-90.0, 90.0),
    "lumbar_bending": (-90.0, 90.0),
    "lumbar_rotation": (-90.0, 90.0),
}


def parse_mot(path) -> dict:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    table_start = 0

    for index, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            table_start = index + 1
            break

    columns = []
    rows = []
    for line in lines[table_start:]:
        stripped = line.strip()
        if not stripped:
            continue

        if not columns:
            columns = stripped.split()
            continue

        values = stripped.split()
        row = []
        for value in values[: len(columns)]:
            try:
                row.append(float(value))
            except ValueError:
                row.append(np.nan)
        if len(row) < len(columns):
            row.extend([np.nan] * (len(columns) - len(row)))
        rows.append(row)

    if not columns:
        return {"columns": [], "data": np.empty((0, 0), dtype=np.float64)}

    data = np.asarray(rows, dtype=np.float64)
    if data.size == 0:
        data = np.empty((0, len(columns)), dtype=np.float64)

    return {"columns": columns, "data": data}


def _coordinate_columns(columns):
    return [index for index, name in enumerate(columns) if name.lower() != "time"]


def _empty_summary(finite=True):
    zero = {"rms": 0.0, "max_abs": 0.0}
    return {
        "finite": bool(finite),
        "num_coordinates": 0,
        "num_range_violations": 0,
        "num_coordinate_jumps": 0,
        "coordinate_velocity": zero.copy(),
        "coordinate_acceleration": zero.copy(),
        "coordinate_jerk": zero.copy(),
    }


def summarize_mot_coordinates(
    path,
    coordinate_ranges=None,
    jump_threshold=45.0,
) -> dict:
    parsed = parse_mot(path)
    columns = parsed["columns"]
    data = parsed["data"]
    coordinate_indexes = _coordinate_columns(columns)

    if data.size == 0 or not coordinate_indexes:
        return _empty_summary(finite=np.all(np.isfinite(data)))

    coordinates = data[:, coordinate_indexes]
    finite = bool(np.all(np.isfinite(coordinates)))
    ranges = DEFAULT_COORDINATE_RANGES if coordinate_ranges is None else coordinate_ranges

    num_range_violations = 0
    for output_index, column_index in enumerate(coordinate_indexes):
        limits = ranges.get(columns[column_index])
        if limits is None:
            continue
        low, high = limits
        values = coordinates[:, output_index]
        num_range_violations += int(
            np.count_nonzero((values < float(low)) | (values > float(high)))
        )

    if coordinates.shape[0] <= 1:
        num_coordinate_jumps = 0
    else:
        jumps = np.abs(np.diff(coordinates, axis=0)) > float(jump_threshold)
        num_coordinate_jumps = int(np.count_nonzero(jumps))

    return {
        "finite": finite,
        "num_coordinates": len(coordinate_indexes),
        "num_range_violations": num_range_violations,
        "num_coordinate_jumps": num_coordinate_jumps,
        "coordinate_velocity": temporal_derivative_summary(coordinates, order=1),
        "coordinate_acceleration": temporal_derivative_summary(coordinates, order=2),
        "coordinate_jerk": temporal_derivative_summary(coordinates, order=3),
    }
