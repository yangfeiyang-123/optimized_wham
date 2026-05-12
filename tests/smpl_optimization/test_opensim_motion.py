import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.smpl_optimization.opensim_motion import (  # noqa: E402
    parse_mot,
    summarize_mot_coordinates,
)


def test_parse_mot_reads_header_and_rows(tmp_path: Path):
    mot = tmp_path / "sample.mot"
    mot.write_text(
        "name sample\n"
        "endheader\n"
        "time\thip_flexion_r\tknee_angle_r\n"
        "0.0\t1.0\t2.0\n"
        "0.1\t1.5\t2.5\n",
        encoding="utf-8",
    )

    parsed = parse_mot(mot)

    assert parsed["columns"] == ["time", "hip_flexion_r", "knee_angle_r"]
    assert parsed["data"].shape == (2, 3)
    np.testing.assert_allclose(
        parsed["data"],
        np.array(
            [
                [0.0, 1.0, 2.0],
                [0.1, 1.5, 2.5],
            ]
        ),
    )


def test_summarize_mot_coordinates_counts_jumps(tmp_path: Path):
    mot = tmp_path / "jump.mot"
    mot.write_text(
        "endheader\n"
        "time\tknee_angle_r\n"
        "0.0\t0.0\n"
        "0.1\t0.1\n"
        "0.2\t5.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(mot, jump_threshold=1.0)

    assert summary["finite"] is True
    assert summary["num_coordinate_jumps"] == 1
    assert summary["coordinate_jerk"]["rms"] == 0.0


def test_summarize_mot_coordinates_counts_range_violations(tmp_path: Path):
    mot = tmp_path / "range.mot"
    mot.write_text(
        "endheader\n"
        "time\tknee_angle_r\n"
        "0.0\t0.0\n"
        "0.1\t200.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(
        mot,
        coordinate_ranges={"knee_angle_r": (-120.0, 160.0)},
    )

    assert summary["num_range_violations"] == 1


def test_summarize_mot_coordinates_checks_time_for_finiteness(tmp_path: Path):
    mot = tmp_path / "bad_time.mot"
    mot.write_text(
        "endheader\n"
        "time\tknee_angle_r\n"
        "nan\t0.0\n"
        "inf\t0.1\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(mot)

    assert summary["finite"] is False


def test_summarize_mot_coordinates_marks_long_ragged_rows_non_finite(tmp_path: Path):
    mot = tmp_path / "long_row.mot"
    mot.write_text(
        "endheader\n"
        "time\tknee_angle_r\n"
        "0.0\t0.0\t10.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(mot)

    assert summary["finite"] is False


def test_summarize_mot_coordinates_marks_short_ragged_rows_non_finite(tmp_path: Path):
    mot = tmp_path / "short_row.mot"
    mot.write_text(
        "endheader\n"
        "time\tknee_angle_r\n"
        "0.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(mot)

    assert summary["finite"] is False


def test_summarize_mot_coordinates_handles_coordinate_only_files(tmp_path: Path):
    mot = tmp_path / "coordinate_only.mot"
    mot.write_text(
        "endheader\n"
        "knee_angle_r\n"
        "0.0\n"
        "5.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(mot, jump_threshold=1.0)

    assert summary["finite"] is True
    assert summary["num_coordinates"] == 1
    assert summary["num_coordinate_jumps"] == 1


def test_summarize_mot_coordinates_ignores_unknown_default_ranges(tmp_path: Path):
    mot = tmp_path / "unknown_coordinate.mot"
    mot.write_text(
        "endheader\n"
        "time\tcustom_coordinate\n"
        "0.0\t100000.0\n",
        encoding="utf-8",
    )

    summary = summarize_mot_coordinates(mot)

    assert summary["num_range_violations"] == 0
