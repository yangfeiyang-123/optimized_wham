from pathlib import Path

from lib.smpl_optimization.opensim_motion import parse_mot, summarize_mot_coordinates


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
