from pathlib import Path

import pytest

import scripts.video_to_fixed_smpl_to_opensim as pipeline


def _patch_fast_pipeline(monkeypatch, commands):
    def fake_run(cmd, cwd=pipeline.REPO_ROOT):
        commands.append(cmd)
        script = Path(cmd[1]).name
        if script == "world_grounded_smpl_optimizer.py":
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "optimized_canonical_wham_output.pkl").write_bytes(b"")
        if script == "optimize_smpl_lower_body.py":
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "corrected_smpl.pkl").write_bytes(b"")

    monkeypatch.setattr(pipeline, "require_file", lambda path, label: path)
    monkeypatch.setattr(
        pipeline,
        "load_fixed_beta_report",
        lambda report_path: {"pipeline_beta_variation_after_max_abs": 0.0},
    )
    monkeypatch.setattr(pipeline, "run", fake_run)


def test_pipeline_accepts_optimize_lower_body_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "video_to_fixed_smpl_to_opensim.py",
            "--video",
            "input.mp4",
            "--optimize-lower-body",
        ],
    )
    args = pipeline.parse_args()
    assert args.optimize_lower_body is True


def test_pipeline_lower_body_args_have_expected_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "video_to_fixed_smpl_to_opensim.py",
            "--video",
            "input.mp4",
        ],
    )
    args = pipeline.parse_args()
    assert args.optimize_lower_body is False
    assert args.lower_body_out_dir is None
    assert args.lower_body_max_root_y_shift == 0.25
    assert args.disable_lower_body_pose_pass is False
    assert args.lower_body_pose_iterations == 80


def test_pipeline_accepts_lower_body_disable_and_tuning_args(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "video_to_fixed_smpl_to_opensim.py",
            "--video",
            "input.mp4",
            "--lower-body-out-dir",
            "lb-out",
            "--lower-body-max-root-y-shift",
            "0.1",
            "--disable-lower-body-pose-pass",
            "--lower-body-pose-iterations",
            "12",
        ],
    )
    args = pipeline.parse_args()
    assert args.lower_body_out_dir == "lb-out"
    assert args.lower_body_max_root_y_shift == 0.1
    assert args.disable_lower_body_pose_pass is True
    assert args.lower_body_pose_iterations == 12


def test_pipeline_rejects_lower_body_without_world_grounded(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "video_to_fixed_smpl_to_opensim.py",
            "--video",
            "input.mp4",
            "--optimize-lower-body",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        pipeline.main()

    assert exc_info.value.code == 2
    assert (
        "--optimize-lower-body requires --world-grounded so ground_y=0 is meaningful."
        in capsys.readouterr().err
    )


def test_pipeline_world_grounded_lower_body_routes_to_lower_body_retarget(
    monkeypatch, tmp_path
):
    commands = []
    _patch_fast_pipeline(monkeypatch, commands)
    video = tmp_path / "clip.mp4"
    output_root = tmp_path / "output"
    safe_sequence = pipeline.safe_ascii_name("clip")
    lower_body_out = output_root / "_lower_body_optimized" / safe_sequence
    retarget_out = output_root / "_opensim_retarget_lower_body" / safe_sequence
    monkeypatch.setattr(
        "sys.argv",
        [
            "video_to_fixed_smpl_to_opensim.py",
            "--video",
            str(video),
            "--output-pth",
            str(output_root),
            "--world-grounded",
            "--optimize-lower-body",
        ],
    )

    assert pipeline.main() == 0

    assert [Path(cmd[1]).name for cmd in commands] == [
        "demo.py",
        "canonicalize_wham_fixed_beta.py",
        "world_grounded_smpl_optimizer.py",
        "optimize_smpl_lower_body.py",
        "retarget_smpl_to_opensim.py",
    ]
    retarget_cmd = commands[-1]
    assert Path(retarget_cmd[2]) == lower_body_out / "corrected_smpl.pkl"
    assert Path(retarget_cmd[retarget_cmd.index("--out-dir") + 1]) == retarget_out


def test_pipeline_skip_ik_omits_run_ik_and_prints_skipped_motion(
    monkeypatch, tmp_path, capsys
):
    commands = []
    _patch_fast_pipeline(monkeypatch, commands)
    video = tmp_path / "clip.mp4"
    output_root = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "video_to_fixed_smpl_to_opensim.py",
            "--video",
            str(video),
            "--output-pth",
            str(output_root),
            "--world-grounded",
            "--skip-ik",
        ],
    )

    assert pipeline.main() == 0

    retarget_cmd = commands[-1]
    assert "--run-ik" not in retarget_cmd
    assert "opensim_motion: <not run; --skip-ik>" in capsys.readouterr().out
