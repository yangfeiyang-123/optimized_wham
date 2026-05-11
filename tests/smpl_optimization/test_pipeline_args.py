import scripts.video_to_fixed_smpl_to_opensim as pipeline


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
