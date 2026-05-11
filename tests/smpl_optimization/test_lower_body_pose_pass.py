import numpy as np
import torch

from lib.smpl_optimization.lower_body import (
    LOWER_BODY_SMPL_JOINTS,
    LowerBodyOptimizerConfig,
    lower_body_pose_mask,
    optimize_record,
)
from lib.smpl_optimization.smpl_forward import expand_betas, smpl_forward_axis_angle


def test_lower_body_joint_indices_are_expected_smpl_joints():
    assert LOWER_BODY_SMPL_JOINTS == (1, 2, 4, 5, 7, 8, 10, 11)


def test_lower_body_pose_mask_contains_only_lower_body_dims():
    mask = lower_body_pose_mask(72)
    assert mask.dtype == bool
    assert mask.sum() == len(LOWER_BODY_SMPL_JOINTS) * 3
    assert bool(mask[0]) is False
    assert bool(mask[3]) is True


def test_expand_betas_repeats_single_beta():
    beta = np.arange(10, dtype=np.float32)

    expanded = expand_betas(beta, 3)

    assert expanded.shape == (3, 10)
    assert expanded.dtype == np.float32
    assert np.max(np.abs(expanded[2] - beta)) == 0.0


def test_smpl_forward_axis_angle_uses_body_model_output():
    class FakeOutput:
        def __init__(self):
            self.vertices = torch.zeros((2, 4, 3), dtype=torch.float32)
            self.feet = torch.ones((2, 2, 3), dtype=torch.float32)

    class FakeModel:
        def get_output(self, **kwargs):
            assert kwargs["body_pose"].shape == (2, 69)
            assert kwargs["global_orient"].shape == (2, 3)
            assert kwargs["betas"].shape == (2, 10)
            assert kwargs["transl"].shape == (2, 3)
            return FakeOutput()

    out = smpl_forward_axis_angle(
        FakeModel(),
        pose=torch.zeros((2, 72), dtype=torch.float32),
        betas=torch.zeros((2, 10), dtype=torch.float32),
        transl=torch.zeros((2, 3), dtype=torch.float32),
    )

    assert out["vertices"].shape == (2, 4, 3)
    assert out["feet"].shape == (2, 2, 3)


def test_pose_pass_changes_only_lower_body_when_enabled(monkeypatch):
    record = {
        "betas": np.zeros((3, 10), dtype=np.float32),
        "pose": np.zeros((3, 72), dtype=np.float32),
        "trans_world": np.zeros((3, 3), dtype=np.float32),
        "feet_refined": np.array(
            [[[0.0, -0.05, 0.0]], [[0.0, -0.05, 0.0]], [[0.0, 0.01, 0.0]]],
            dtype=np.float32,
        ),
        "contact": np.ones((3, 1), dtype=np.float32),
    }

    def fake_optimize_pose(record, config, frame_weights=None):
        out = {key: value.copy() if hasattr(value, "copy") else value for key, value in record.items()}
        mask = lower_body_pose_mask(out["pose"].shape[1])
        out["pose"][:, mask] += 0.01
        return out, {"pose_optimizer_used": True, "iterations": 1, "final_loss": 0.0}

    monkeypatch.setattr("lib.smpl_optimization.lower_body.optimize_lower_body_pose_smpl", fake_optimize_pose)

    out, reports = optimize_record(
        record,
        LowerBodyOptimizerConfig(fps=30.0, enable_pose_pass=True, pose_iterations=1, device="cpu"),
    )

    lower_dims = []
    for joint in LOWER_BODY_SMPL_JOINTS:
        lower_dims.extend([joint * 3, joint * 3 + 1, joint * 3 + 2])
    upper_dims = [idx for idx in range(72) if idx not in lower_dims]

    assert np.max(np.abs(out["pose"][:, upper_dims] - record["pose"][:, upper_dims])) == 0.0
    assert np.max(np.abs(out["pose"][:, lower_dims] - record["pose"][:, lower_dims])) == 0.01
    assert reports["lower_body_optimization_report"]["pose_optimizer_used"] is True
    assert reports["pose_delta_report"]["upper_body_max_abs"] == 0.0
    assert np.isclose(reports["pose_delta_report"]["lower_body_max_abs"], 0.01)
