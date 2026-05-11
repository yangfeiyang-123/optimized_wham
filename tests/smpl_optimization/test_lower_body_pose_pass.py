import sys
import types

import numpy as np
import torch

from lib.smpl_optimization.lower_body import (
    LOWER_BODY_SMPL_JOINTS,
    LowerBodyOptimizerConfig,
    lower_body_pose_mask,
    optimize_lower_body_pose_smpl,
    optimize_record,
)
from lib.smpl_optimization.smpl_forward import expand_betas, smpl_forward_axis_angle


class DifferentiableFakeOutput:
    def __init__(self, feet):
        self.feet = feet
        self.vertices = torch.cat([feet, feet + torch.tensor([0.0, 0.2, 0.0], device=feet.device)], dim=1)


class DifferentiableFakeModel:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def get_output(self, **kwargs):
        transl = kwargs["transl"]
        body_pose = kwargs["body_pose"]
        assert transl.shape[0] == self.batch_size
        assert body_pose.shape[0] == self.batch_size
        lower_pose_y = body_pose[:, 0]
        foot_y = transl[:, 1] - 0.2 + 0.5 * lower_pose_y
        left = torch.stack([transl[:, 0] - 0.1, foot_y, transl[:, 2]], dim=-1)
        right = torch.stack([transl[:, 0] + 0.1, foot_y, transl[:, 2]], dim=-1)
        return DifferentiableFakeOutput(torch.stack([left, right], dim=1))


def make_pose_optimizer_record(n_frames=4):
    feet = np.full((n_frames, 2, 3), 0.1, dtype=np.float32)
    return {
        "betas": np.tile(np.arange(10, dtype=np.float32), (n_frames, 1)),
        "pose": np.zeros((n_frames, 72), dtype=np.float32),
        "trans_world": np.zeros((n_frames, 3), dtype=np.float32),
        "feet": feet.copy(),
        "feet_world": feet.copy() + 1.0,
        "feet_refined": feet.copy() + 2.0,
        "contact": np.ones((n_frames, 2), dtype=np.float32),
    }


def install_differentiable_fake_model(monkeypatch):
    batch_sizes = []

    def fake_build_body_model(device, batch_size):
        batch_sizes.append(batch_size)
        return DifferentiableFakeModel(batch_size)

    monkeypatch.setitem(sys.modules, "lib.models", types.SimpleNamespace(build_body_model=fake_build_body_model))
    return batch_sizes


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


def test_pose_pass_updates_pose_world_for_retargeted_pose(monkeypatch):
    record = {
        "betas": np.zeros((3, 10), dtype=np.float32),
        "pose": np.zeros((3, 72), dtype=np.float32),
        "pose_world": np.zeros((3, 72), dtype=np.float32),
        "trans_world": np.zeros((3, 3), dtype=np.float32),
        "feet_refined": np.zeros((3, 1, 3), dtype=np.float32),
        "contact": np.ones((3, 1), dtype=np.float32),
    }

    def fake_optimize_pose(record, config, frame_weights=None):
        out = {key: value.copy() if hasattr(value, "copy") else value for key, value in record.items()}
        mask = lower_body_pose_mask(out["pose"].shape[1])
        out["pose"][:, mask] += 0.25
        return out, {"pose_optimizer_used": True, "iterations": 1, "final_loss": 0.0}

    monkeypatch.setattr("lib.smpl_optimization.lower_body.optimize_lower_body_pose_smpl", fake_optimize_pose)

    out, _ = optimize_record(
        record,
        LowerBodyOptimizerConfig(fps=30.0, enable_pose_pass=True, pose_iterations=1, device="cpu"),
    )

    mask = lower_body_pose_mask(72)
    np.testing.assert_array_equal(out["pose_world"][:, mask], out["pose"][:, mask])
    np.testing.assert_array_equal(out["pose_world"][:, ~mask], record["pose_world"][:, ~mask])
    np.testing.assert_array_equal(out["pose"][:, ~mask], record["pose"][:, ~mask])
    np.testing.assert_array_equal(out["pose_world"][:, mask], np.full((3, int(mask.sum())), 0.25, dtype=np.float32))


def test_pose_pass_cannot_push_total_root_y_shift_past_budget(monkeypatch):
    record = {
        "betas": np.zeros((3, 10), dtype=np.float32),
        "pose": np.zeros((3, 72), dtype=np.float32),
        "trans_world": np.zeros((3, 3), dtype=np.float32),
        "feet_refined": np.array(
            [[[0.0, -0.10, 0.0]], [[0.0, -0.10, 0.0]], [[0.0, -0.10, 0.0]]],
            dtype=np.float32,
        ),
        "contact": np.ones((3, 1), dtype=np.float32),
    }
    max_root_y_shift = 0.05

    def fake_optimize_pose(record, config, frame_weights=None):
        out = {key: value.copy() if hasattr(value, "copy") else value for key, value in record.items()}
        out["trans_world"][:, 1] += max_root_y_shift
        return out, {"pose_optimizer_used": True, "root_residual_y_max_abs": max_root_y_shift}

    monkeypatch.setattr("lib.smpl_optimization.lower_body.optimize_lower_body_pose_smpl", fake_optimize_pose)

    out, reports = optimize_record(
        record,
        LowerBodyOptimizerConfig(
            fps=30.0,
            enable_pose_pass=True,
            max_root_y_shift=max_root_y_shift,
            device="cpu",
        ),
    )

    total_y_shift = out["trans_world"][:, 1] - record["trans_world"][:, 1]
    lower_report = reports["lower_body_optimization_report"]
    assert np.max(np.abs(total_y_shift)) <= max_root_y_shift + 1e-6
    assert lower_report["deterministic_root_y_shift_max_abs"] == max_root_y_shift
    assert lower_report["pose_root_residual_y_max_abs"] == max_root_y_shift
    assert lower_report["total_root_y_shift_max_abs"] <= max_root_y_shift + 1e-6
    assert lower_report["root_y_shift_budget"] == max_root_y_shift


def test_pose_optimizer_builds_models_no_larger_than_chunk_size(monkeypatch):
    batch_sizes = install_differentiable_fake_model(monkeypatch)
    record = make_pose_optimizer_record(n_frames=5)

    optimize_lower_body_pose_smpl(
        record,
        LowerBodyOptimizerConfig(
            fps=30.0,
            enable_pose_pass=True,
            pose_iterations=1,
            pose_lr=0.01,
            device="cpu",
            chunk_size=2,
        ),
    )

    assert batch_sizes == [2, 2, 1]
    assert max(batch_sizes) <= 2


def test_real_pose_optimizer_preserves_invariants_and_updates_existing_feet(monkeypatch):
    install_differentiable_fake_model(monkeypatch)
    record = make_pose_optimizer_record(n_frames=4)
    max_root_y_shift = 0.05

    out, reports = optimize_record(
        record,
        LowerBodyOptimizerConfig(
            fps=30.0,
            enable_pose_pass=True,
            pose_iterations=2,
            pose_lr=0.05,
            device="cpu",
            chunk_size=2,
            max_root_y_shift=max_root_y_shift,
        ),
    )

    mask = lower_body_pose_mask(out["pose"].shape[1])
    np.testing.assert_array_equal(out["betas"], record["betas"])
    np.testing.assert_array_equal(out["pose"][:, ~mask], record["pose"][:, ~mask])
    np.testing.assert_allclose(out["trans_world"][:, 0], record["trans_world"][:, 0], atol=1e-6)
    np.testing.assert_allclose(out["trans_world"][:, 2], record["trans_world"][:, 2], atol=1e-6)
    assert np.max(np.abs(out["trans_world"][:, 1] - record["trans_world"][:, 1])) <= max_root_y_shift + 1e-6
    assert reports["lower_body_optimization_report"]["pose_optimizer_used"] is True
    np.testing.assert_array_equal(out["feet"], out["feet_refined"])
    np.testing.assert_array_equal(out["feet_world"], out["feet_refined"])
