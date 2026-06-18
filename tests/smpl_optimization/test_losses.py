import torch

from lib.smpl_optimization.losses import (
    contact_ground_loss,
    foot_lock_loss,
    penetration_loss,
    smoothness_loss,
    stance_anchor_xz_loss,
    stance_anchor_y_loss,
    weighted_l2,
)


def test_penetration_loss_only_penalizes_points_below_ground():
    y = torch.tensor([[0.1, -0.2, -0.1]])
    loss = penetration_loss(y, ground_y=0.0)
    assert torch.isclose(loss, torch.tensor((0.2**2 + 0.1**2) / 3.0))


def test_contact_ground_loss_uses_soft_contact_weights():
    y = torch.tensor([[0.1, 0.3]])
    contact = torch.tensor([[1.0, 0.0]])
    loss = contact_ground_loss(y, contact, ground_y=0.0)
    assert torch.isclose(loss, torch.tensor(0.01))


def test_foot_lock_loss_uses_contact_pairs():
    points = torch.tensor([[[0.0, 0.0, 0.0]], [[0.2, 0.0, 0.0]], [[0.5, 0.0, 0.0]]])
    contact = torch.tensor([[1.0], [1.0], [0.0]])
    loss = foot_lock_loss(points, contact)
    assert torch.isclose(loss, torch.tensor(0.04))


def test_smoothness_loss_zero_for_linear_motion():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    assert smoothness_loss(x).item() == 0.0


def test_weighted_l2_supports_zero_weights():
    value = torch.tensor([1.0, 2.0])
    target = torch.zeros_like(value)
    weight = torch.tensor([0.0, 1.0])
    assert torch.isclose(weighted_l2(value, target, weight), torch.tensor(4.0))


def test_weighted_l2_returns_zero_for_all_zero_tensor_weights():
    value = torch.tensor([1.0, 2.0])
    target = torch.zeros_like(value)
    weight = torch.zeros_like(value)
    assert weighted_l2(value, target, weight).item() == 0.0


def test_weighted_l2_scalar_tensor_weight_matches_python_scalar_weight():
    value = torch.tensor([1.0, 2.0])
    target = torch.zeros_like(value)

    tensor_weight_loss = weighted_l2(value, target, torch.tensor(1.0))
    scalar_weight_loss = weighted_l2(value, target, 1.0)

    assert torch.isclose(tensor_weight_loss, scalar_weight_loss)


def test_temporal_losses_return_zero_for_short_sequences():
    points = torch.tensor([[[1.0, 0.0, 0.0]]])
    contact = torch.tensor([[1.0]])
    value = torch.tensor([[1.0], [2.0]])

    assert foot_lock_loss(points, contact).item() == 0.0
    assert smoothness_loss(value).item() == 0.0


def test_stance_anchor_losses_only_use_masked_frames():
    points = torch.tensor(
        [
            [[0.0, 0.0, 0.0]],
            [[1.0, 2.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    target_xz = torch.zeros((2, 1, 2), dtype=torch.float32)
    mask = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    assert torch.isclose(stance_anchor_xz_loss(points, target_xz, mask, huber_delta=10.0), torch.tensor(2.5))
    assert torch.isclose(stance_anchor_y_loss(points, ground_y=0.0, mask=mask, huber_delta=10.0), torch.tensor(2.0))
