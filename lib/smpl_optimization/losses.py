import torch


def _zero_like_scalar(value):
    return value.new_zeros(())


def weighted_l2(value, target, weight=1.0):
    error = (value - target) ** 2

    if torch.is_tensor(weight):
        if weight.ndim == 0:
            return error.mean() * weight

        weighted_error = error * weight
        weight_sum = weight.sum()
        if weight_sum.item() == 0:
            return _zero_like_scalar(value)
        return weighted_error.sum() / weight_sum

    return (error * weight).mean()


def penetration_loss(foot_y, ground_y=0.0):
    penetration = torch.clamp(ground_y - foot_y, min=0.0)
    return (penetration**2).mean()


def contact_ground_loss(foot_y, contact, ground_y=0.0):
    target = torch.as_tensor(ground_y, dtype=foot_y.dtype, device=foot_y.device)
    return weighted_l2(foot_y, target, contact)


def foot_lock_loss(points, contact):
    if points.shape[0] < 2:
        return _zero_like_scalar(points)

    velocity = points[1:] - points[:-1]
    pair_contact = contact[1:] * contact[:-1]
    return weighted_l2(velocity, torch.zeros_like(velocity), pair_contact.unsqueeze(-1))


def smoothness_loss(value):
    if value.shape[0] < 3:
        return _zero_like_scalar(value)

    acceleration = value[2:] - (2 * value[1:-1]) + value[:-2]
    return (acceleration**2).mean()


def huber_l2(error, delta=0.03):
    threshold = torch.as_tensor(delta, dtype=error.dtype, device=error.device)
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, threshold)
    linear = abs_error - quadratic
    return 0.5 * quadratic**2 + threshold * linear


def stance_anchor_xz_loss(points, target_xz, mask, huber_delta=0.03):
    if points.numel() == 0:
        return _zero_like_scalar(points)
    error = points[..., [0, 2]] - target_xz
    weight = mask.to(dtype=points.dtype).unsqueeze(-1)
    denom = torch.clamp(weight.sum() * 2.0, min=1.0)
    return (huber_l2(error, huber_delta) * weight).sum() / denom


def stance_anchor_y_loss(points, ground_y=0.0, mask=None, huber_delta=0.03):
    if points.numel() == 0:
        return _zero_like_scalar(points)
    target = torch.as_tensor(ground_y, dtype=points.dtype, device=points.device)
    error = points[..., 1] - target
    if mask is None:
        return huber_l2(error, huber_delta).mean()
    weight = mask.to(dtype=points.dtype)
    denom = torch.clamp(weight.sum(), min=1.0)
    return (huber_l2(error, huber_delta) * weight).sum() / denom
