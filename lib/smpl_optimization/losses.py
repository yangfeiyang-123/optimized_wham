import torch


def _zero_like_scalar(value):
    return value.new_zeros(())


def weighted_l2(value, target, weight=1.0):
    error = (value - target) ** 2

    if torch.is_tensor(weight):
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
