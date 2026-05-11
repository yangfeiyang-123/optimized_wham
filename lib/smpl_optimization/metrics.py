import numpy as np


def _clean_float(value):
    return float(round(float(value), 8))


def beta_variation_max_abs(betas):
    betas = np.asarray(betas)
    if betas.size == 0 or betas.ndim < 2 or betas.shape[0] <= 1:
        return 0.0
    return _clean_float(np.max(np.abs(betas - betas[0:1])))


def foot_penetration_depth(foot_y, ground_y=0.0):
    foot_y = np.asarray(foot_y)
    if foot_y.size == 0:
        return {
            "max_penetration": 0.0,
            "mean_penetration": 0.0,
            "num_penetrating": 0,
        }

    penetration = np.maximum(float(ground_y) - foot_y, 0.0)
    return {
        "max_penetration": _clean_float(np.max(penetration)),
        "mean_penetration": _clean_float(np.mean(penetration)),
        "num_penetrating": int(np.count_nonzero(penetration > 0.0)),
    }


def contact_foot_sliding(points, contact, fps, threshold=0.15):
    points = np.asarray(points)
    contact = np.asarray(contact)
    if points.size == 0 or contact.size == 0 or points.ndim < 3 or points.shape[0] <= 1:
        return {
            "mean_contact_speed": 0.0,
            "max_contact_speed": 0.0,
            "num_sliding": 0,
        }

    horizontal_points = points[..., [0, 2]]
    speed = np.linalg.norm(np.diff(horizontal_points, axis=0), axis=-1) * float(fps)
    contact_pairs = (contact[:-1] > 0.0) & (contact[1:] > 0.0)
    contact_speed = speed[contact_pairs]
    if contact_speed.size == 0:
        return {
            "mean_contact_speed": 0.0,
            "max_contact_speed": 0.0,
            "num_sliding": 0,
        }

    return {
        "mean_contact_speed": _clean_float(np.mean(contact_speed)),
        "max_contact_speed": _clean_float(np.max(contact_speed)),
        "num_sliding": int(np.count_nonzero(contact_speed > float(threshold))),
    }


def root_vertical_jitter(trans_world):
    trans_world = np.asarray(trans_world)
    if (
        trans_world.size == 0
        or trans_world.ndim < 2
        or trans_world.shape[0] <= 2
        or trans_world.shape[-1] <= 1
    ):
        return {
            "rms_vertical_accel": 0.0,
            "max_vertical_accel": 0.0,
        }

    vertical_accel = np.diff(trans_world[:, 1], n=2)
    return {
        "rms_vertical_accel": _clean_float(np.sqrt(np.mean(vertical_accel**2))),
        "max_vertical_accel": _clean_float(np.max(np.abs(vertical_accel))),
    }


def pose_delta_max_abs(original_pose, corrected_pose):
    original_pose = np.asarray(original_pose)
    corrected_pose = np.asarray(corrected_pose)
    if original_pose.size == 0 or corrected_pose.size == 0:
        return 0.0

    if original_pose.ndim == 1:
        original_pose = original_pose.reshape(1, -1)
    if corrected_pose.ndim == 1:
        corrected_pose = corrected_pose.reshape(1, -1)

    common_shape = tuple(
        min(original_pose.shape[axis], corrected_pose.shape[axis])
        for axis in range(min(original_pose.ndim, corrected_pose.ndim))
    )
    if any(size == 0 for size in common_shape):
        return 0.0

    slices = tuple(slice(0, size) for size in common_shape)
    return _clean_float(np.max(np.abs(original_pose[slices] - corrected_pose[slices])))
