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
        or trans_world.ndim != 2
        or trans_world.shape[0] <= 2
        or trans_world.shape[1] <= 1
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

    original_pose = original_pose.reshape(original_pose.shape[0], -1)
    corrected_pose = corrected_pose.reshape(corrected_pose.shape[0], -1)

    common_frames = min(original_pose.shape[0], corrected_pose.shape[0])
    common_values = min(original_pose.shape[1], corrected_pose.shape[1])
    if common_frames == 0 or common_values == 0:
        return 0.0

    return _clean_float(
        np.max(
            np.abs(
                original_pose[:common_frames, :common_values]
                - corrected_pose[:common_frames, :common_values]
            )
        )
    )


def temporal_derivative_summary(values, order=1):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.shape[0] <= int(order):
        return {"rms": 0.0, "max_abs": 0.0}

    flat = values.reshape(values.shape[0], -1)
    derivative = np.diff(flat, n=int(order), axis=0)
    if derivative.size == 0:
        return {"rms": 0.0, "max_abs": 0.0}

    return {
        "rms": _clean_float(np.sqrt(np.mean(derivative**2))),
        "max_abs": _clean_float(np.max(np.abs(derivative))),
    }


def pose_smoothness(pose):
    pose = np.asarray(pose)
    return {
        "velocity": temporal_derivative_summary(pose, order=1),
        "acceleration": temporal_derivative_summary(pose, order=2),
        "jerk": temporal_derivative_summary(pose, order=3),
    }


def root_translation_smoothness(trans):
    trans = np.asarray(trans)
    return {
        "velocity": temporal_derivative_summary(trans, order=1),
        "acceleration": temporal_derivative_summary(trans, order=2),
        "jerk": temporal_derivative_summary(trans, order=3),
    }
