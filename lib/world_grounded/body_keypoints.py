"""Reconstruct BODY_GRAPH world-frame keypoints from SMPL mesh vertices.

The upstream WHAM / world-grounded / lower-body pickles carry ``verts`` (the full
SMPL mesh, world frame, Y-up) but do not carry an explicit joint array. The
reference bundle needs the 21 BODY_GRAPH keypoints to compute ``body_laplacian``
(the structure-preserving reward signal). Rather than re-running a torch SMPL
forward pass, we regress the joints straight from ``verts`` with the same
J_regressor matrices WHAM already ships:

* ``J_regressor_h36m`` (17, 6890) -> torso / limb joints in the well-known H36M order.
* ``J_regressor_feet`` (4, 6890)  -> heel/toe points, already validated to match
  the pickle's ``feet_world`` to < 0.2 mm.

Everything here is plain NumPy so it can be unit-tested without model files.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from lib.world_grounded.body_graph import graph_labels

# H36M 17-joint order produced by J_regressor_h36m (see lib/utils/kp_utils.get_h36m_joint_names).
H36M_JOINT_NAMES = (
    "hip", "lhip", "lknee", "lankle", "rhip", "rknee", "rankle",
    "spine", "neck", "head", "headtop",
    "lshoulder", "lelbow", "lwrist", "rshoulder", "relbow", "rwrist",
)
_H36M = {name: idx for idx, name in enumerate(H36M_JOINT_NAMES)}

# J_regressor_feet row order (see lib/world_grounded/foot_points.DEFAULT_FOOT_LABELS):
# left_heel_or_ankle, left_toe_or_forefoot, right_heel_or_ankle, right_toe_or_forefoot.
_FEET = {"left_heel": 0, "left_toe": 1, "right_heel": 2, "right_toe": 3}

# How each BODY_GRAPH keypoint is sourced. Values are either:
#   ("h36m", name)            -> take that H36M joint,
#   ("feet", name)            -> take that foot point,
#   ("mid_h36m", (a, b))      -> midpoint of two H36M joints (thorax = between shoulders).
_KEYPOINT_SOURCES: dict[str, tuple] = {
    "pelvis": ("h36m", "hip"),
    "spine": ("h36m", "spine"),
    "thorax": ("mid_h36m", ("lshoulder", "rshoulder")),
    "neck": ("h36m", "neck"),
    "head": ("h36m", "head"),
    "left_shoulder": ("h36m", "lshoulder"),
    "left_elbow": ("h36m", "lelbow"),
    "left_wrist": ("h36m", "lwrist"),
    "right_shoulder": ("h36m", "rshoulder"),
    "right_elbow": ("h36m", "relbow"),
    "right_wrist": ("h36m", "rwrist"),
    "left_hip": ("h36m", "lhip"),
    "left_knee": ("h36m", "lknee"),
    "left_ankle": ("h36m", "lankle"),
    "left_toe": ("feet", "left_toe"),
    "left_heel": ("feet", "left_heel"),
    "right_hip": ("h36m", "rhip"),
    "right_knee": ("h36m", "rknee"),
    "right_ankle": ("h36m", "rankle"),
    "right_toe": ("feet", "right_toe"),
    "right_heel": ("feet", "right_heel"),
}


def assemble_body_graph_keypoints(h36m: np.ndarray, feet: np.ndarray) -> np.ndarray:
    """Assemble the 21 BODY_GRAPH keypoints (in graph order) from H36M joints and feet.

    Args:
        h36m: ``[T, 17, 3]`` joints in H36M order (see ``H36M_JOINT_NAMES``).
        feet: ``[T, 4, 3]`` points in ``[left_heel, left_toe, right_heel, right_toe]`` order.

    Returns:
        ``[T, 21, 3]`` array in the exact order of ``BODY_GRAPH['keypoints']``.
    """
    h36m = np.asarray(h36m, dtype=np.float32)
    feet = np.asarray(feet, dtype=np.float32)
    if h36m.ndim != 3 or h36m.shape[1] < len(H36M_JOINT_NAMES) or h36m.shape[2] != 3:
        raise ValueError(f"h36m must be [T, >={len(H36M_JOINT_NAMES)}, 3], got {h36m.shape}")
    if feet.ndim != 3 or feet.shape[1] < 4 or feet.shape[2] != 3:
        raise ValueError(f"feet must be [T, >=4, 3], got {feet.shape}")
    if h36m.shape[0] != feet.shape[0]:
        raise ValueError("h36m and feet must share the frame dimension.")

    labels = graph_labels()
    out = np.zeros((h36m.shape[0], len(labels), 3), dtype=np.float32)
    for idx, label in enumerate(labels):
        kind, ref = _KEYPOINT_SOURCES[label]
        if kind == "h36m":
            out[:, idx] = h36m[:, _H36M[ref]]
        elif kind == "feet":
            out[:, idx] = feet[:, _FEET[ref]]
        elif kind == "mid_h36m":
            left, right = ref
            out[:, idx] = 0.5 * (h36m[:, _H36M[left]] + h36m[:, _H36M[right]])
        else:  # pragma: no cover - guarded by the static table above
            raise ValueError(f"Unknown keypoint source kind: {kind}")
    return out


def body_keypoints_from_verts(
    verts: np.ndarray,
    j_regressor_h36m: np.ndarray,
    j_regressor_feet: np.ndarray,
) -> np.ndarray:
    """Regress BODY_GRAPH keypoints directly from world-frame SMPL vertices.

    Args:
        verts: ``[T, 6890, 3]`` world-frame SMPL vertices (Y-up, as stored in the pickles).
        j_regressor_h36m: ``[17, 6890]`` H36M joint regressor.
        j_regressor_feet: ``[4, 6890]`` heel/toe regressor.

    Returns:
        ``[T, 21, 3]`` BODY_GRAPH keypoints in graph order, same frame/units as ``verts``.
    """
    verts = np.asarray(verts, dtype=np.float32)
    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(f"verts must be [T, V, 3], got {verts.shape}")
    j_h36m = np.asarray(j_regressor_h36m, dtype=np.float32)
    j_feet = np.asarray(j_regressor_feet, dtype=np.float32)
    if j_h36m.shape[1] != verts.shape[1] or j_feet.shape[1] != verts.shape[1]:
        raise ValueError("Regressor vertex dimension does not match verts.")
    h36m = np.einsum("kv,tvc->tkc", j_h36m, verts)
    feet = np.einsum("kv,tvc->tkc", j_feet, verts)
    return assemble_body_graph_keypoints(h36m, feet)


def load_default_regressors() -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load (J_regressor_h36m, J_regressor_feet) from the repo's body_models.

    Resolves the regressor files relative to this file's repository root (which
    works in any Python environment, torch or not). Falls back to
    ``configs.constants`` paths if the default location is missing. Returns
    ``None`` when the regressors are unavailable so callers can degrade
    gracefully to a warning instead of crashing the export.
    """
    from pathlib import Path

    candidates: list[tuple[Path, Path]] = []
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(
        (
            repo_root / "body_models" / "J_regressor_h36m.npy",
            repo_root / "body_models" / "J_regressor_feet.npy",
        )
    )
    try:  # optional: honour an explicit constants override if importable
        from configs import constants as _C

        candidates.append(
            (Path(_C.BMODEL.JOINTS_REGRESSOR_H36M), Path(_C.BMODEL.JOINTS_REGRESSOR_FEET))
        )
    except Exception:
        pass

    for h36m_path, feet_path in candidates:
        if not (h36m_path.exists() and feet_path.exists()):
            continue
        try:
            j_h36m = np.load(h36m_path)
            j_feet = np.load(feet_path)
        except Exception:
            continue
        if j_h36m.shape[0] < 17 or j_feet.shape[0] < 4:
            continue
        return np.asarray(j_h36m, dtype=np.float32), np.asarray(j_feet, dtype=np.float32)
    return None
