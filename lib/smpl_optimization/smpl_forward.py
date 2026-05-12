from __future__ import annotations

import numpy as np
import torch


def expand_betas(betas, n_frames: int) -> np.ndarray:
    arr = np.asarray(betas, dtype=np.float32)
    expanded = np.zeros((n_frames, 10), dtype=np.float32)
    if arr.ndim == 1:
        cols = min(10, arr.shape[0])
        expanded[:, :cols] = arr[:cols][None]
        return expanded
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return expanded

    rows = min(n_frames, arr.shape[0])
    cols = min(10, arr.shape[1])
    expanded[:rows, :cols] = arr[:rows, :cols]
    if rows < n_frames:
        expanded[rows:] = expanded[rows - 1]
    return expanded


def smpl_forward_axis_angle(model, pose: torch.Tensor, betas: torch.Tensor, transl: torch.Tensor) -> dict:
    output = model.get_output(
        global_orient=pose[:, :3],
        body_pose=pose[:, 3:],
        betas=betas,
        transl=transl,
    )
    return {
        "vertices": output.vertices,
        "feet": output.feet,
    }
