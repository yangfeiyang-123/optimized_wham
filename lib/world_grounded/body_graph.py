from __future__ import annotations

from typing import Any, Sequence

import numpy as np


BODY_GRAPH = {
    "version": "body_graph_v1",
    "keypoints": [
        "pelvis",
        "spine",
        "thorax",
        "neck",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_hip",
        "left_knee",
        "left_ankle",
        "left_toe",
        "left_heel",
        "right_hip",
        "right_knee",
        "right_ankle",
        "right_toe",
        "right_heel",
    ],
    "edges": [
        ["pelvis", "spine"],
        ["spine", "thorax"],
        ["thorax", "neck"],
        ["neck", "head"],
        ["thorax", "left_shoulder"],
        ["left_shoulder", "left_elbow"],
        ["left_elbow", "left_wrist"],
        ["thorax", "right_shoulder"],
        ["right_shoulder", "right_elbow"],
        ["right_elbow", "right_wrist"],
        ["pelvis", "left_hip"],
        ["left_hip", "left_knee"],
        ["left_knee", "left_ankle"],
        ["left_ankle", "left_toe"],
        ["left_ankle", "left_heel"],
        ["pelvis", "right_hip"],
        ["right_hip", "right_knee"],
        ["right_knee", "right_ankle"],
        ["right_ankle", "right_toe"],
        ["right_ankle", "right_heel"],
        ["left_hip", "right_hip"],
        ["left_shoulder", "right_shoulder"],
    ],
    "default_weights": {
        "pelvis": 1.0,
        "thorax": 1.0,
        "head": 0.5,
        "left_wrist": 0.2,
        "right_wrist": 0.2,
        "left_toe": 1.0,
        "left_heel": 1.0,
        "right_toe": 1.0,
        "right_heel": 1.0,
    },
}


def graph_labels(graph: dict[str, Any] | None = None) -> list[str]:
    return [str(label) for label in (graph or BODY_GRAPH)["keypoints"]]


def edge_indices(labels: Sequence[str], graph: dict[str, Any] | None = None) -> list[tuple[int, int]]:
    index = {str(label): idx for idx, label in enumerate(labels)}
    edges: list[tuple[int, int]] = []
    for left, right in (graph or BODY_GRAPH)["edges"]:
        if left in index and right in index:
            edges.append((index[left], index[right]))
    return edges


def laplacian_coordinates(
    points: np.ndarray,
    labels: Sequence[str] | None = None,
    graph: dict[str, Any] | None = None,
) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"points must have shape [T, K, 3], got {arr.shape}")
    fitted_labels = list(labels) if labels is not None else graph_labels(graph)[: arr.shape[1]]
    if len(fitted_labels) != arr.shape[1]:
        raise ValueError("labels length must match points keypoint dimension")

    neighbors: list[list[int]] = [[] for _ in fitted_labels]
    for left, right in edge_indices(fitted_labels, graph):
        neighbors[left].append(right)
        neighbors[right].append(left)

    out = np.zeros_like(arr, dtype=np.float32)
    for idx, neighbor_ids in enumerate(neighbors):
        if neighbor_ids:
            out[:, idx] = arr[:, idx] - np.mean(arr[:, neighbor_ids], axis=1)
    return out
