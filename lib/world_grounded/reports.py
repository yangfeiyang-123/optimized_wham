from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: Any):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(data), indent=2), encoding="utf-8")


def contact_switch_rate(contact_confidence: np.ndarray, threshold: float = 0.5) -> float:
    active = np.asarray(contact_confidence) > float(threshold)
    if len(active) < 2:
        return 0.0
    return float(np.mean(active[1:] != active[:-1]))


def contact_coverage(contact_confidence: np.ndarray, threshold: float = 0.35) -> float:
    confidence = np.asarray(contact_confidence)
    if confidence.size == 0:
        return 0.0
    return float(np.mean(confidence > float(threshold)))
