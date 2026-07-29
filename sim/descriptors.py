"""Behavior descriptors. Stored from day one, unused in v1 (spec §3) — they exist so
the MAP-Elites grid can be backfilled retroactively over the whole back catalog.
"""
import numpy as np

from common.config import CLIP_FPS


def desc_velocity(frames: np.ndarray) -> float:
    """Mean horizontal root speed (m/s) over the episode."""
    dxy = np.diff(frames[:, 0:2], axis=0) * CLIP_FPS
    return float(np.linalg.norm(dxy, axis=1).mean())


def desc_torso_h(frames: np.ndarray) -> float:
    """Mean torso height (m) over the episode."""
    return float(frames[:, 2].mean())


def policy_descriptors(episodes: dict[int, np.ndarray]) -> tuple[float, float]:
    eps = list(episodes.values())
    return (
        float(np.mean([desc_velocity(e) for e in eps])),
        float(np.mean([desc_torso_h(e) for e in eps])),
    )
