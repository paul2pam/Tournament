"""Deterministic episode rollout -> 3s clip windows -> gzipped trajectory JSON blobs.

An episode always runs the full 10s wall-clock regardless of falls — termination is a
training concept; for clips, the fall itself is content (the degeneracy filter judges it).
"""
import numpy as np

from common import blobs
from common.config import (
    CLIP_FRAMES,
    CLIP_FPS,
    CLIPS_PER_POLICY,
    EPISODE_FRAMES,
    SETTLE_SECONDS,
)
from sim.env import HumanoidEnv


def run_episode(net, seed: int) -> np.ndarray:
    """Returns qpos frames, shape (EPISODE_FRAMES, 28), recorded at 30 fps."""
    env = HumanoidEnv(seed=seed)
    obs = env.reset(seed=seed)
    frames = np.empty((EPISODE_FRAMES, env.model.nq), dtype=np.float64)
    for t in range(EPISODE_FRAMES):
        obs, _, _ = env.step(net.act_deterministic(obs))
        frames[t] = env.qpos()
    return frames


def sample_windows(rng: np.random.Generator, n: int = 2) -> list[int]:
    """Window start frames: skip settling, spread starts across the episode via bins."""
    lo = int(SETTLE_SECONDS * CLIP_FPS)          # 15
    hi = EPISODE_FRAMES - CLIP_FRAMES            # 210
    edges = np.linspace(lo, hi, n + 1)
    return [int(rng.integers(int(edges[i]), int(edges[i + 1]) + 1)) for i in range(n)]


def clip_json(window: np.ndarray) -> dict:
    """Trajectory format shipped to the browser. ~7KB gzipped at 4-decimal precision."""
    w = np.round(window, 4)
    return {
        "fps": CLIP_FPS,
        "n_frames": int(w.shape[0]),
        "root_pos": w[:, 0:3].tolist(),
        "root_quat": w[:, 3:7].tolist(),   # (w, x, y, z)
        "joints": w[:, 7:].tolist(),
    }


def rollout_policy(net, policy_id: int, base_seed: int, n_clips: int = CLIPS_PER_POLICY):
    """Roll out a policy into candidate clips.

    Returns (clips, episodes):
      clips: list of {seed, window_start_s, blob_key, frames} — frames kept in-memory
             for the degeneracy filter and descriptors; blob already written.
      episodes: {seed: full episode frames} for policy-level descriptors.
    """
    n_episodes = max(1, n_clips // 2)
    rng = np.random.default_rng(base_seed)
    clips, episodes = [], {}
    for e in range(n_episodes):
        seed = base_seed + e
        ep = run_episode(net, seed)
        episodes[seed] = ep
        for start in sample_windows(rng, n=n_clips // n_episodes):
            window = ep[start : start + CLIP_FRAMES]
            key = f"clips/{policy_id}/{seed}_{start}.json.gz"
            blobs.put_json(key, clip_json(window))
            clips.append(
                {
                    "seed": seed,
                    "window_start_s": start / CLIP_FPS,
                    "blob_key": key,
                    "frames": window,
                }
            )
    return clips, episodes
