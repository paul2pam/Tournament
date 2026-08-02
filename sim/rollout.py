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


def rollout_policy(
    net,
    policy_id: int,
    base_seed: int,
    n_clips: int = CLIPS_PER_POLICY,
    prefer_alive: bool = False,
    prefer_still: bool = False,
):
    """Roll out a policy into candidate clips.

    prefer_alive: sample extra candidate windows and keep the ones that pass the
    degeneracy check / have the most upright frames. Used for the bootstrap seed
    incumbent, whose weak policy spends most of each episode fallen — better to
    show it trying and falling than the aftermath. Normal challenger rollouts
    keep unbiased sampling (spec §7: window choice must not become a tell).

    prefer_still: keep the LEAST dynamic windows — the ragdoll attention-check
    policy, where the spec wants an *obvious* corpse, not the dramatic fall.

    Returns (clips, episodes):
      clips: list of {seed, window_start_s, blob_key, frames} — frames kept in-memory
             for the degeneracy filter and descriptors; blob already written.
      episodes: {seed: full episode frames} for policy-level descriptors.
    """
    from sim.degeneracy import check_clip

    n_episodes = max(1, n_clips // 2)
    per_ep = n_clips // n_episodes
    rng = np.random.default_rng(base_seed)
    clips, episodes = [], {}
    for e in range(n_episodes):
        seed = base_seed + e
        ep = run_episode(net, seed)
        episodes[seed] = ep
        starts = sample_windows(rng, n=10 if (prefer_alive or prefer_still) else per_ep)
        windows = [(s, ep[s : s + CLIP_FRAMES]) for s in starts]
        if prefer_alive:
            windows.sort(
                key=lambda sw: (
                    check_clip(sw[1]) is None,          # passing the filter first
                    float((sw[1][:, 2] > 0.9).mean()),  # then most upright frames
                ),
                reverse=True,
            )
            windows = windows[:per_ep]
        elif prefer_still:
            windows.sort(key=lambda sw: float(np.abs(np.diff(sw[1][:, 7:], axis=0)).max()))
            windows = windows[:per_ep]
        for start, window in windows:
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
