"""Parallel vector env: one process per HumanoidEnv, pipe-scatter/gather per step.

The sync VecEnv in train_stand.py steps envs sequentially in-process — right for a
laptop, wrong for a 256-core box. This one steps all envs concurrently; the training
loop's throughput then bounds at the PPO update instead of physics.
"""
import multiprocessing as mp

import numpy as np

OBS_DIM = 53
ACT_DIM = 21


def _worker(pipe, seed: int, horizon: int, task: str):
    # Import inside the child so 'spawn' start method works (macOS default).
    from sim.env import HumanoidEnv

    env = HumanoidEnv(seed=seed, task=task)
    t, ep_ret = 0, 0.0
    obs = env.reset()
    pipe.send(obs)
    while True:
        msg = pipe.recv()
        if msg is None:
            break
        obs, r, term = env.step(msg)
        t += 1
        ep_ret += r
        done = term or t >= horizon
        finished = ep_ret if done else None
        if done:
            obs = env.reset()
            t, ep_ret = 0, 0.0
        pipe.send((obs, r, done, term, finished))
    pipe.close()


class SubprocVecEnv:
    """Same interface as train_stand.VecEnv: reset(), step(), finished_returns."""

    obs_dim = OBS_DIM
    act_dim = ACT_DIM

    def __init__(self, n: int, seed: int, horizon: int = 1000, task: str = "stand"):
        ctx = mp.get_context()
        self.pipes, self.procs = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, seed + i, horizon, task), daemon=True)
            p.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(p)
        self._initial_obs = np.stack([pipe.recv() for pipe in self.pipes])
        self.finished_returns: list[float] = []

    def reset(self) -> np.ndarray:
        # Workers reset themselves on episode end; initial obs collected at spawn.
        return self._initial_obs

    def step(self, actions: np.ndarray):
        """Returns (obs, rew, done, term). done includes horizon truncation; term is
        true environment termination only (SAC bootstraps through truncation)."""
        for pipe, a in zip(self.pipes, actions):
            pipe.send(a)
        obs, rews, dones, terms = [], [], [], []
        for pipe in self.pipes:
            o, r, d, t, fin = pipe.recv()
            obs.append(o)
            rews.append(r)
            dones.append(d)
            terms.append(t)
            if fin is not None:
                self.finished_returns.append(fin)
        return (
            np.stack(obs),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(terms, dtype=np.float32),
        )

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(None)
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=2)
