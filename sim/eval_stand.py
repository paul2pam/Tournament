"""Phase 0 checkpoint: policy survives 10s upright across 20 seeds. Exits nonzero on failure.

Run:  python -m sim.eval_stand [--ckpt checkpoints/stand.pt]
"""
import argparse
import sys

from sim.env import HumanoidEnv
from sim.policy import load_policy
from common.config import EPISODE_FRAMES


def survives(net, seed: int, n_steps: int = EPISODE_FRAMES) -> tuple[bool, int]:
    env = HumanoidEnv(seed=seed)
    obs = env.reset(seed=seed)
    for t in range(n_steps):
        obs, _, terminated = env.step(net.act_deterministic(obs))
        if terminated:
            return False, t
    return True, n_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/stand.pt")
    p.add_argument("--n-seeds", type=int, default=20)
    args = p.parse_args()

    net = load_policy(args.ckpt)
    failures = 0
    for seed in range(args.n_seeds):
        ok, t = survives(net, seed)
        status = "OK " if ok else f"FELL at {t / 30.0:.2f}s"
        print(f"seed {seed:2d}: {status}")
        failures += not ok

    if failures:
        print(f"FAIL: {failures}/{args.n_seeds} seeds fell before 10s")
        sys.exit(1)
    print(f"PASS: all {args.n_seeds} seeds upright for 10s")


if __name__ == "__main__":
    main()
