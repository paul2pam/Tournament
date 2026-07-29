"""Phase 0: PPO training of a standing policy (CleanRL-style, in-process vector env).

Run:  python -m sim.train_stand --total-steps 3000000 --out checkpoints/stand.pt
Also imported by sim/challengers.py for short stay-alive fine-tunes.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sim.env import HumanoidEnv
from sim.policy import ActorCritic, save_policy


class VecEnv:
    """Synchronous in-process vector of HumanoidEnvs with autoreset."""

    def __init__(self, n: int, seed: int, horizon: int = 1000):
        self.envs = [HumanoidEnv(seed=seed + i) for i in range(n)]
        self.horizon = horizon
        self.t = np.zeros(n, dtype=int)
        self.ep_ret = np.zeros(n)
        self.finished_returns: list[float] = []

    def reset(self) -> np.ndarray:
        self.t[:] = 0
        self.ep_ret[:] = 0
        return np.stack([e.reset() for e in self.envs])

    def step(self, actions: np.ndarray):
        obs, rews, dones = [], [], []
        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, term = env.step(a)
            self.t[i] += 1
            self.ep_ret[i] += r
            done = term or self.t[i] >= self.horizon
            if done:
                self.finished_returns.append(self.ep_ret[i])
                o = env.reset()
                self.t[i] = 0
                self.ep_ret[i] = 0
            obs.append(o)
            rews.append(r)
            dones.append(done)
        return np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones, dtype=np.float32)


def train(
    net: ActorCritic,
    total_steps: int = 3_000_000,
    num_envs: int = 16,
    num_steps: int = 128,
    lr: float = 3e-4,
    anneal_lr: bool = True,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    update_epochs: int = 10,
    num_minibatches: int = 32,
    clip_coef: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 0,
    log_every: int = 5,
    save_path: str | None = None,
    quiet: bool = False,
) -> ActorCritic:
    torch.manual_seed(seed)
    np.random.seed(seed)

    venv = VecEnv(num_envs, seed=seed * 1000 + 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_iterations = max(1, total_steps // batch_size)

    obs_buf = torch.zeros(num_steps, num_envs, venv.envs[0].obs_dim)
    act_buf = torch.zeros(num_steps, num_envs, venv.envs[0].act_dim)
    logp_buf = torch.zeros(num_steps, num_envs)
    rew_buf = torch.zeros(num_steps, num_envs)
    done_buf = torch.zeros(num_steps, num_envs)
    val_buf = torch.zeros(num_steps, num_envs)

    next_obs = torch.as_tensor(venv.reset(), dtype=torch.float32)
    next_done = torch.zeros(num_envs)
    t0 = time.time()
    global_step = 0

    for it in range(1, num_iterations + 1):
        if anneal_lr:
            frac = 1.0 - (it - 1) / num_iterations
            opt.param_groups[0]["lr"] = frac * lr

        for s in range(num_steps):
            global_step += num_envs
            obs_buf[s] = next_obs
            done_buf[s] = next_done
            with torch.no_grad():
                action, logp, _, value = net.act(next_obs)
            act_buf[s] = action
            logp_buf[s] = logp
            val_buf[s] = value.flatten()
            o, r, d = venv.step(action.numpy())
            rew_buf[s] = torch.as_tensor(r)
            next_obs = torch.as_tensor(o, dtype=torch.float32)
            next_done = torch.as_tensor(d)

        # GAE
        with torch.no_grad():
            next_value = net.value(next_obs).flatten()
            adv = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for s in reversed(range(num_steps)):
                if s == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[s + 1]
                    nextvalues = val_buf[s + 1]
                delta = rew_buf[s] + gamma * nextvalues * nextnonterminal - val_buf[s]
                adv[s] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            ret = adv + val_buf

        b_obs = obs_buf.reshape(batch_size, -1)
        b_act = act_buf.reshape(batch_size, -1)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)

        idx = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, minibatch_size):
                mb = idx[start : start + minibatch_size]
                _, newlogp, entropy, newval = net.act(b_obs[mb], b_act[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1 - clip_coef, 1 + clip_coef),
                ).mean()
                v_loss = 0.5 * ((newval.flatten() - b_ret[mb]) ** 2).mean()
                loss = pg_loss - ent_coef * entropy.mean() + vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                opt.step()

        if not quiet and (it % log_every == 0 or it == num_iterations):
            recent = venv.finished_returns[-20:]
            mean_ret = float(np.mean(recent)) if recent else float("nan")
            sps = int(global_step / (time.time() - t0))
            print(
                f"iter {it}/{num_iterations}  steps {global_step}  "
                f"ep_return(last20) {mean_ret:.1f}  sps {sps}",
                flush=True,
            )
            if save_path:
                save_policy(net, save_path)

    if save_path:
        save_policy(net, save_path)
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/stand.pt")
    p.add_argument("--resume", type=str, default=None, help="checkpoint to continue from")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        from sim.policy import load_policy

        net = load_policy(args.resume)
        net.train()
        print(f"resumed from {args.resume}")
    else:
        net = ActorCritic()
    train(
        net,
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        save_path=args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
