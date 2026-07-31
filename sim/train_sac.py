"""SAC for the standing task — the reliability path where PPO plateaus.

Off-policy, twin Q, squashed-Gaussian actor, auto-tuned alpha (CleanRL-style).
GPU-aware: gradient updates dominate SAC wall-clock and the CSL boxes have L40S
GPUs, so device is cuda when available. Env stepping stays on CPU via SubprocVecEnv.

Checkpoints save as {"format": "sac", "actor": ...}; sim/distill.py converts the
final actor into the pipeline's ActorCritic format.

Run:  python -m sim.train_sac --total-steps 2000000 --out checkpoints/stand_sac.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.env import HumanoidEnv
from sim.vecenv import SubprocVecEnv

OBS_DIM, ACT_DIM = 53, 21
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class SoftQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class SACActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, ACT_DIM)
        self.log_std = nn.Linear(256, ACT_DIM)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        x = dist.rsample()
        y = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)
        return y, logp.sum(-1, keepdim=True)

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray, device="cpu") -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mean, _ = self(x)
        return torch.tanh(mean).squeeze(0).cpu().numpy()


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.obs = np.empty((capacity, OBS_DIM), dtype=np.float32)
        self.next_obs = np.empty((capacity, OBS_DIM), dtype=np.float32)
        self.act = np.empty((capacity, ACT_DIM), dtype=np.float32)
        self.rew = np.empty((capacity, 1), dtype=np.float32)
        self.term = np.empty((capacity, 1), dtype=np.float32)
        self.idx, self.full = 0, False

    def add(self, obs, act, rew, next_obs, term):
        i = self.idx
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.term[i] = next_obs, term
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch: int, device):
        j = np.random.randint(0, len(self), size=batch)
        to = lambda a: torch.as_tensor(a[j], device=device)
        return to(self.obs), to(self.act), to(self.rew), to(self.next_obs), to(self.term)


def evaluate(actor, device, n_seeds: int = 3, n_steps: int = 300) -> float:
    """Mean survival seconds over deterministic episodes (10s max)."""
    times = []
    for seed in range(1000, 1000 + n_seeds):
        env = HumanoidEnv(seed=seed)
        obs = env.reset(seed=seed)
        t = n_steps
        for i in range(n_steps):
            obs, _, term = env.step(actor.act_deterministic(obs, device))
            if term:
                t = i
                break
        times.append(t / 30.0)
    return float(np.mean(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--q-lr", type=float, default=1e-3)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/stand_sac.pt")
    p.add_argument("--eval-every", type=int, default=50_000)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"device: {device}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    actor = SACActor().to(device)
    q1, q2 = SoftQ().to(device), SoftQ().to(device)
    q1_t, q2_t = SoftQ().to(device), SoftQ().to(device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.q_lr)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    target_entropy = -float(ACT_DIM)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.q_lr)

    buf = ReplayBuffer(args.buffer_size)
    venv = SubprocVecEnv(args.num_envs, seed=args.seed * 1000 + 1)
    obs = venv.reset()

    t0 = time.time()
    global_step = 0
    n_updates = 0
    best_survival = 0.0

    while global_step < args.total_steps:
        if global_step < args.learning_starts:
            actions = np.random.uniform(-1, 1, size=(args.num_envs, ACT_DIM)).astype(np.float32)
        else:
            with torch.no_grad():
                a, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device))
            actions = a.cpu().numpy()

        next_obs, rews, dones, terms = venv.step(actions)
        for i in range(args.num_envs):
            # truncation (horizon) is not termination: bootstrap through it
            buf.add(obs[i], actions[i], rews[i], next_obs[i], terms[i])
        obs = next_obs
        global_step += args.num_envs

        if global_step >= args.learning_starts:
            for _ in range(args.num_envs):
                b_obs, b_act, b_rew, b_next, b_term = buf.sample(args.batch_size, device)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    na, nlogp = actor.sample(b_next)
                    tq = torch.min(q1_t(b_next, na), q2_t(b_next, na)) - alpha * nlogp
                    target = b_rew + args.gamma * (1.0 - b_term) * tq
                q_loss = F.mse_loss(q1(b_obs, b_act), target) + F.mse_loss(q2(b_obs, b_act), target)
                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()

                n_updates += 1
                if n_updates % 2 == 0:
                    pa, plogp = actor.sample(b_obs)
                    a_loss = (log_alpha.exp().detach() * plogp - torch.min(q1(b_obs, pa), q2(b_obs, pa))).mean()
                    a_opt.zero_grad()
                    a_loss.backward()
                    a_opt.step()

                    alpha_loss = (-log_alpha.exp() * (plogp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()

                with torch.no_grad():
                    for tp, sp in zip(q1_t.parameters(), q1.parameters()):
                        tp.lerp_(sp, args.tau)
                    for tp, sp in zip(q2_t.parameters(), q2.parameters()):
                        tp.lerp_(sp, args.tau)

        if global_step % args.eval_every < args.num_envs:
            surv = evaluate(actor, device)
            recent = venv.finished_returns[-20:]
            mean_ret = float(np.mean(recent)) if recent else float("nan")
            sps = int(global_step / (time.time() - t0))
            print(
                f"step {global_step}  ep_return(last20) {mean_ret:.1f}  "
                f"eval_survival {surv:.2f}s  alpha {log_alpha.exp().item():.3f}  sps {sps}",
                flush=True,
            )
            torch.save({"format": "sac", "actor": actor.state_dict()}, args.out)
            if surv > best_survival:
                best_survival = surv
                torch.save({"format": "sac", "actor": actor.state_dict()}, args.out + ".best")
            if surv >= 10.0:
                print("eval survival hit 10s — standing solved, stopping early", flush=True)
                break

    torch.save({"format": "sac", "actor": actor.state_dict()}, args.out)
    venv.close()
    print(f"done: best eval survival {best_survival:.2f}s; saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
