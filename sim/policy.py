"""Policy network shared by training, challenger generation, and rollout.

Diagonal-Gaussian MLP actor + value head (CleanRL style). Deterministic mode
(rollout) uses the mean action; stochastic mode (training) samples.
"""
import numpy as np
import torch
import torch.nn as nn

OBS_DIM = 53
ACT_DIM = 21
HIDDEN = 256


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            _layer_init(nn.Linear(OBS_DIM, HIDDEN)), nn.Tanh(),
            _layer_init(nn.Linear(HIDDEN, HIDDEN)), nn.Tanh(),
            _layer_init(nn.Linear(HIDDEN, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            _layer_init(nn.Linear(OBS_DIM, HIDDEN)), nn.Tanh(),
            _layer_init(nn.Linear(HIDDEN, HIDDEN)), nn.Tanh(),
            _layer_init(nn.Linear(HIDDEN, ACT_DIM), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, ACT_DIM))

    def value(self, x):
        return self.critic(x)

    def act(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(x)

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.actor_mean(x).squeeze(0).numpy()


def save_policy(net: ActorCritic, path) -> None:
    torch.save({"state_dict": net.state_dict()}, path)


def load_policy(path) -> ActorCritic:
    net = ActorCritic()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net
