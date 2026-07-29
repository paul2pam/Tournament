"""Challenger generation: Gaussian weight perturbation at varied scales + a short
stay-alive PPO fine-tune so large perturbations stay viable instead of instantly
ragdolling (decided with user; spec §7 mandates the *varied* scale spread).

Only actor parameters are perturbed — the critic is left intact so the fine-tune
starts from a sane value function.
"""
import copy

import torch

from common.config import FINETUNE_STEPS, PERTURB_SCALES, POOL_SIZE
from sim.policy import ActorCritic
from sim.train_stand import train

FINETUNE_ENVS = 8


def perturb(net: ActorCritic, scale: float, seed: int) -> ActorCritic:
    g = torch.Generator().manual_seed(seed)
    child = copy.deepcopy(net)
    with torch.no_grad():
        for name, p in child.named_parameters():
            if name.startswith("actor"):
                p.add_(torch.randn(p.shape, generator=g) * scale * (p.std() + 1e-8))
    return child


def make_challenger(incumbent: ActorCritic, scale: float, seed: int, finetune: bool = True) -> ActorCritic:
    child = perturb(incumbent, scale, seed)
    if finetune:
        train(
            child,
            total_steps=FINETUNE_STEPS,
            num_envs=FINETUNE_ENVS,
            num_steps=256,
            lr=1e-4,
            anneal_lr=False,
            seed=seed,
            quiet=True,
        )
    return child


def challenger_scales(pool_size: int = POOL_SIZE) -> list[float]:
    """Varied spread of perturbation scales, cycling the configured ladder."""
    return [PERTURB_SCALES[i % len(PERTURB_SCALES)] for i in range(pool_size)]
