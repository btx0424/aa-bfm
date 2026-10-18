"""Discriminators on ``z`` only.

The positive class is ``B`` of dataset frames. The negative class is ``B`` of
visited states. ``compute_loss`` trains the score. ``reward`` is what ``Q_D``
backs up.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _accuracy(score_data: torch.Tensor, score_policy: torch.Tensor) -> torch.Tensor:
    """Fraction of each class on the correct side of zero, averaged."""
    return 0.5 * ((score_data > 0).float().mean() + (score_policy < 0).float().mean())


def _penalty(score: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    gradients = torch.autograd.grad(score.sum(), z, create_graph=True)[0]
    return (gradients.norm(dim=-1) - 1.0).square().mean()


def _interpolated(z_data: torch.Tensor, z_policy: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(z_data.shape[0], 1, device=z_data.device)
    mixed = alpha * z_data + (1.0 - alpha) * z_policy
    return mixed.detach().requires_grad_(True)


class _ZScore(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        linears = [module for module in self.net if isinstance(module, nn.Linear)]
        for linear in linears[:-1]:
            nn.init.orthogonal_(linear.weight, gain=gain)
            nn.init.zeros_(linear.bias)
        nn.init.orthogonal_(linears[-1].weight, gain=0.01)
        nn.init.zeros_(linears[-1].bias)

    def score(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

    def compute_loss(
        self, z_data: torch.Tensor, z_policy: torch.Tensor, grad_penalty: float
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def reward(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VanillaDiscriminator(_ZScore):
    """Binary classifier. Reward is ``log D - log(1-D)``, which is negative below 0.5."""

    def compute_loss(self, z_data, z_policy, grad_penalty):
        score_data = self.score(z_data)
        score_policy = self.score(z_policy)
        loss_data = -F.logsigmoid(score_data)
        loss_policy = F.softplus(score_policy)
        loss = (loss_data + loss_policy).mean()
        penalty = score_data.new_zeros(())
        if grad_penalty > 0.0:
            mixed = _interpolated(z_data, z_policy)
            penalty = _penalty(self.score(mixed), mixed)
            loss = loss + grad_penalty * penalty
        return {
            "loss": loss,
            "data": loss_data.mean(),
            "policy": loss_policy.mean(),
            "penalty": penalty,
            "acc": _accuracy(score_data, score_policy),
        }

    def reward(self, z: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        probability = torch.sigmoid(self.score(z)).clamp(eps, 1.0 - eps)
        return probability.log() - (1.0 - probability).log()


class LeastSquaresDiscriminator(_ZScore):
    """LSGAN. Dataset scores are pulled to +1 and policy scores to -1.

    The policy reward is the AMP clip ``max(0, 1 - ¼(D - 1)²)``, which stays
    in ``[0, 1]``.
    """

    def compute_loss(self, z_data, z_policy, grad_penalty):
        score_data = self.score(z_data)
        score_policy = self.score(z_policy)
        loss_data = (score_data - 1.0).square().mean()
        loss_policy = (score_policy + 1.0).square().mean()
        loss = 0.5 * (loss_data + loss_policy)
        penalty = score_data.new_zeros(())
        if grad_penalty > 0.0:
            real = z_data.detach().requires_grad_(True)
            gradients = torch.autograd.grad(self.score(real).sum(), real, create_graph=True)[0]
            penalty = gradients.square().sum(dim=-1).mean()
            loss = loss + grad_penalty * penalty
        return {
            "loss": loss,
            "data": loss_data,
            "policy": loss_policy,
            "penalty": penalty,
            "acc": _accuracy(score_data, score_policy),
        }

    def reward(self, z: torch.Tensor) -> torch.Tensor:
        return (1.0 - 0.25 * (self.score(z) - 1.0).square()).clamp_min(0.0)


class WassersteinDiscriminator(_ZScore):
    """WGAN-GP. The reward is the critic score and is not bounded below."""

    def compute_loss(self, z_data, z_policy, grad_penalty):
        score_data = self.score(z_data)
        score_policy = self.score(z_policy)
        loss = score_policy.mean() - score_data.mean()
        penalty = score_data.new_zeros(())
        if grad_penalty > 0.0:
            mixed = _interpolated(z_data, z_policy)
            penalty = _penalty(self.score(mixed), mixed)
            loss = loss + grad_penalty * penalty
        return {
            "loss": loss,
            "data": score_data.mean(),
            "policy": score_policy.mean(),
            "penalty": penalty,
            "acc": (score_data > score_policy).float().mean(),
        }

    def reward(self, z: torch.Tensor) -> torch.Tensor:
        return self.score(z)


def make_discriminator(name: str, z_dim: int, hidden_dim: int) -> _ZScore:
    classes = {
        "vanilla": VanillaDiscriminator,
        "lsgan": LeastSquaresDiscriminator,
        "wgan": WassersteinDiscriminator,
    }
    if name not in classes:
        raise ValueError(f"disc must be one of {sorted(classes)}, got {name!r}")
    return classes[name](z_dim, hidden_dim)
