"""Critic factory ported from quadmanip's SAC RLPD.

Three losses, one ensemble trunk:

* ``c51`` — :class:`C51Critic`, distributional Bellman backup on a fixed atom grid.
* ``regression`` — :class:`RegressionCritic`, scalar heads, squared error.
* ``classification`` — :class:`ClassificationCritic`, scalar target encoded with
  HL-Gauss, then cross-entropy. See Farebrother et al., "Stop Regressing"
  (https://arxiv.org/abs/2403.03950).

``act_dim=0`` is a state value: the action tensor is empty and is dropped
by the concatenation in :class:`EnsembleCritic`.

``critic_outcome`` appends a one-hot ``(continue, terminated, truncated)`` to
the critic trunk only. Pass the logged transition's one-hot into the
regression. A missing one-hot is continue, which is what the bootstrap and
the actor query.
"""

from __future__ import annotations

import math
from typing import Callable

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from active_adaptation.learning.offpolicy.distributional import (
    cvar_from_logits,
    project_categorical_bellman,
)
from active_adaptation.learning.offpolicy.sac import CriticTrunk

CRITIC_KINDS = ("c51", "regression", "classification")

# continue, terminated, truncated. Terminated wins when both end flags are set.
OUTCOME_DIM = 3


def reward_group_keys(reward_spec) -> list[str]:
    """Ordered reward-group names under ``reward_spec['reward']``."""
    rew = reward_spec.zero()
    if isinstance(rew, TensorDict) and "reward" in rew.keys():
        rew = rew["reward"]
    if not isinstance(rew, TensorDict):
        raise ValueError(
            "group_critic requires a TensorDict reward spec under 'reward', "
            f"got {type(rew).__name__}"
        )
    keys = list(rew.keys())
    if len(keys) < 1:
        raise ValueError("group_critic found no reward groups")
    return keys


def collate_reward(
    rew: torch.Tensor | TensorDict,
    *,
    group_keys: list[str] | None,
    clamp: bool,
) -> torch.Tensor:
    """One column per group, or a single sum when ``group_keys`` is None."""
    if isinstance(rew, TensorDict):
        keys = list(rew.keys()) if group_keys is None else group_keys
        rew = torch.cat([rew[key].sum(dim=-1, keepdim=True) for key in keys], dim=-1)
    if group_keys is None:
        rew = rew.sum(dim=-1, keepdim=True)
    if clamp:
        rew = rew.clamp_min(0.0)
    return rew


def two_hot(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Project scalars onto a categorical support via adjacent-atom encoding."""
    if support.ndim != 1:
        raise ValueError(f"support must be 1-D, got shape {tuple(support.shape)}")
    num_atoms = support.shape[0]
    if num_atoms < 2:
        raise ValueError(f"support must contain at least 2 atoms, got {num_atoms}.")

    squeeze_last = values.ndim >= 2 and values.shape[-1] == 1
    flat = values.reshape(-1)
    batch = flat.shape[0]

    device = flat.device
    dtype = flat.dtype
    support = support.to(device=device, dtype=dtype)
    v_lo = support[0]
    v_hi = support[-1]
    delta_raw = (v_hi - v_lo) / (num_atoms - 1)
    min_delta = torch.finfo(dtype).tiny * torch.tensor(256.0, device=device, dtype=dtype)
    delta_z = torch.clamp(delta_raw, min=min_delta)

    target = flat.to(dtype=dtype).clamp(v_lo, v_hi)
    b = (target - v_lo) / delta_z
    b_max = float(num_atoms - 1)
    b = torch.nan_to_num(b, nan=0.0, neginf=0.0, posinf=b_max).clamp(0.0, b_max)

    lower = torch.floor(b).long().clamp(0, num_atoms - 1)
    upper = (lower + 1).clamp(max=num_atoms - 1)
    frac = b - lower.to(dtype=b.dtype)

    proj = flat.new_zeros(batch, num_atoms)
    proj.scatter_add_(1, lower.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    proj.scatter_add_(1, upper.unsqueeze(-1), frac.unsqueeze(-1))

    if squeeze_last:
        return proj.reshape(*values.shape[:-1], num_atoms)
    return proj.reshape(*values.shape, num_atoms)


def hl_gauss(
    values: torch.Tensor,
    support: torch.Tensor,
    *,
    sigma_ratio: float = 0.75,
) -> torch.Tensor:
    """Histogram-of-Gaussians encoding of a scalar onto ``support`` bin centers."""
    if support.ndim != 1 or support.shape[0] < 2:
        raise ValueError(f"support must be 1-D with at least 2 atoms, got {tuple(support.shape)}")
    squeeze_last = values.ndim >= 2 and values.shape[-1] == 1
    flat = values.reshape(-1, 1)
    z = support.to(device=flat.device, dtype=flat.dtype).view(1, -1)
    delta = (z[0, -1] - z[0, 0]) / (z.shape[-1] - 1)
    sigma = (sigma_ratio * delta).clamp_min(torch.finfo(flat.dtype).tiny * 256)
    center = flat.clamp(z[0, 0], z[0, -1])
    scale = sigma * math.sqrt(2.0)

    def cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / scale))

    probs = (cdf(z + 0.5 * delta - center) - cdf(z - 0.5 * delta - center)).clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if squeeze_last:
        return probs.reshape(*values.shape[:-1], z.shape[-1])
    return probs.reshape(*values.shape, z.shape[-1])


def window_outcome(
    terminated: torch.Tensor,
    done: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """One-hot at the same step :class:`MultiStepReturn` uses for the discount.

    ``terminated`` wins over ``done``. A flat ``[B]`` or ``[B, 1]`` flag is
    one transition. A leading time axis of length ``n_steps`` is the window.
    """
    time_major = terminated.dim() >= 3 or (
        n_steps > 1 and terminated.dim() == 2 and terminated.shape[0] == n_steps
    )
    if time_major:
        done_bool = done.bool()
        cum_not_done = (~done_bool).cumprod(dim=0)
        alive_steps = cum_not_done.sum(dim=0)
        last = alive_steps.clamp_max(n_steps - 1).reshape(-1).long()
        cols = torch.arange(last.shape[0], device=terminated.device)
        term_last = terminated[last, cols].reshape(-1).bool()
        done_last = done[last, cols].reshape(-1).bool()
    else:
        term_last = terminated.reshape(-1).bool()
        done_last = done.reshape(-1).bool()
    trunc = done_last & ~term_last
    cont = ~term_last & ~trunc
    out = torch.zeros(term_last.shape[0], OUTCOME_DIM, device=terminated.device, dtype=torch.float32)
    out[:, 0] = cont.to(dtype=out.dtype)
    out[:, 1] = term_last.to(dtype=out.dtype)
    out[:, 2] = trunc.to(dtype=out.dtype)
    return out


def _column(values: torch.Tensor) -> torch.Tensor:
    if values.ndim >= 1 and (values.ndim == 1 or values.shape[-1] != 1):
        return values.unsqueeze(-1)
    return values


def _td_target(
    ev: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bootstrap from the lowest ensemble head."""
    min_ev = ev.min(dim=-1, keepdim=True).values
    reward = reward.reshape_as(min_ev).to(dtype=min_ev.dtype)
    discount = discount.reshape_as(min_ev).to(dtype=min_ev.dtype)
    return reward + discount * min_ev, min_ev


def _group_bootstrap(
    ev: torch.Tensor,
    reward: torch.Tensor,
    discount: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Min over the ensemble. ``num_groups == 1`` keeps the scalar ``[B, 1]`` target."""
    if num_groups == 1:
        return _td_target(ev, reward, discount)
    min_ev = ev.min(dim=-2).values
    reward = reward.reshape(min_ev.shape[0], num_groups).to(dtype=min_ev.dtype)
    discount = discount.reshape(min_ev.shape[0], 1).expand_as(min_ev).to(dtype=min_ev.dtype)
    return reward + discount * min_ev, min_ev


class EnsembleCritic(nn.Module):
    """K independent trunks; concatenates heads on the last dim."""

    def __init__(
        self,
        fn: Callable[..., nn.Module],
        num_critics: int = 2,
        outcome_dim: int = 0,
    ):
        super().__init__()
        if num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {num_critics}")
        self.num_critics = int(num_critics)
        self.outcome_dim = int(outcome_dim)
        for i in range(self.num_critics):
            self.add_module(f"critic_{i + 1}", fn())

    def _iter_critics(self):
        for i in range(self.num_critics):
            yield getattr(self, f"critic_{i + 1}")

    def _outcome(self, obs: torch.Tensor, outcome: torch.Tensor | None) -> torch.Tensor | None:
        if self.outcome_dim == 0:
            return None
        if outcome is None:
            outcome = obs.new_zeros(*obs.shape[:-1], self.outcome_dim)
            outcome[..., 0] = 1
            return outcome
        if outcome.shape[-1] != self.outcome_dim:
            raise ValueError(
                f"outcome last dim must be {self.outcome_dim}, got shape {tuple(outcome.shape)}"
            )
        return outcome.to(dtype=obs.dtype)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        outcome: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outcome = self._outcome(obs, outcome)
        if act.dim() == 2:
            parts = [] if act.shape[-1] == 0 else [act]
            if outcome is not None:
                parts.append(outcome)
            inp = obs if not parts else torch.cat([obs, *parts], dim=-1)
            return torch.cat([c(inp) for c in self._iter_critics()], dim=-1)
        if act.dim() == 3:
            b, k, _ = act.shape
            obs_flat = einops.repeat(obs, "batch obs -> (batch k) obs", k=k)
            act_flat = einops.rearrange(act, "batch k act_dim -> (batch k) act_dim")
            outcome_flat = None
            if outcome is not None:
                outcome_flat = einops.repeat(outcome, "batch d -> (batch k) d", k=k)
            qs = self.forward(obs_flat, act_flat, outcome_flat)
            return einops.rearrange(qs, "(batch k) fused -> batch k fused", batch=b, k=k)
        raise ValueError(f"act must be rank 2 or 3, got shape {tuple(act.shape)}")


class EnsembleValue(nn.Module):
    """Ensemble forward. Subclasses own the backup and the loss."""

    def __init__(self, module: nn.Module, num_critics: int, num_groups: int = 1):
        super().__init__()
        if num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {num_critics}")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        self.module = module
        self.num_critics = int(num_critics)
        self.num_groups = int(num_groups)

    def sum_groups(self, values: torch.Tensor) -> torch.Tensor:
        """Sum reward-group values. The last dim is the group axis when there is more than one."""
        if self.num_groups == 1:
            return values
        return values.sum(dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        outcome: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.module(obs, act, outcome)

    def get_values(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        risk_alpha: torch.Tensor | None = None,
        *,
        outcome: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.expected_values(self(obs, act, outcome), risk_alpha)

    def expected_values(
        self,
        logits: torch.Tensor,
        risk_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def project(self, values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def support_saturation_stats(self, logits: torch.Tensor) -> dict[str, float]:
        raise NotImplementedError


class CategoricalCritic(EnsembleValue):
    """Atom logits, cross-entropy, and expected-value readout. Subclasses encode the target."""

    def __init__(
        self,
        module: nn.Module,
        *,
        v_min: float,
        v_max: float,
        num_atoms: int,
        num_critics: int,
        num_groups: int = 1,
    ):
        super().__init__(module, num_critics, num_groups)
        support = torch.linspace(v_min, v_max, int(num_atoms))
        self.register_buffer("q_support", support)
        self.q_support: torch.Tensor

    def _ensemble_atom_logits(self, logits: torch.Tensor) -> torch.Tensor:
        n_atom = self.q_support.shape[0]
        k = self.num_critics
        g = self.num_groups
        if g == 1:
            if logits.shape[-1] == k * n_atom:
                return logits.reshape(*logits.shape[:-1], k, n_atom)
            if logits.shape[-1] == k and logits.shape[-2] == n_atom:
                return logits.movedim(-1, -2)
            raise ValueError(
                f"Expected logits [..., {k * n_atom}] or [..., {n_atom}, {k}], "
                f"got shape {tuple(logits.shape)}."
            )
        flat = k * g * n_atom
        if logits.shape[-1] != flat:
            raise ValueError(
                f"Expected logits [..., {flat}] for {k} critics and {g} groups, "
                f"got shape {tuple(logits.shape)}."
            )
        return logits.reshape(*logits.shape[:-1], k, g, n_atom)

    def expected_values(
        self,
        logits: torch.Tensor,
        risk_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits_k = self._ensemble_atom_logits(logits)
        if self.num_groups == 1:
            if risk_alpha is not None:
                heads = [
                    cvar_from_logits(logits_k[..., i, :], self.q_support, risk_alpha)
                    for i in range(logits_k.shape[-2])
                ]
                return torch.cat(heads, dim=-1)
            p = F.softmax(logits_k, dim=-1)
            z = self.q_support.to(device=logits.device, dtype=logits.dtype).view(
                *([1] * (p.ndim - 1)), -1
            )
            return (p * z).sum(dim=-1)
        if risk_alpha is not None:
            heads = []
            for i in range(logits_k.shape[-3]):
                cv = cvar_from_logits(logits_k[..., i, :, :], self.q_support, risk_alpha)
                heads.append(cv.squeeze(-1))
            return torch.stack(heads, dim=-2)
        p = F.softmax(logits_k, dim=-1)
        z = self.q_support.to(device=logits.device, dtype=logits.dtype).view(
            *([1] * (p.ndim - 1)), -1
        )
        return (p * z).sum(dim=-1)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_k = self._ensemble_atom_logits(pred)
        target = target.to(dtype=logits_k.dtype)
        if self.num_groups == 1:
            target = target.unsqueeze(-2)
            log_p = F.log_softmax(logits_k, dim=-1).clamp(min=-30.0)
            return -(target * log_p).sum(dim=(-1, -2))
        target = target.unsqueeze(-3)
        log_p = F.log_softmax(logits_k, dim=-1).clamp(min=-30.0)
        return -(target * log_p).sum(dim=(-1, -2, -3))

    @torch.no_grad()
    def support_saturation_stats(self, logits: torch.Tensor) -> dict[str, float]:
        logits_k = self._ensemble_atom_logits(logits)
        p = F.softmax(logits_k, dim=-1)
        if self.num_groups > 1:
            p = p.mean(dim=-3)
        else:
            p = p.mean(dim=-2)
        z = self.q_support.to(device=p.device, dtype=p.dtype)
        v_min = float(z[0].item())
        v_max = float(z[-1].item())
        width = max(v_max - v_min, 1e-8)
        q = (p * z.view(*([1] * (p.ndim - 1)), -1)).sum(dim=-1)
        q_mean = q.mean()
        return {
            "critic/p_at_vmin": p[..., 0].mean().item(),
            "critic/p_at_vmax": p[..., -1].mean().item(),
            "critic/q_headroom_lo": ((q_mean - v_min) / width).item(),
            "critic/q_headroom_hi": ((v_max - q_mean) / width).item(),
        }


class C51Critic(CategoricalCritic):
    """Distributional Bellman backup. The target is the lowest-head next distribution."""

    def project(self, values: torch.Tensor) -> torch.Tensor:
        return two_hot(values, self.q_support)

    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits is None:
            logits = self(next_obs, next_act)
        logits_k = self._ensemble_atom_logits(logits)
        batch = logits_k.shape[0]
        if self.num_groups == 1:
            reward_b = reward.reshape(batch, 1)
            discount_b = discount.reshape(batch, 1)
        else:
            reward_b = reward.reshape(batch, 1, self.num_groups)
            discount_b = discount.reshape(batch, 1, 1)
        p = project_categorical_bellman(logits_k, reward_b, discount_b, self.q_support)
        z = self.q_support.to(device=logits.device, dtype=logits.dtype)
        ev = (p * z).sum(dim=-1)
        min_ev, idx = ev.min(dim=1)
        gather = idx.reshape(batch, 1, *idx.shape[1:], 1).expand_as(p[:, :1])
        target = p.gather(1, gather).squeeze(1)
        if self.num_groups == 1:
            return target, min_ev.unsqueeze(-1)
        return target, min_ev


class ClassificationCritic(CategoricalCritic):
    """Cross-entropy on an HL-Gauss encoding of the scalar TD target."""

    def __init__(
        self,
        module: nn.Module,
        *,
        v_min: float,
        v_max: float,
        num_atoms: int,
        num_critics: int,
        num_groups: int = 1,
        sigma_ratio: float = 0.75,
    ):
        super().__init__(
            module,
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
            num_critics=num_critics,
            num_groups=num_groups,
        )
        self.sigma_ratio = float(sigma_ratio)

    def project(self, values: torch.Tensor) -> torch.Tensor:
        return hl_gauss(values, self.q_support, sigma_ratio=self.sigma_ratio)

    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits is None:
            logits = self(next_obs, next_act)
        ev = self.expected_values(logits)
        y, min_ev = _group_bootstrap(ev, reward, discount, self.num_groups)
        return self.project(y), min_ev


class RegressionCritic(EnsembleValue):
    """Scalar heads. Squared error to the lowest-head bootstrap."""

    def project(self, values: torch.Tensor) -> torch.Tensor:
        if self.num_groups > 1:
            return values.reshape(values.shape[0], self.num_groups)
        return _column(values)

    def expected_values(
        self,
        logits: torch.Tensor,
        risk_alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del risk_alpha
        if self.num_groups == 1:
            return logits
        return logits.reshape(*logits.shape[:-1], self.num_critics, self.num_groups)

    def compute_target(
        self,
        next_obs: torch.Tensor,
        next_act: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits is None:
            logits = self(next_obs, next_act)
        ev = self.expected_values(logits)
        y, min_ev = _group_bootstrap(ev, reward, discount, self.num_groups)
        return self.project(y), min_ev

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.num_groups == 1:
            goal = target.reshape(pred.shape[0], -1).to(dtype=pred.dtype)
            if goal.shape[-1] == 1:
                goal = goal.expand_as(pred)
            return (pred - goal).square().sum(dim=-1)
        pred_g = pred.reshape(pred.shape[0], self.num_critics, self.num_groups)
        goal = target.reshape(pred.shape[0], self.num_groups).to(dtype=pred.dtype)
        goal = goal.unsqueeze(1).expand_as(pred_g)
        return (pred_g - goal).square().sum(dim=(-1, -2))

    def support_saturation_stats(self, logits: torch.Tensor) -> dict[str, float]:
        del logits
        return {}


def _ensemble(
    obs_dim: int,
    act_dim: int,
    output_dim: int,
    *,
    num_critics: int,
    hidden_dim: int,
    activation: type[nn.Module],
    outcome_dim: int,
    trunk: Callable[[int, int, int], nn.Module] | None = None,
) -> EnsembleCritic:
    in_dim = obs_dim + act_dim + outcome_dim

    def build() -> nn.Module:
        if trunk is None:
            return CriticTrunk(
                input_dim=in_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                activation=activation,
            )
        return trunk(in_dim, hidden_dim, output_dim)

    return EnsembleCritic(fn=build, num_critics=num_critics, outcome_dim=outcome_dim)


def make_critic(
    kind: str,
    obs_dim: int,
    act_dim: int,
    *,
    num_atoms: int = 101,
    v_min: float = -0.5,
    v_max: float = 5.0,
    num_critics: int = 2,
    hidden_dim: int = 512,
    activation: type[nn.Module] = nn.SiLU,
    sigma_ratio: float = 0.75,
    outcome: bool = False,
    num_groups: int = 1,
    trunk: Callable[[int, int, int], nn.Module] | None = None,
) -> EnsembleValue:
    """Build an ensemble critic. ``kind`` is ``c51``, ``regression``, or ``classification``."""
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    outcome_dim = OUTCOME_DIM if outcome else 0
    if kind == "c51":
        return C51Critic(
            _ensemble(
                obs_dim, act_dim, num_atoms * num_groups,
                num_critics=num_critics, hidden_dim=hidden_dim, activation=activation,
                outcome_dim=outcome_dim, trunk=trunk,
            ),
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
            num_critics=num_critics,
            num_groups=num_groups,
        )
    if kind == "regression":
        return RegressionCritic(
            _ensemble(
                obs_dim, act_dim, num_groups,
                num_critics=num_critics, hidden_dim=hidden_dim, activation=activation,
                outcome_dim=outcome_dim, trunk=trunk,
            ),
            num_critics,
            num_groups,
        )
    if kind == "classification":
        return ClassificationCritic(
            _ensemble(
                obs_dim, act_dim, num_atoms * num_groups,
                num_critics=num_critics, hidden_dim=hidden_dim, activation=activation,
                outcome_dim=outcome_dim, trunk=trunk,
            ),
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
            num_critics=num_critics,
            num_groups=num_groups,
            sigma_ratio=sigma_ratio,
        )
    raise ValueError(f"critic kind must be one of {CRITIC_KINDS}, got {kind!r}")
