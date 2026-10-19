"""PPO with an AMP reward on ``fb_state``.

The task return is unchanged. This policy adds the discriminator reward on
the rollout's ``fb_state`` before the advantage. The default score is LSGAN:
dataset frames are pulled to +1 and policy frames to -1, and the reward is
``max(0, 1 - ¼(D - 1)²)``.

Expert rows are the clip table ``BFMCommand.fb_frames`` with joint velocity
and root linear velocity removed, so they match the ``fb_state`` group
(joint position, projected gravity, root angular velocity).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict, TensorDictBase
from torch.nn.parallel import DistributedDataParallel as DDP

import active_adaptation as aa
from active_adaptation.learning.modules import VecNorm
from active_adaptation.learning.ppo.common import CMD_KEY, OBS_KEY, REWARD_KEY
from active_adaptation.learning.ppo.ppo import PPOConfig, PPOPolicy

from .adversarial import make_discriminator

cs = ConfigStore.instance()

FB_KEY = "fb_state"


def expert_fb_state(frames: torch.Tensor) -> torch.Tensor:
    """``fb_frames`` rows in the ``fb_state`` observation layout.

    The table is joint position, joint velocity, gravity, root angular
    velocity, root linear velocity. The observation drops both velocities.
    """
    width = int(frames.shape[-1])
    if (width - 9) % 2 != 0:
        raise ValueError(f"fb_frames width {width} is not 2 * joints + 9.")
    joints = (width - 9) // 2
    joint_pos = frames[..., :joints]
    gravity = frames[..., 2 * joints : 2 * joints + 3]
    ang_vel = frames[..., 2 * joints + 3 : 2 * joints + 6]
    return torch.cat((joint_pos, gravity, ang_vel), dim=-1)


@dataclass
class AMPPPOConfig(PPOConfig):
    _target_: str = "aa_bfm_learning.ppo_amp.AMPPPOConfig"
    name: str = "ppo_amp"
    # Keep ``fb_state`` in the env. The actor still reads command and policy.
    in_keys: tuple[str, ...] = (CMD_KEY, OBS_KEY, FB_KEY)
    # ``vanilla`` | ``lsgan`` | ``wgan``.
    disc: str = "lsgan"
    disc_hidden_dim: int = 256
    lr_discriminator: float = 1.0e-4
    grad_penalty: float = 10.0
    # Passes over the rollout before the PPO epochs.
    disc_updates: int = 1
    # Weight of the style reward added to the task sum.
    amp_coef: float = 1.0

    def __post_init__(self):
        if self.disc_updates < 1:
            raise ValueError(f"disc_updates must be >= 1, got {self.disc_updates}.")
        if self.amp_coef < 0.0:
            raise ValueError(f"amp_coef must be >= 0, got {self.amp_coef}.")

    def get_class(self):
        return AMPPPO


cs.store("ppo_amp", node=AMPPPOConfig, group="algo")


class AMPPPO(PPOPolicy):
    def __init__(self, cfg: AMPPPOConfig, observation_spec, action_spec, reward_spec, device, env=None):
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        if env is None or not hasattr(env, "command_manager"):
            raise TypeError("ppo_amp requires the env command manager.")
        if FB_KEY not in observation_spec.keys(True, True):
            raise KeyError(
                "ppo_amp requires the fb_state observation. "
                "Keep it in algo.in_keys so the env does not drop it."
            )
        self.command_manager = env.command_manager
        if not hasattr(self.command_manager, "fb_frames"):
            raise TypeError("ppo_amp requires BFMCommand.fb_frames.")
        frames = expert_fb_state(self.command_manager.fb_frames[:1])
        obs_dim = int(observation_spec[FB_KEY].shape[-1])
        if int(frames.shape[-1]) != obs_dim:
            raise ValueError(
                f"fb_state is {obs_dim}-d but the clip table slice is {frames.shape[-1]}-d."
            )
        self.fb_dim = obs_dim
        self.vecnorm_amp = VecNorm(self.fb_dim, decay=1.0).to(self.device)
        self.D = make_discriminator(cfg.disc, self.fb_dim, cfg.disc_hidden_dim).to(self.device)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.lr_discriminator)
        if aa.is_distributed() and cfg.use_ddp:
            self.D = DDP(self.D, device_ids=[aa.get_local_cuda_index()])

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.to(self.device)
        disc = self._update_discriminator(tensordict)
        reward = self._amp_reward(tensordict)
        _add_reward(tensordict, reward)
        info = super().train_op(tensordict)
        info.update(disc)
        info["amp/reward"] = float(reward.mean())
        return dict(sorted(info.items()))

    def _flat_fb(self, tensordict: TensorDictBase) -> torch.Tensor:
        state = tensordict[FB_KEY]
        return state.reshape(-1, self.fb_dim)

    def _sample_expert(self, count: int) -> torch.Tensor:
        frames = self.command_manager.fb_frames
        index = torch.randint(0, frames.shape[0], (count,), device=frames.device)
        return expert_fb_state(frames[index]).to(self.device)

    def _normalize(self, policy: torch.Tensor, expert: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if aa.is_distributed():
            self.vecnorm_amp.synchronize(mode="broadcast")
        self.vecnorm_amp(torch.cat((policy, expert), dim=0))
        with VecNorm.freeze():
            return self.vecnorm_amp(policy), self.vecnorm_amp(expert)

    def _update_discriminator(self, tensordict: TensorDictBase) -> dict[str, float]:
        policy = self._flat_fb(tensordict)
        expert = self._sample_expert(policy.shape[0])
        policy_n, expert_n = self._normalize(policy, expert)
        disc = self.D.module if isinstance(self.D, DDP) else self.D
        last: dict[str, float] = {}
        for _ in range(int(self.cfg.disc_updates)):
            stats = disc.compute_loss(expert_n, policy_n.detach(), float(self.cfg.grad_penalty))
            self.opt_D.zero_grad(set_to_none=True)
            stats["loss"].backward()
            grad_norm = nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
            self.opt_D.step()
            last = {
                "amp/loss": float(stats["loss"].detach()),
                "amp/data": float(stats["data"].detach()),
                "amp/policy": float(stats["policy"].detach()),
                "amp/acc": float(stats["acc"].detach()),
                "amp/penalty": float(stats["penalty"].detach()),
                "amp/grad_norm": float(grad_norm),
            }
        return last

    @torch.no_grad()
    def _amp_reward(self, tensordict: TensorDictBase) -> torch.Tensor:
        state = tensordict[FB_KEY]
        flat = state.reshape(-1, self.fb_dim)
        with VecNorm.freeze():
            normalized = self.vecnorm_amp(flat)
        disc = self.D.module if isinstance(self.D, DDP) else self.D
        reward = disc.reward(normalized) * float(self.cfg.amp_coef)
        return reward.reshape(*state.shape[:-1], 1)


def _add_reward(tensordict: TensorDictBase, reward: torch.Tensor) -> None:
    groups = tensordict[REWARD_KEY]
    if isinstance(groups, TensorDictBase):
        groups["amp"] = reward
        return
    tensordict[REWARD_KEY] = groups + reward
