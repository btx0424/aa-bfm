"""PPO with an AMP reward on ``amp_state``.

The task return is unchanged. This policy adds the discriminator reward on
the rollout's ``amp_state`` before the advantage. The default score is LSGAN:
dataset frames are pulled to +1 and policy frames to -1, and the reward is
``max(0, 1 - ¼(D - 1)²)``.

Expert rows use the same layout as the ``amp_state`` group: simulation-order
joint position and velocity, root angular velocity, projected gravity, then
ankle and wrist positions in the root frame.
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
from active_adaptation.utils.math import quat_rotate_inverse

from .adversarial import make_discriminator

cs = ConfigStore.instance()

AMP_STATE_KEY = "amp_state"
# ``find_bodies`` walks ``body_names_simulation``, so the regexes
# ``.*_ankle_roll_link`` then ``.*_wrist_roll_link`` resolve in this order.
_AMP_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
)

@dataclass
class AMPPPOConfig(PPOConfig):
    _target_: str = "aa_bfm_learning.ppo_amp.AMPPPOConfig"
    name: str = "ppo_amp"
    # Keep ``amp_state`` in the env. The actor still reads command and policy.
    in_keys: tuple[str, ...] = (CMD_KEY, OBS_KEY, "amp_state")
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
        if AMP_STATE_KEY not in observation_spec.keys(True, True):
            raise KeyError(
                "ppo_amp requires the amp_state observation. "
                "Keep it in algo.in_keys so the env does not drop it."
            )
        self.command_manager = env.command_manager
        from aa_bfm.commands import BFMCommand
        if not isinstance(self.command_manager, BFMCommand):
            raise TypeError("ppo_amp requires BFMCommand.")
        self.expert_amp = self._expert_amp().to(self.device)
        obs_dim = int(observation_spec[AMP_STATE_KEY].shape[-1])
        if int(self.expert_amp.shape[-1]) != obs_dim:
            raise ValueError(
                f"amp_state is {obs_dim}-d but the clip table is {self.expert_amp.shape[-1]}-d."
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
        state = tensordict[AMP_STATE_KEY]
        return state.reshape(-1, self.fb_dim)

    def _sample_expert(self, count: int) -> torch.Tensor:
        index = torch.randint(0, self.expert_amp.shape[0], (count,), device=self.expert_amp.device)
        return self.expert_amp[index]

    def _expert_amp(self) -> torch.Tensor:
        """Clip frames in the ``amp_state`` layout.

        Joints follow ``BFMCommand.fb_joint_names`` (simulation order, the same
        order as ``joint_pos`` / ``joint_vel`` with ``joint_names: ".*"``).
        Body positions are looked up by name because the archive is in MuJoCo
        body order.
        """
        command = self.command_manager
        columns = command.fb_joint_motion_index
        present = columns >= 0
        joint_pos = command.joint_pos.new_zeros(command.joint_pos.shape[0], columns.numel())
        joint_vel = command.joint_vel.new_zeros(command.joint_vel.shape[0], columns.numel())
        joint_pos[:, present] = command.joint_pos[:, columns[present]]
        joint_vel[:, present] = command.joint_vel[:, columns[present]]

        gravity_w = command.root_quat.new_tensor((0.0, 0.0, -1.0)).expand(command.root_quat.shape[0], 3)
        gravity_b = quat_rotate_inverse(command.root_quat, gravity_w)
        gravity_b = gravity_b / gravity_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ang_vel_b = quat_rotate_inverse(command.root_quat, command.root_ang_vel)

        name_to_index = {name: index for index, name in enumerate(command.motion_body_names)}
        missing = [name for name in _AMP_BODY_NAMES if name not in name_to_index]
        if missing:
            raise KeyError(f"amp_state bodies missing from the motion archive: {missing}")
        body_index = [name_to_index[name] for name in _AMP_BODY_NAMES]
        delta = command.body_pos[:, body_index] - command.root_pos[:, None, :]
        body_pos_b = quat_rotate_inverse(command.root_quat[:, None, :], delta)
        body_pos_b = body_pos_b.reshape(body_pos_b.shape[0], -1)
        return torch.cat((joint_pos, joint_vel, ang_vel_b, gravity_b, body_pos_b), dim=-1)

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
        state = tensordict[AMP_STATE_KEY]
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
