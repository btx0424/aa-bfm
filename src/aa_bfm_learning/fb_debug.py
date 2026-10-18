"""Paired forward-backward with a task-reward critic driving the actor.

``F`` and ``B`` still train with the paired FB loss. The behavior policy does
not ascend ``F^T z``. It ascends a twin critic of the ``track`` reward group,
conditioned on the policy observation and the ``command`` reference. The
command is the lookahead the reward was computed from; ``z`` does not contain
root position or keypoints, so the critic has to see ``command`` itself.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.objectives import hold_out_net

import active_adaptation as aa
from active_adaptation.learning.modules import VecNorm
from active_adaptation.learning.ppo.common import ACTION_KEY, CMD_KEY, OBS_KEY, soft_copy_
from active_adaptation.learning.utils.distributed import unwrap_ddp, wrap_ddp

from .fb_paired import FBPaired, FBPairedConfig, FBRolloutPolicy, uncertainty_target

cs = ConfigStore.instance()


class TaskActor(nn.Module):
    def __init__(self, obs_dim: int, cmd_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + cmd_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(torch.cat([obs, command], dim=-1)))


class TaskCritic(nn.Module):
    def __init__(self, obs_dim: int, cmd_dim: int, action_dim: int, hidden_dim: int, num_parallel: int):
        super().__init__()
        self.members = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim + cmd_dim + action_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Mish(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_parallel)
            ]
        )

    def forward(self, obs: torch.Tensor, command: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([obs, command, action], dim=-1)
        values = [member(inputs).squeeze(-1) for member in self.members]
        return torch.stack(values, dim=0)


@dataclass
class FBDebugConfig(FBPairedConfig):
    _target_: str = "aa_bfm_learning.fb_debug.FBDebugConfig"
    name: str = "fb_debug"
    lr_q: float = 3.0e-4
    task_reward_group: str = "track"
    task_pessimism: float = 0.5

    def get_class(self):
        return FBDebug


cs.store(name="fb_debug", node=FBDebugConfig, group="algo")


class FBDebug(FBPaired):
    def __init__(
        self,
        cfg: FBDebugConfig,
        observation_spec,
        action_spec,
        reward_spec,
        device,
        *,
        command_manager,
    ):
        super().__init__(
            cfg=cfg,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            device=device,
            command_manager=command_manager,
        )
        if CMD_KEY not in observation_spec.keys(True, True):
            raise KeyError(f"fb_debug requires an observation group {CMD_KEY!r}.")
        reward_groups = list(reward_spec["reward"].keys())
        if cfg.task_reward_group not in reward_groups:
            raise KeyError(
                f"task reward group {cfg.task_reward_group!r} is not in the env rewards {reward_groups}."
            )
        fake = observation_spec.zero()
        self.cmd_dim = int(fake[CMD_KEY].shape[-1])
        self.vecnorm_cmd = VecNorm(self.cmd_dim).to(device)
        # The FB actor built above is not the behavior policy. Actions come from
        # the reference command so the task reward is a function of the inputs.
        self.actor = TaskActor(self.obs_dim, self.cmd_dim, self.act_dim, cfg.hidden_dim).to(device)
        self.Q = TaskCritic(
            self.obs_dim, self.cmd_dim, self.act_dim, cfg.hidden_dim, cfg.num_parallel
        ).to(device)
        self.Q_target = copy.deepcopy(self.Q).requires_grad_(False).eval()
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_Q = torch.optim.Adam(self.Q.parameters(), lr=cfg.lr_q)
        self._action_command: torch.Tensor | None = None

        if self._distributed:
            self.actor = wrap_ddp(self.actor)
            self.Q = wrap_ddp(self.Q)
            for module in (self.actor, self.Q, self.Q_target, self.vecnorm_cmd):
                for param in module.parameters():
                    torch.distributed.broadcast(param.data, src=0)
                for buf in module.buffers():
                    torch.distributed.broadcast(buf.data, src=0)

    def get_rollout_policy(self, mode: str = "train", critic: bool = False) -> TensorDictModuleBase:
        del critic
        policy = FBDebugRolloutPolicy(self)
        if mode == "eval":
            policy.eval_mode = True
        return policy

    @VecNorm.freeze()
    def train_op(self) -> dict[str, float]:
        if self._distributed:
            self.vecnorm_obs.synchronize(mode="broadcast")
            self.vecnorm_b.synchronize(mode="broadcast")
            self.vecnorm_cmd.synchronize(mode="broadcast")
        infos: dict[str, float] = {"rb_size": float(len(self.rb))}
        updates = self.cfg.train_every * self.cfg.utd_ratio
        last: dict[str, float] = {}
        for i in range(updates):
            batch = self.rb.sample(
                batch_size=self.cfg.fb_batch_size,
                steps=1,
                next_obs=True,
            ).to(self.device)
            if batch.batch_dims >= 2 and batch.batch_size[0] == 1:
                batch = batch[0]
            last = self._update_fb(batch)
            last.update(self._update_q(batch))
            last.update(self._update_actor(batch))
            if i == updates - 1:
                infos.update(last)
        soft_copy_(unwrap_ddp(self.F), self.F_target, tau=self.cfg.tau)
        soft_copy_(unwrap_ddp(self.B), self.B_target, tau=self.cfg.tau)
        soft_copy_(unwrap_ddp(self.Q), self.Q_target, tau=self.cfg.tau)
        infos["rb_size"] = float(len(self.rb))
        return infos

    def _normalize_cmd(self, command: torch.Tensor) -> torch.Tensor:
        return self.vecnorm_cmd(command)

    def _task_reward(self, batch: TensorDict) -> torch.Tensor:
        reward = batch["next", "reward", self.cfg.task_reward_group]
        return reward.sum(dim=-1)

    def _sample_action(self, state: torch.Tensor, z: torch.Tensor, mean: bool) -> torch.Tensor:
        del z
        if self._action_command is None or self._action_command.shape[0] != state.shape[0]:
            raise RuntimeError("Task actor requires the normalized command for this batch.")
        actor = self.actor if torch.is_grad_enabled() else unwrap_ddp(self.actor)
        loc = actor(state, self._action_command)
        if mean:
            return 5.0 * loc
        noise = torch.randn_like(loc) * self.cfg.actor_std
        noise = noise.clamp(-self.cfg.stddev_clip, self.cfg.stddev_clip)
        action = (loc + noise).clamp(-1.0 + 1.0e-6, 1.0 - 1.0e-6)
        return 5.0 * action

    def _update_fb(self, batch: TensorDict) -> dict[str, float]:
        # Bootstrap action is the behavior policy at the next command, not at z.
        self._action_command = self._normalize_cmd(batch["next", CMD_KEY]).detach()
        return super()._update_fb(batch)

    def _update_q(self, batch: TensorDict) -> dict[str, float]:
        obs = self._normalize_obs(batch[OBS_KEY])
        next_obs = self._normalize_obs(batch["next", OBS_KEY])
        command = self._normalize_cmd(batch[CMD_KEY])
        next_command = self._normalize_cmd(batch["next", CMD_KEY])
        action = batch[ACTION_KEY]
        reward = self._task_reward(batch)
        discount = self._discount(batch).squeeze(-1)
        with torch.no_grad():
            self._action_command = next_command.detach()
            next_action = self._sample_action(next_obs, batch["z"], mean=False)
            next_q = uncertainty_target(
                unwrap_ddp(self.Q_target)(next_obs, next_command, next_action),
                self.cfg.task_pessimism,
            )
            target = reward + discount * next_q
        q = self.Q(obs, command, action)
        # ``track`` is a sum of O(1) exponential terms. A true return is
        # O(reward / (1 - γ)). The logged loss is the raw MSE, RMSE = sqrt(loss).
        loss = F.mse_loss(q.float(), target.float().unsqueeze(0).expand_as(q))
        self.opt_Q.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.Q.parameters(), self.cfg.max_grad_norm)
        self.opt_Q.step()
        return {
            "task/q_loss": float(loss.detach()),
            "task/q_value": float(q.mean().detach()),
            "task/reward": float(reward.mean().detach()),
            "task/grad_norm": float(grad_norm),
        }

    def _update_actor(self, batch: TensorDict) -> dict[str, float]:
        obs = self._normalize_obs(batch[OBS_KEY])
        command = self._normalize_cmd(batch[CMD_KEY])
        self._action_command = command.detach()
        action = self._sample_action(obs, batch["z"], mean=False)
        with hold_out_net(self.Q):
            q = uncertainty_target(self.Q(obs, command, action), self.cfg.task_pessimism)
            loss = -q.mean()
            self.opt_actor.zero_grad(set_to_none=True)
            loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.opt_actor.step()
        return {
            "actor/loss": float(loss.detach()),
            "actor/q": float(q.mean().detach()),
            "actor/grad_norm": float(grad_norm),
        }

    def state_dict(self):
        state = super().state_dict()
        state["Q"] = unwrap_ddp(self.Q).state_dict()
        state["vecnorm_cmd"] = self.vecnorm_cmd.state_dict()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        unwrap_ddp(self.Q).load_state_dict(state_dict["Q"], strict=strict)
        self.vecnorm_cmd.load_state_dict(state_dict["vecnorm_cmd"])
        self.Q_target.load_state_dict(unwrap_ddp(self.Q).state_dict())


class FBDebugRolloutPolicy(FBRolloutPolicy):
    def forward(self, tensordict: TensorDict) -> TensorDict:
        algo = self.algo
        algo._action_command = algo.vecnorm_cmd(tensordict[CMD_KEY])
        return super().forward(tensordict)
