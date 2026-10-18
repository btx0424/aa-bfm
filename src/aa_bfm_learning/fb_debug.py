"""Paired forward-backward with a task-reward critic driving the actor.

``F`` and ``B`` still train with the one-step paired FB loss. The behavior
policy does not ascend ``F^T z``. It ascends a twin critic of the sum of
every enabled reward group, conditioned on the policy observation and the
``command`` reference. The command is the lookahead the tracking reward was
computed from; ``z`` does not contain root position or keypoints, so the
critic has to see ``command`` itself. The critic target is an n-step return
of that sum. The bootstrap action comes from a target actor, Polyak-averaged
at ``tau_actor``, evaluated at the command of the state ``n`` steps ahead.
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
from active_adaptation.learning.offpolicy.objectives import MultiStepReturn
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    CMD_KEY,
    DONE_KEY,
    OBS_KEY,
    TERM_KEY,
    soft_copy_,
)
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
    # Critic only. ``F`` and ``B`` stay one-step. ``1`` is the previous backup.
    n_steps: int = 3
    task_pessimism: float = 0.5
    # Lag of the actor used only for the Q bootstrap.
    tau_actor: float = 0.01

    def __post_init__(self):
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}.")
        if not 0.0 < self.tau_actor <= 1.0:
            raise ValueError(f"tau_actor must be in (0, 1], got {self.tau_actor}.")

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
        # Disabled groups are omitted from the spec, so these are the enabled ones.
        self.reward_group_keys = list(reward_spec["reward"].keys())
        if not self.reward_group_keys:
            raise KeyError("fb_debug requires at least one enabled reward group.")
        fake = observation_spec.zero()
        self.cmd_dim = int(fake[CMD_KEY].shape[-1])
        self.vecnorm_cmd = VecNorm(self.cmd_dim).to(device)
        # The FB actor built above is not the behavior policy. Actions come from
        # the reference command so the task reward is a function of the inputs.
        self.actor = TaskActor(self.obs_dim, self.cmd_dim, self.act_dim, cfg.hidden_dim).to(device)
        self.Q = TaskCritic(
            self.obs_dim, self.cmd_dim, self.act_dim, cfg.hidden_dim, cfg.num_parallel
        ).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.Q_target = copy.deepcopy(self.Q).requires_grad_(False).eval()
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_Q = torch.optim.Adam(self.Q.parameters(), lr=cfg.lr_q)
        self.msr = (
            MultiStepReturn(cfg.gamma, cfg.n_steps).to(device) if cfg.n_steps > 1 else None
        )
        self._action_command: torch.Tensor | None = None

        if self._distributed:
            self.actor = wrap_ddp(self.actor)
            self.Q = wrap_ddp(self.Q)
            for module in (self.actor, self.actor_target, self.Q, self.Q_target, self.vecnorm_cmd):
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
            window = self.rb.sample(
                batch_size=self.cfg.fb_batch_size,
                steps=self.cfg.n_steps,
                next_obs=True,
            ).to(self.device)
            # ``fake_bootstrap`` always returns a time axis. ``F``, ``B``, and
            # the actor stay on the first transition of the window.
            step = window[0] if window.batch_dims >= 2 else window
            last = self._update_fb(step)
            last.update(self._update_q(window if self.cfg.n_steps > 1 else step))
            last.update(self._update_actor(step))
            if i == updates - 1:
                infos.update(last)
        soft_copy_(unwrap_ddp(self.F), self.F_target, tau=self.cfg.tau)
        soft_copy_(unwrap_ddp(self.B), self.B_target, tau=self.cfg.tau)
        soft_copy_(unwrap_ddp(self.actor), self.actor_target, tau=self.cfg.tau_actor)
        soft_copy_(unwrap_ddp(self.Q), self.Q_target, tau=self.cfg.tau)
        infos["rb_size"] = float(len(self.rb))
        return infos

    def _normalize_cmd(self, command: torch.Tensor) -> torch.Tensor:
        return self.vecnorm_cmd(command)

    def _task_reward(self, batch: TensorDict) -> torch.Tensor:
        reward = batch["next", "reward"]
        total = reward[self.reward_group_keys[0]].sum(dim=-1)
        for key in self.reward_group_keys[1:]:
            total = total + reward[key].sum(dim=-1)
        return total

    def _sample_action(
        self, state: torch.Tensor, z: torch.Tensor, mean: bool, *, target: bool = False
    ) -> torch.Tensor:
        del z
        if self._action_command is None or self._action_command.shape[0] != state.shape[0]:
            raise RuntimeError("Task actor requires the normalized command for this batch.")
        # The target actor is the Q bootstrap only. Rollout, the actor update,
        # and the forward-backward backup stay on the live policy.
        if target:
            actor = self.actor_target
        else:
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

    def _q_backup(self, batch: TensorDict) -> tuple[torch.Tensor, ...]:
        """``(s, command, a)`` at ``t``, and the bootstrap at ``t+n``.

        ``n == 1`` is ``r_t + γ_t Q(s_{t+1}, command_{t+1}, a')``. Longer
        windows use :class:`MultiStepReturn`: the reward is the discounted
        sum along the window, and the bootstrap state is the last step that
        is still in the episode. The command is carried along with that
        state, because the critic is conditioned on the lookahead.
        """
        if self.cfg.n_steps == 1:
            obs = self._normalize_obs(batch[OBS_KEY])
            command = self._normalize_cmd(batch[CMD_KEY])
            action = batch[ACTION_KEY]
            next_obs = self._normalize_obs(batch["next", OBS_KEY])
            next_command = self._normalize_cmd(batch["next", CMD_KEY])
            reward = self._task_reward(batch)
            discount = self._discount(batch).squeeze(-1)
            return obs, command, action, next_obs, next_command, reward, discount, reward

        assert self.msr is not None
        obs = self._normalize_obs(batch[OBS_KEY][0])
        command = self._normalize_cmd(batch[CMD_KEY][0])
        action = batch[ACTION_KEY][0]
        next_obs = self._normalize_obs(batch["next", OBS_KEY])
        next_command = self._normalize_cmd(batch["next", CMD_KEY])
        # Index the bootstrap command with the same alive-step as ``next_obs``.
        next_feat = torch.cat([next_obs, next_command], dim=-1)
        reward = self._task_reward(batch).unsqueeze(-1)
        _, next_feat, ret, discount, _terminated = self.msr(
            actions=batch[ACTION_KEY],
            next_observations=next_feat,
            rewards=reward,
            terminated=batch[TERM_KEY],
            done=batch[DONE_KEY],
            env_discount=batch["next", "discount"],
        )
        next_obs, next_command = next_feat.split([self.obs_dim, self.cmd_dim], dim=-1)
        return (
            obs,
            command,
            action,
            next_obs,
            next_command,
            ret.squeeze(-1),
            discount.squeeze(-1),
            reward[0].squeeze(-1),
        )

    def _update_q(self, batch: TensorDict) -> dict[str, float]:
        obs, command, action, next_obs, next_command, reward, discount, immediate = self._q_backup(
            batch
        )
        with torch.no_grad():
            self._action_command = next_command.detach()
            # The task actor ignores ``z`` and reads ``_action_command``.
            next_action = self._sample_action(next_obs, command, mean=False, target=True)
            next_q = uncertainty_target(
                unwrap_ddp(self.Q_target)(next_obs, next_command, next_action),
                self.cfg.task_pessimism,
            )
            target = reward + discount * next_q
        q = self.Q(obs, command, action)
        # The target is the sum of the enabled groups. ``track`` is O(1) per
        # term; ``reg`` is a small penalty. A true return is O(reward / (1 - γ)).
        # The logged loss is the raw MSE, RMSE = sqrt(loss).
        loss = F.mse_loss(q.float(), target.float().unsqueeze(0).expand_as(q))
        self.opt_Q.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.Q.parameters(), self.cfg.max_grad_norm)
        self.opt_Q.step()
        return {
            "task/q_loss": float(loss.detach()),
            "task/q_value": float(q.mean().detach()),
            "task/reward": float(immediate.mean().detach()),
            "task/return": float(reward.mean().detach()),
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
        self.actor_target.load_state_dict(unwrap_ddp(self.actor).state_dict())


class FBDebugRolloutPolicy(FBRolloutPolicy):
    def forward(self, tensordict: TensorDict) -> TensorDict:
        algo = self.algo
        algo._action_command = algo.vecnorm_cmd(tensordict[CMD_KEY])
        return super().forward(tensordict)
