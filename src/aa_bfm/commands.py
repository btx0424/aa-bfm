"""Paired LAFAN1 reference command for supervised FB training.

Each episode is one clip. The robot starts at frame 0, the episode ends at the
last frame, and the command exposes an ``H``-step lookahead of that same clip.
"""

from __future__ import annotations

import re

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override

from active_adaptation.envs.mdp.commands.base import Command
from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    rot6d,
)
from active_adaptation.utils.symmetry import SymmetryTransform, joint_space_symmetry

from .data import TARGET_FPS, Dataloader


class BFMCommand(Command):
    """Track one LAFAN1 clip from frame 0, with an H-step reference lookahead."""

    out_keys = ["crash_active", "fall_over_active"]

    def __init__(
        self,
        horizon: int = 8,
        data_path: str | None = None,
        reward_std: float = 0.5,
        fall_xy_thres: float = 0.8,
        ground_height: float = 0.4,
        clip_regex: list[str] | None = None,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be positive, got {horizon}")
        self.horizon = int(horizon)
        self.data_path = data_path
        self.reward_std = float(reward_std)
        self.fall_xy_thres = float(fall_xy_thres)
        self.ground_height = float(ground_height)
        self.clip_regex = list(clip_regex) if clip_regex else None

    @override
    def _initialize(self, env) -> None:
        super()._initialize(env)
        if abs(float(self.env.step_dt) - 1.0 / TARGET_FPS) > 1.0e-4:
            raise ValueError(
                f"BFMCommand motions are {TARGET_FPS} Hz, but step_dt is {self.env.step_dt}"
            )
        archive = Dataloader(self.data_path).ensure()
        motions = np.load(archive, allow_pickle=False)
        joint_names = [str(name) for name in motions["joint_names"].tolist()]
        name_to_index = {name: index for index, name in enumerate(self.asset.joint_names)}
        missing = [name for name in joint_names if name not in name_to_index]
        if missing:
            raise KeyError(f"Clip joints missing from g1_waist_unlocked: {missing}")
        self.asset_joint_index = torch.tensor(
            [name_to_index[name] for name in joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.joint_names = joint_names
        clip_names = [str(name) for name in motions["clip_names"].tolist()]
        keep = self._select_clip_ids(clip_names)
        starts = motions["starts"]
        ends = motions["ends"]
        self.clip_names = [clip_names[i] for i in keep]

        def take(field: str) -> torch.Tensor:
            parts = [motions[field][int(starts[i]) : int(ends[i])] for i in keep]
            return torch.as_tensor(np.concatenate(parts, axis=0), device=self.device)

        lengths = [int(ends[i] - starts[i]) for i in keep]
        if min(lengths) < self.horizon:
            raise ValueError(f"Every selected clip must be at least horizon={self.horizon} frames")
        cursor = 0
        packed_starts = []
        packed_ends = []
        for length in lengths:
            packed_starts.append(cursor)
            cursor += length
            packed_ends.append(cursor)
        self.starts = torch.tensor(packed_starts, dtype=torch.long, device=self.device)
        self.ends = torch.tensor(packed_ends, dtype=torch.long, device=self.device)
        self.lengths = self.ends - self.starts
        self.root_pos = take("root_pos")
        self.root_quat = take("root_quat")
        self.root_lin_vel = take("root_lin_vel")
        self.root_ang_vel = take("root_ang_vel")
        self.joint_pos = take("joint_pos")
        self.joint_vel = take("joint_vel")
        self._clip_flags()

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.t = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ref_joint_pos = torch.zeros(
            self.num_envs, self.horizon, len(joint_names), device=self.device
        )
        self.ref_root_pos = torch.zeros(self.num_envs, self.horizon, 3, device=self.device)
        self.ref_root_quat = torch.zeros(self.num_envs, self.horizon, 4, device=self.device)
        self.ref_root_rot6d_b = torch.zeros(self.num_envs, self.horizon, 6, device=self.device)
        self.track_reward = torch.zeros(self.num_envs, 1, device=self.device)
        self._lookahead = torch.arange(self.horizon, device=self.device)
        self.ref_frame_marker = None
        if self.env.sim.has_gui():
            self.ref_frame_marker = self.env.scene.create_frame_marker(
                "/Visuals/BFM/ref_root",
                scale=(0.2, 0.2, 0.2),
            )

    def _select_clip_ids(self, clip_names: list[str]) -> list[int]:
        if not self.clip_regex:
            return list(range(len(clip_names)))
        patterns = [re.compile(pattern) for pattern in self.clip_regex]
        keep = [
            index
            for index, name in enumerate(clip_names)
            if any(pattern.search(name) for pattern in patterns)
        ]
        if not keep:
            raise ValueError(f"No clips matched {self.clip_regex}. Available: {clip_names}")
        return keep

    def _clip_flags(self) -> None:
        """Clips whose reference falls or drops to the ground skip those terminations."""
        gravity = self.root_quat.new_tensor([0.0, 0.0, -1.0]).expand(self.root_quat.shape[0], 3)
        gravity_xy = quat_rotate_inverse(self.root_quat, gravity)[:, :2].norm(dim=-1)
        num_clips = int(self.lengths.numel())
        self.clip_falls = torch.zeros(num_clips, dtype=torch.bool, device=self.device)
        self.clip_ground = torch.zeros(num_clips, dtype=torch.bool, device=self.device)
        for clip_id in range(num_clips):
            frames = slice(int(self.starts[clip_id]), int(self.ends[clip_id]))
            self.clip_falls[clip_id] = gravity_xy[frames].amax() >= self.fall_xy_thres
            self.clip_ground[clip_id] = self.root_pos[frames, 2].amin() <= self.ground_height
        for name, falls, ground in zip(
            self.clip_names, self.clip_falls.tolist(), self.clip_ground.tolist()
        ):
            print(f"{name}: fall={falls} ground={ground}", flush=True)
        self.clip_ground = self.clip_ground | self.clip_falls

    @override
    def reset(self, env_ids: torch.Tensor, tensordict: TensorDictBase) -> torch.Tensor:
        env_ids = env_ids.to(self.device)
        count = env_ids.numel()
        self.motion_ids[env_ids] = torch.randint(0, int(self.lengths.numel()), (count,), device=self.device)
        self.t[env_ids] = 0
        max_len = self.env.max_episode_length.clone()
        max_len[env_ids] = self.lengths[self.motion_ids[env_ids]].unsqueeze(-1)
        self.env.max_episode_length = max_len

        frame = self.starts[self.motion_ids[env_ids]]
        origins = self.env.scene.sample_spawn_origin_candidates(env_ids)
        root_state = self.init_root_state[env_ids].clone()
        root_state[:, 0:3] = origins + self.root_pos[frame]
        root_state[:, 3:7] = self.root_quat[frame]
        root_state[:, 7:10] = self.root_lin_vel[frame]
        root_state[:, 10:13] = self.root_ang_vel[frame]
        self._write_initial_states({"robot": root_state}, env_ids)

        joint_pos = self.init_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.asset_joint_index] = self.joint_pos[frame]
        joint_vel[:, self.asset_joint_index] = self.joint_vel[frame]
        self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._load_lookahead()
        return origins

    def _gather(self, field: torch.Tensor) -> torch.Tensor:
        local = self.t.unsqueeze(1) + self._lookahead
        limit = self.lengths[self.motion_ids].unsqueeze(1) - 1
        local = torch.minimum(local, limit)
        return field[self.starts[self.motion_ids].unsqueeze(1) + local]

    def _load_lookahead(self) -> None:
        self.ref_joint_pos = self._gather(self.joint_pos)
        self.ref_root_pos = self._gather(self.root_pos)
        self.ref_root_quat = self._gather(self.root_quat)
        current = self.asset.data.root_link_quat_w.unsqueeze(1)
        relative = quat_mul(quat_conjugate(current), self.ref_root_quat)
        self.ref_root_rot6d_b = rot6d(relative)

    def _refresh_track_reward(self) -> None:
        current = self.asset.data.joint_pos[:, self.asset_joint_index]
        error = (current - self.ref_joint_pos[:, 0]).square().mean(dim=-1, keepdim=True)
        self.track_reward = torch.exp(-error / self.reward_std**2)

    def _termination_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        clip_ids = self.motion_ids
        crash_active = ~self.clip_ground[clip_ids]
        fall_over_active = ~self.clip_falls[clip_ids]
        # print(f"crash_active: {crash_active}, fall_over_active: {fall_over_active}")
        return crash_active.unsqueeze(-1), fall_over_active.unsqueeze(-1)

    @override
    def _update(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._load_lookahead()
        self._refresh_track_reward()
        # return self._termination_masks()
        return torch.ones(self.num_envs, 1, device=self.device, dtype=torch.bool), torch.ones(self.num_envs, 1, device=self.device, dtype=torch.bool)

    @override
    def step(self) -> None:
        self.t = torch.minimum(self.t + 1, self.lengths[self.motion_ids] - 1)
        self._load_lookahead()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            [
                self.ref_joint_pos.reshape(self.num_envs, -1),
                self.ref_root_rot6d_b.reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

    def symmetry_transform(self) -> SymmetryTransform:
        """Mirror future joint targets and the body-frame reference orientation."""
        joints = joint_space_symmetry(self.asset, self.joint_names).repeat(self.horizon)
        orientation = SymmetryTransform.rot6d().repeat(self.horizon)
        return SymmetryTransform.cat([joints, orientation])

    @override
    def get_state(self) -> TensorDict:
        return TensorDict(
            {
                "ref_joint_pos": self.ref_joint_pos,
                "ref_root_pos": self.ref_root_pos,
                "ref_root_quat": self.ref_root_quat,
                "ref_root_rot6d_b": self.ref_root_rot6d_b,
                "track_reward": self.track_reward,
                "motion_id": self.motion_ids,
                "frame": self.t,
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    @override
    def command_state_spec(self) -> dict[str, tuple[int, ...]]:
        joint_count = len(self.joint_names)
        return {
            "ref_joint_pos": (self.horizon, joint_count),
            "ref_root_pos": (self.horizon, 3),
            "ref_root_quat": (self.horizon, 4),
            "ref_root_rot6d_b": (self.horizon, 6),
            "track_reward": (1,),
            "motion_id": (),
            "frame": (),
        }

    @override
    def from_state(self, command_state: TensorDictBase, key: str | None = None) -> torch.Tensor:
        del key
        joints = command_state["ref_joint_pos"].reshape(command_state.shape[0], -1)
        orientation = command_state["ref_root_rot6d_b"].reshape(command_state.shape[0], -1)
        return torch.cat([joints, orientation], dim=-1)

    def debug_draw(self) -> None:
        if self.ref_frame_marker is None:
            return
        self.ref_frame_marker.visualize(
            translations=self.ref_root_pos[:, 0] + self.env.episode_origin,
            orientations=self.ref_root_quat[:, 0],
        )


class tracking_joint_exp(Reward):
    """Exponential joint-position match against the current reference frame."""

    def __init__(self, weight: float, std: float | None = None):
        super().__init__(weight)
        self.std = None if std is None else float(std)

    @override
    def _compute(self) -> torch.Tensor:
        command = self.command_manager
        if not isinstance(command, BFMCommand):
            raise TypeError("tracking_joint_exp requires BFMCommand")
        if self.std is None or abs(self.std - command.reward_std) < 1.0e-8:
            return command.track_reward
        current = command.asset.data.joint_pos[:, command.asset_joint_index]
        error = (current - command.ref_joint_pos[:, 0]).square().mean(dim=-1, keepdim=True)
        return torch.exp(-error / self.std**2)


class tracking_root_pos_exp(Reward):
    """Exponential root-position match against the current reference frame."""

    def __init__(self, weight: float, std: float = 0.3):
        super().__init__(weight)
        self.std = float(std)

    @override
    def _compute(self) -> torch.Tensor:
        command = self.command_manager
        if not isinstance(command, BFMCommand):
            raise TypeError("tracking_root_pos_exp requires BFMCommand")
        current = command.asset.data.root_link_pos_w
        target = command.env.episode_origin + command.ref_root_pos[:, 0]
        error = (current - target).square().sum(dim=-1, keepdim=True)
        return torch.exp(-error / self.std**2)


class tracking_root_ori_exp(Reward):
    """Exponential root-orientation match against the current reference frame."""

    def __init__(self, weight: float, std: float = 0.4):
        super().__init__(weight)
        self.std = float(std)

    @override
    def _compute(self) -> torch.Tensor:
        command = self.command_manager
        if not isinstance(command, BFMCommand):
            raise TypeError("tracking_root_ori_exp requires BFMCommand")
        current = command.asset.data.root_link_quat_w
        target = command.ref_root_quat[:, 0]
        delta = quat_mul(quat_conjugate(target), current)
        error = axis_angle_from_quat(delta).square().sum(dim=-1, keepdim=True)
        return torch.exp(-error / self.std**2)
