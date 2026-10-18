"""One-frame kinematic state for the SMP prior.

The rollout buffer already stacks time, so this term does not keep a window.
``r_smp`` slices that buffer and passes the slice through :func:`smp_features`,
which anchors the window on its last frame the same way ``train_smp.py`` does.
"""

from __future__ import annotations

import torch
from typing_extensions import override

from active_adaptation.envs.mdp.observations.base import Observation
from active_adaptation.envs.utils.api import find_bodies
from active_adaptation.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
    yaw_quat,
)
from active_adaptation.utils.symmetry import (
    SymmetryTransform,
    cartesian_space_symmetry,
    joint_space_symmetry,
)

from .commands import BFMCommand

# Same bodies and order as ``scripts/train_smp.py``.
EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def smp_features(state: torch.Tensor, num_joints: int) -> torch.Tensor:
    """Yaw-anchor a window of :class:`smp_state` vectors.

    ``state`` is ``(..., window, state_dim)`` in time order, oldest first.
    The result is ``(..., window, 59)`` for the 29-DoF G1:
    root position, tan-norm root rotation, joint positions, end-effector
    positions, root linear velocity, root angular velocity.
    """
    num_ee = len(EE_BODY_NAMES)
    joint_end = 7 + num_joints
    ee_end = joint_end + 3 * num_ee
    if state.shape[-1] != ee_end + 6:
        raise ValueError(f"Expected state dim {ee_end + 6}, got {state.shape[-1]}")
    pos = state[..., 0:3]
    quat = state[..., 3:7]
    joints = state[..., 7:joint_end]
    ee = state[..., joint_end:ee_end].reshape(*state.shape[:-1], num_ee, 3)
    lin_vel = state[..., ee_end : ee_end + 3]
    ang_vel = state[..., ee_end + 3 : ee_end + 6]

    anchor_pos = pos[..., -1, :]
    yaw = yaw_quat(quat[..., -1, :])
    yaw_inv = quat_conjugate(yaw)
    window = pos.shape[-2]
    yaw_w = yaw.unsqueeze(-2).expand(*pos.shape[:-2], window, 4).reshape(-1, 4)
    yaw_inv_w = yaw_inv.unsqueeze(-2).expand(*pos.shape[:-2], window, 4).reshape(-1, 4)

    root_local = quat_rotate_inverse(yaw_w, (pos - anchor_pos.unsqueeze(-2)).reshape(-1, 3))
    root_local = root_local.reshape(*pos.shape).clone()
    root_local[..., 2] = pos[..., 2]
    root_rot = _tan_norm(quat_mul(yaw_inv_w, quat.reshape(-1, 4)).reshape(*quat.shape))

    ee_delta = ee - pos.unsqueeze(-2)
    yaw_ee = yaw.reshape(*yaw.shape[:-1], 1, 1, 4).expand(*ee.shape[:-1], 4).reshape(-1, 4)
    ee_local = quat_rotate_inverse(yaw_ee, ee_delta.reshape(-1, 3)).reshape(*ee.shape[:-2], -1)
    lin_local = quat_rotate_inverse(yaw_w, lin_vel.reshape(-1, 3)).reshape(*lin_vel.shape)
    ang_local = quat_rotate_inverse(yaw_w, ang_vel.reshape(-1, 3)).reshape(*ang_vel.shape)
    return torch.cat((root_local, root_rot, joints, ee_local, lin_local, ang_local), dim=-1)


def _tan_norm(quat: torch.Tensor) -> torch.Tensor:
    """``[x-column, z-column]`` of the rotation matrix."""
    mat = matrix_from_quat(quat)
    return torch.cat((mat[..., :, 0], mat[..., :, 2]), dim=-1)


class fb_prompt(Observation):
    """Indices into ``BFMCommand.fb_frames`` for the lookahead window.

    The frames themselves stay on the command. Replay stores ``H`` integers.
    """

    @override
    def _initialize(self, env) -> None:
        super()._initialize(env)
        command = self.command_manager
        if not isinstance(command, BFMCommand):
            raise TypeError("fb_prompt requires BFMCommand")
        self.command = command
        self.horizon = int(command.num_lookahead)
        frame_dim = int(command.fb_frames.shape[-1])
        expected = 2 * len(command.fb_joint_names) + 9
        if frame_dim != expected:
            raise ValueError(
                f"fb_frames width is {frame_dim}, expected {expected} "
                f"({len(command.fb_joint_names)} simulation-order joints, twice, plus 9)"
            )

    @override
    def compute(self) -> torch.Tensor:
        return self.command.ref_fb_index

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        """Indices are not a spatial feature. Leave the lookahead slots in place."""
        return SymmetryTransform(perm=torch.arange(self.horizon), signs=torch.ones(self.horizon))


class smp_state(Observation):
    """Current G1 kinematics, one frame, episode-origin relative.

    Layout, matching the fields ``smp_features`` reads back:

    ``root_pos(3), root_quat(4, wxyz), joint_pos(29), ee_pos(15),
    root_lin_vel_w(3), root_ang_vel_w(3)``.

    Joint order is the LAFAN CSV order used to train the prior. End effectors
    are ankles, torso, and wrist yaw, in that order. Positions have
    ``episode_origin`` removed. Velocities stay in the world frame. The prior
    does not see joint velocity.
    """

    @override
    def _initialize(self, env) -> None:
        super()._initialize(env)
        command = self.command_manager
        if not isinstance(command, BFMCommand):
            raise TypeError("smp_state requires BFMCommand")
        self.asset = command.asset
        self.joint_index = command.asset_joint_index
        self.joint_names = list(command.joint_names)
        found, found_names = find_bodies(self.asset, list(EE_BODY_NAMES))
        name_to_index = dict(zip(found_names, found))
        missing = [name for name in EE_BODY_NAMES if name not in name_to_index]
        if missing:
            raise KeyError(f"SMP end-effectors missing from the robot: {missing}")
        self.ee_names = list(EE_BODY_NAMES)
        self.ee_index = torch.tensor(
            [name_to_index[name] for name in self.ee_names],
            dtype=torch.long,
            device=self.device,
        )
        joint_count = int(self.asset.data.joint_pos.shape[-1])
        body_count = int(self.asset.data.body_link_pos_w.shape[1])
        if int(self.joint_index.max()) >= joint_count:
            raise IndexError(
                f"SMP joint index {int(self.joint_index.max())} is outside joint_pos ({joint_count})"
            )
        if int(self.ee_index.max()) >= body_count:
            raise IndexError(
                f"SMP body index {int(self.ee_index.max())} is outside body_link_pos_w ({body_count})"
            )

    @override
    def compute(self) -> torch.Tensor:
        origin = self.env.episode_origin
        root_pos = self.asset.data.root_link_pos_w - origin
        root_quat = self.asset.data.root_link_quat_w
        joint_pos = self.asset.data.joint_pos[:, self.joint_index]
        ee_pos = self.asset.data.body_link_pos_w[:, self.ee_index] - origin[:, None, :]
        lin_vel = self.asset.data.root_link_lin_vel_w
        ang_vel = self.asset.data.root_link_ang_vel_w
        return torch.cat(
            (
                root_pos,
                root_quat,
                joint_pos,
                ee_pos.reshape(self.num_envs, -1),
                lin_vel,
                ang_vel,
            ),
            dim=-1,
        )

    @override
    def symmetry_transform(self) -> SymmetryTransform:
        """Sagittal mirror of the raw frame, before the yaw anchor."""
        quat = SymmetryTransform(
            perm=torch.arange(4),
            signs=torch.tensor([1.0, -1.0, 1.0, -1.0]),
        )
        return SymmetryTransform.cat(
            [
                SymmetryTransform.xyz(),
                quat,
                joint_space_symmetry(self.asset, self.joint_names),
                cartesian_space_symmetry(self.asset, self.ee_names, sign=(1, -1, 1)),
                SymmetryTransform.xyz(),
                SymmetryTransform.rpy(),
            ]
        )
