# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from humanoid.envs.custom.humanoid_env import XBotLFreeEnv

import torch


class H1FreeEnv(XBotLFreeEnv):
    """Unitree H1 leg-only feasibility environment.

    The first H1 slice intentionally reuses the existing humanoid locomotion
    logic and overrides only XBot-L-specific index assumptions.
    """

    def _joint_index(self, name: str) -> int:
        if not hasattr(self, "_joint_index_cache"):
            self._joint_index_cache = {joint_name: i for i, joint_name in enumerate(self.dof_names)}
        return self._joint_index_cache[name]

    def _joint_indices(self, names: list[str]) -> list[int]:
        return [self._joint_index(name) for name in names]

    def compute_ref_state(self):
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1

        sin_pos_l[sin_pos_l > 0] = 0
        for joint_name, scale in (
            ("left_hip_pitch_joint", scale_1),
            ("left_knee_joint", scale_2),
            ("left_ankle_joint", scale_1),
        ):
            self.ref_dof_pos[:, self._joint_index(joint_name)] = sin_pos_l * scale

        sin_pos_r[sin_pos_r < 0] = 0
        for joint_name, scale in (
            ("right_hip_pitch_joint", scale_1),
            ("right_knee_joint", scale_2),
            ("right_ankle_joint", scale_1),
        ):
            self.ref_dof_pos[:, self._joint_index(joint_name)] = sin_pos_r * scale

        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0
        self.ref_action = 2 * self.ref_dof_pos

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        n = self.num_actions
        noise_vec[0:5] = 0.0
        noise_vec[5 : 5 + n] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[5 + n : 5 + 2 * n] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[5 + 2 * n : 5 + 3 * n] = 0.0
        noise_vec[5 + 3 * n : 8 + 3 * n] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[8 + 3 * n : 11 + 3 * n] = noise_scales.quat * self.obs_scales.quat
        return noise_vec

    def _reward_default_joint_pos(self):
        joint_diff = self.dof_pos - self.default_joint_pd_target
        hip_yaw_roll = joint_diff[
            :,
            self._joint_indices(
                [
                    "left_hip_yaw_joint",
                    "left_hip_roll_joint",
                    "right_hip_yaw_joint",
                    "right_hip_roll_joint",
                ]
            ),
        ]
        yaw_roll = torch.norm(hip_yaw_roll, dim=1)
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
