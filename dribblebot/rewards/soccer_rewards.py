import torch
import numpy as np
from dribblebot.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from .rewards import Rewards

class SoccerRewards(Rewards):
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        # k_q = -0.75
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target - self.env.last_joint_pos_target)
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target - 2 * self.env.last_joint_pos_target + self.env.last_last_joint_pos_target)
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:,:12] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    # encourage robot velocity align vector from robot body to ball
    # r_cv
    def _reward_dribbling_robot_ball_vel(self):
        FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        FR_HIP_velocities = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,7:10].view(self.env.num_envs,3))
        
        delta_dribbling_robot_ball_vel = 1.0
        robot_ball_vec = self.env.object_local_pos[:,0:2] - FR_HIP_positions[:,0:2]
        d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).unsqueeze(dim=-1)
        ball_robot_velocity_projection = torch.norm(self.env.commands[:,:2], dim=-1) - torch.sum(d_robot_ball * FR_HIP_velocities[:,0:2], dim=-1) # set approaching speed to velocity command
        velocity_concatenation = torch.cat((torch.zeros(self.env.num_envs,1, device=self.env.device), ball_robot_velocity_projection.unsqueeze(dim=-1)), dim=-1)
        rew_dribbling_robot_ball_vel=torch.exp(-delta_dribbling_robot_ball_vel* torch.pow(torch.max(velocity_concatenation,dim=-1).values, 2) )
        return rew_dribbling_robot_ball_vel

    # encourage robot near ball
    # r_cp
    def _reward_dribbling_robot_ball_pos(self):

        FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)

        delta_dribbling_robot_ball_pos = 4.0
        rew_dribbling_robot_ball_pos = torch.exp(-delta_dribbling_robot_ball_pos * torch.pow(torch.norm(self.env.object_local_pos - FR_HIP_positions, dim=-1), 2) )
        return rew_dribbling_robot_ball_pos 

    # encourage ball vel align with unit vector between ball target and ball current position
    # r^bv
    def _reward_dribbling_ball_vel(self):
        # target velocity is command input
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.object_lin_vel[:, :2]), dim=1)
        # rew_dribbling_ball_vel = torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
        return torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
        
    def _reward_dribbling_robot_ball_yaw(self):
        robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - self.env.base_pos[:,0:2]
        d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).unsqueeze(dim=-1)

        unit_command_vel = self.env.commands[:,:2] / torch.norm(self.env.commands[:,:2], dim=-1).unsqueeze(dim=-1)
        robot_ball_cmd_yaw_error = torch.norm(unit_command_vel, dim=-1) - torch.sum(d_robot_ball * unit_command_vel, dim=-1)

        # robot ball vector align with body yaw angle
        roll, pitch, yaw = get_euler_xyz(self.env.base_quat)
        body_yaw_vec = torch.zeros(self.env.num_envs, 2, device=self.env.device)
        body_yaw_vec[:,0] = torch.cos(yaw)
        body_yaw_vec[:,1] = torch.sin(yaw)
        robot_ball_body_yaw_error = torch.norm(body_yaw_vec, dim=-1) - torch.sum(d_robot_ball * body_yaw_vec, dim=-1)
        delta_dribbling_robot_ball_cmd_yaw = 2.0
        rew_dribbling_robot_ball_yaw = torch.exp(-delta_dribbling_robot_ball_cmd_yaw * (robot_ball_cmd_yaw_error+robot_ball_body_yaw_error))
        return rew_dribbling_robot_ball_yaw
    
    def _reward_dribbling_ball_vel_norm(self):
        # target velocity is command input
        vel_norm_diff = torch.pow(torch.norm(self.env.commands[:, :2], dim=-1) - torch.norm(self.env.object_lin_vel[:, :2], dim=-1), 2)
        delta_vel_norm = 2.0
        rew_vel_norm_tracking = torch.exp(-delta_vel_norm * vel_norm_diff)
        return rew_vel_norm_tracking

    # def _reward_dribbling_ball_vel_angle(self):
    #     angle_diff = torch.atan2(self.env.commands[:,1], self.env.commands[:,0]) - torch.atan2(self.env.object_lin_vel[:,1], self.env.object_lin_vel[:,0])
    #     angle_diff_in_pi = torch.pow(wrap_to_pi(angle_diff), 2)
    #     rew_vel_angle_tracking = torch.exp(-5.0*angle_diff_in_pi/(torch.pi**2))
    #     # print("angle_diff", angle_diff, " angle_diff_in_pi: ", angle_diff_in_pi, " rew_vel_angle_tracking", rew_vel_angle_tracking, " commands", self.env.commands[:, :2], " object_lin_vel", self.env.object_lin_vel[:, :2])
    #     return rew_vel_angle_tracking

    def _reward_dribbling_ball_vel_angle(self):
        angle_diff = torch.atan2(self.env.commands[:,1], self.env.commands[:,0]) - torch.atan2(self.env.object_lin_vel[:,1], self.env.object_lin_vel[:,0])
        angle_diff_in_pi = torch.pow(wrap_to_pi(angle_diff), 2)
        rew_vel_angle_tracking = 1.0 - angle_diff_in_pi/(torch.pi**2)
        return rew_vel_angle_tracking