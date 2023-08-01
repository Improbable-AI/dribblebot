def train_go1(headless=True):

    import isaacgym
    assert isaacgym
    import torch

    from dribblebot.envs.base.legged_robot_config import Cfg
    from dribblebot.envs.go1.go1_config import config_go1
    from dribblebot.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

    from dribblebot_learn.ppo_cse import Runner
    from dribblebot.envs.wrappers.history_wrapper import HistoryWrapper
    from dribblebot_learn.ppo_cse.actor_critic import AC_Args
    from dribblebot_learn.ppo_cse.ppo import PPO_Args
    from dribblebot_learn.ppo_cse import RunnerArgs

    config_go1(Cfg)
    Cfg.env.num_envs = 1000

    RunnerArgs.resume = True
    RunnerArgs.resume_path = "improbableailab/dribbling/j34kr9ds"
    RunnerArgs.resume_checkpoint = 'tmp/legged_data/ac_weights_last.pt' 


    Cfg.robot.name = "go1"
    Cfg.sensors.sensor_names = [
                        "ObjectSensor",
                        "OrientationSensor",
                        "RCSensor",
                        "JointPositionSensor",
                        "JointVelocitySensor",
                        "ActionSensor",
                        "LastActionSensor",
                        "ClockSensor",
                        "YawSensor",
                        "TimingSensor",
                        ]
    Cfg.sensors.sensor_args = {
                        "ObjectSensor": {},
                        "OrientationSensor": {},
                        "RCSensor": {},
                        "JointPositionSensor": {},
                        "JointVelocitySensor": {},
                        "ActionSensor": {},
                        "LastActionSensor": {"delay": 1},
                        "ClockSensor": {},
                        "YawSensor": {},
                        "TimingSensor":{},
                        }
    Cfg.sensors.privileged_sensor_names = {
                        "BodyVelocitySensor": {},
                        "ObjectVelocitySensor": {},
    }
    Cfg.sensors.privileged_sensor_args = {
                        "BodyVelocitySensor": {},
                        "ObjectVelocitySensor": {},
    }

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90
    Cfg.curriculum_thresholds.dribbling_ball_vel = 0.8

    Cfg.commands.distributional_commands = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_friction = False # True
    Cfg.domain_rand.randomize_restitution = False # True
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.99
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.com_displacement_range = [-0.15, 0.15]
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.domain_rand.ground_friction_range = [0.0, 0.0]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.99, 1.01]
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.002, 0.002]
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = True
    Cfg.domain_rand.randomize_Kd_factor = True
    Cfg.domain_rand.randomize_ball_drag = True
    Cfg.domain_rand.drag_range = [-0.1, 0.8]
    Cfg.domain_rand.ball_drag_rand_interval_s = 15.0

    Cfg.env.num_observation_history = 15
    Cfg.reward_scales.feet_contact_forces = 0.0
    Cfg.env.num_envs = 1000

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.balance_gait_distribution = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False

    ###############################
    # soccer dribbling configuration
    ###############################

    # ball parameters
    Cfg.env.add_balls = True

    # domain randomization ranges
    Cfg.domain_rand.rand_interval_s = 6
    Cfg.domain_rand.friction_range = [0.0, 1.5]
    Cfg.domain_rand.randomize_ground_friction = True
    Cfg.domain_rand.ground_friction_range = [0.7, 4.0]
    Cfg.domain_rand.restitution_range = [0.0, 0.4]
    Cfg.domain_rand.added_mass_range = [-1.0, 3.0]
    Cfg.domain_rand.gravity_range = [-1.0, 1.0]
    Cfg.domain_rand.motor_strength_range = [0.99, 1.01]
    Cfg.domain_rand.motor_offset_range = [-0.002, 0.002]
    Cfg.domain_rand.tile_roughness_range = [0.0, 0.0]

    # privileged obs in use
    Cfg.env.num_privileged_obs = 6
    Cfg.env.priv_observe_ball_drag = True


    # sensory observation
    Cfg.commands.num_commands = 15
    Cfg.env.episode_length_s = 40.
    Cfg.env.num_observations = 75

    # terrain configuration
    Cfg.terrain.border_size = 0.0
    Cfg.terrain.mesh_type = "boxes_tm"
    Cfg.terrain.num_cols = 20
    Cfg.terrain.num_rows = 20
    Cfg.terrain.terrain_length = 5.0
    Cfg.terrain.terrain_width = 5.0
    Cfg.terrain.num_border_boxes = 5.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False
    Cfg.terrain.center_span = 3
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.curriculum = False
    Cfg.terrain.difficulty_scale = 1.0
    Cfg.terrain.max_step_height = 0.26
    Cfg.terrain.min_step_run = 0.25
    Cfg.terrain.max_step_run = 0.4
    Cfg.terrain.max_init_terrain_level = 1

    # terminal conditions
    Cfg.rewards.use_terminal_body_height = True
    Cfg.rewards.terminal_body_height = 0.2
    Cfg.rewards.use_terminal_roll_pitch = False
    Cfg.rewards.terminal_body_ori = 0.5

    # command sampling
    Cfg.commands.resampling_time = 7
    Cfg.commands.heading_command = False

    Cfg.commands.lin_vel_x = [-1.5, 1.5]
    Cfg.commands.lin_vel_y = [-1.5, 1.5]
    Cfg.commands.ang_vel_yaw = [-0.0, 0.0]
    Cfg.commands.body_height_cmd = [-0.05, 0.05]
    Cfg.commands.gait_frequency_cmd_range = [3.0, 3.0]
    Cfg.commands.gait_phase_cmd_range = [0.5, 0.5]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [0.5, 0.5]
    Cfg.commands.footswing_height_range = [0.09, 0.09]
    Cfg.commands.body_pitch_range = [-0.0, 0.0]
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    Cfg.commands.stance_width_range = [0.0, 0.1]
    Cfg.commands.stance_length_range = [0.0, 0.1]

    Cfg.commands.limit_vel_x = [-1.5, 1.5]
    Cfg.commands.limit_vel_y = [-1.5, 1.5]
    Cfg.commands.limit_vel_yaw = [-0.0, 0.0]
    Cfg.commands.limit_body_height = [-0.05, 0.05]
    Cfg.commands.limit_gait_frequency = [3.0, 3.0]
    Cfg.commands.limit_gait_phase = [0.5, 0.5]
    Cfg.commands.limit_gait_offset = [0.0, 0.0]
    Cfg.commands.limit_gait_bound = [0.0, 0.0]
    Cfg.commands.limit_gait_duration = [0.5, 0.5]
    Cfg.commands.limit_footswing_height = [0.09, 0.09]
    Cfg.commands.limit_body_pitch = [-0.0, 0.0]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.0, 0.1]
    Cfg.commands.limit_stance_length = [0.0, 0.1]

    Cfg.commands.num_bins_vel_x = 1
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 1
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.rewards.constrict = False

    # reward function
    Cfg.reward_scales.orientation = -5.0
    Cfg.reward_scales.torques = -0.0001
    Cfg.reward_scales.dof_vel = -0.0001
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.collision = -5.0
    Cfg.reward_scales.action_rate = -0.01
    Cfg.reward_scales.tracking_contacts_shaped_force = 4.0
    Cfg.reward_scales.tracking_contacts_shaped_vel = 4.0
    Cfg.reward_scales.dof_pos_limits = -10.0
    Cfg.reward_scales.dof_pos = -0.05
    Cfg.reward_scales.action_smoothness_1 = -0.1
    Cfg.reward_scales.action_smoothness_2 = -0.1
    Cfg.reward_scales.dribbling_robot_ball_vel = 0.5
    Cfg.reward_scales.dribbling_robot_ball_pos = 4.0
    Cfg.reward_scales.dribbling_ball_vel = 4.0
    Cfg.reward_scales.dribbling_robot_ball_yaw = 4.0
    Cfg.reward_scales.dribbling_ball_vel_norm = 4.0
    Cfg.reward_scales.dribbling_ball_vel_angle = 4.0

    Cfg.reward_scales.tracking_lin_vel = 0.0
    Cfg.reward_scales.tracking_ang_vel = 0.0
    Cfg.reward_scales.lin_vel_z = 0.0
    Cfg.reward_scales.ang_vel_xy = 0.0
    Cfg.reward_scales.feet_air_time = 0.0

    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.
    
    Cfg.rewards.reward_container_name = "SoccerRewards"
    Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = True
    Cfg.rewards.sigma_rew_neg = 0.02

    # normalization
    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0.7, 4.0]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    # reward function (not in use)
    Cfg.reward_scales.feet_slip = -0.0
    Cfg.reward_scales.jump = 0.0
    Cfg.reward_scales.base_height = -0.0
    Cfg.reward_scales.feet_impact_vel = -0.0
    Cfg.reward_scales.feet_air_time = 0.0

    Cfg.asset.terminate_after_contacts_on = []

    AC_Args.adaptation_labels = []
    AC_Args.adaptation_dims = []

    RunnerArgs.save_video_interval = 500

    import wandb
    wandb.init(
      # set the wandb project where this run will be logged
      project="dribbling",

      # track hyperparameters and run metadata
      config={
      "AC_Args": vars(AC_Args),
      "PPO_Args": vars(PPO_Args),
      "RunnerArgs": vars(RunnerArgs),
      "Cfg": vars(Cfg),
      }
    )

    device = 'cuda:0'
    # device = 'cpu'
    env = VelocityTrackingEasyEnv(sim_device=device, headless=False, cfg=Cfg)

    env = HistoryWrapper(env)
    runner = Runner(env, device=device)
    runner.learn(num_learning_iterations=1000000, init_at_random_ep_len=True, eval_freq=100)


if __name__ == '__main__':
    from pathlib import Path
    from dribblebot import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem
    
    # to see the environment rendering, set headless=False
    train_go1(headless=False)

