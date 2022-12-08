from absl import flags
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="Hopper-v2")  # OpenAI gym environment name
    parser.add_argument("--expert_path", default="./expert_demonstrations/MI_GAIL/DAC_Hopper-v2_johnny.npz",
                        type=str,
                        help="Directory to load the expert demos.")
    parser.add_argument("--num_expert_trajectories", default=40, type=int,
                        help="Number of trajectories taken from the expert.")
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--cuda", default=1, type=int, help="Use Gpu or not")
    parser.add_argument("--GPU_num", default=0, type=int, help="The gpu id of the gpu you want to use.")
    parser.add_argument("--replay_buffer_size", default=1000000, type=int,
                        help="the maximum size of replay buffer.")
    # train_MI_GAIL 和 train_DAC 有一个为1，另一个必须为0，也就是训练算法只能一次选一个。
    parser.add_argument("--train_MI_GAIL", default=1, type=int)
    parser.add_argument("--train_MI_GAIL_no_expert", default=0, type=int)
    parser.add_argument("--train_DAC", default=0, type=int)
    parser.add_argument("--mi_weight", default=0.01, type=float)

    parser.add_argument("--TD3_policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--TD3_actor_lr", default=1e-3, type=float, help="Initial actor learning rate.")
    parser.add_argument("--TD3_critic_lr", default=1e-3, type=float, help="Initial actor learning rate.")
    parser.add_argument("--TD3_start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--TD3_updates_per_step", default=1, type=int, help="Number of updates per step.")
    parser.add_argument("--TD3_eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--TD3_expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--TD3_batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--TD3_discount", default=0.99)  # Discount factor
    parser.add_argument("--TD3_tau", default=0.005)  # Target network update rate
    parser.add_argument("--TD3_policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--TD3_noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--TD3_policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--TD3_save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--TD3_load_model",
                        default="")  # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--DAC_exploration_noise", default=0.1, type=float,
                        help="Scale of noise used for exploration.")
    parser.add_argument("--DAC_gail_lr", default=3e-4, type=float, help="Initial discriminator learning rate.")
    parser.add_argument("--DAC_lambd", default=10, type=float, help="Number of grad penalty factory")
    parser.add_argument("--DAC_random_actions", default=int(1e4), type=int,
                        help='Number of random actions to sample to replay buffer '
                             'before sampling policy actions.')
    parser.add_argument("--DAC_trajectory_size", default=50, type=int,
                        help="Size of every trajectory after subsampling.")

    parser.add_argument("--DAC_batch_size", default=256, type=int, help="Batch size.")
    parser.add_argument("--DAC_min_samples_to_start", default=1000, type=int,
                        help="Minimal number of samples in replay buffer to start ' 'training.")
    # save_dir--是否加载保存的模型
    parser.add_argument("--DAC_save_dir", default="", type=str, help="Directory to save models.")
    parser.add_argument("--DAC_eval_save_dir", default="", type=str,
                        help="Directory to save policy for evaluation.")
    parser.add_argument("--DAC_gail_loss", default="airl", type=str,
                        help="GAIL loss to use, gail is -log(1-sigm(D)), airl is D : ' 'gail | airl.")
    parser.add_argument("--DAC_save_interval", default=int(1e5), type=int, help="Save every N timesteps.")
    parser.add_argument("--DAC_eval_save_interval", default=int(1e3), type=int,
                        help="Save for evaluation every N timesteps.")
    # DAC_parser.add_argument("--seed", default=42, type=int, help="Fixed random seed for training.")
    parser.add_argument("--DAC_learn_absorbing", default=1, type=int,
                        help="Whether to learn the reward for absorbing states or not.")
    # DAC_parser.add_argument("--master", default="local", type=str, help="Location of the session.")
    # DAC_parser.add_argument("--ps_tasks", default=0, type=int, help="Number of Parameter Server tasks.")
    # DAC_parser.add_argument("--task_id", default=0, type=int, help="Id of the current TF task.")
    parser.add_argument("--DAC_algo", default='td3', type=str, help="Algorithm to use for training: ddpg | td3.")

    parser.add_argument("--MI_target_interval", default=100, type=int,
                        help="the interval between each target state")
    # DAC_parser.add_argument("--actor_lr", default=1e-3, type=float, help="Initial actor learning rate.")

    parser.add_argument("--VD_env_name", default="Hopper-v2",
                        type=str, help="Directory to load expert demos.")
    parser.add_argument("--VD_seed", default=42,
                        type=int, help="Directory to load expert demos.")
    parser.add_argument("--VD_expert_dir", default="./expert_demonstrations/MI/",
                        type=str, help="Directory to load expert demos.")
    parser.add_argument("--VD_sample_batch_size", default=256, type=int, help="'Batch size.'")
    parser.add_argument("--VD_actor_update_freq", default=1, type=int, help="Update actor every N steps.")
    parser.add_argument("--VD_discount", default=0.99, type=float,
                        help="Discount used for returns.")
    parser.add_argument("--VD_nu_lr", default=1e-3, type=float, help="nu network learning rate.")
    parser.add_argument("--VD_actor_lr", default=1e-5, type=float, help="Actor learning rate.")
    parser.add_argument("--VD_critic_lr", default=1e-3, type=float, help="Critic learning rate.")
    parser.add_argument("--VD_sac_alpha", default=0.1, type=float, help="SAC temperature.")
    parser.add_argument("--VD_tau", default=0.005, type=float, help="Soft update coefficient for the target "
                                                                    "network.")
    parser.add_argument('--VD_hidden_size', type=int, default=256, metavar='G', help='Hidden size.')
    parser.add_argument("--VD_updates_per_step", default=5, type=int, help="Updates per time step.")
    parser.add_argument("--VD_max_timesteps", default=int(1e5), type=int, help="Max timesteps to train.")
    parser.add_argument("--VD_num_trajectories", default=1, type=int, help="Number of trajectories to use.")
    parser.add_argument("--VD_num_random_actions", default=int(2e3), type=int,
                        help="Fill replay buffer with N random actions.")
    parser.add_argument("--VD_start_training_timesteps", default=int(1e3), type=int,
                        help="Start training when replay buffer contains N timesteps.")

    parser.add_argument("--VD_learn_alpha", default=1, type=int, help="Whether to learn temperature for SAC.")
    parser.add_argument("--VD_normalize_states", default=1, type=int, help="Normalize states using expert stats.")
    parser.add_argument("--VD_log_interval", default=int(1e-3), type=int, help="Log every N timesteps.")
    parser.add_argument("--VD_eval_interval", default=int(1e3), type=int, help="Evaluate every N timesteps.")

    parser.add_argument("--VD_algo", default='value_dice', type=str,
                        help="Algorithm to use to compute occupancy ration.")
    parser.add_argument("--VD_absorbing_per_episode", default=10, type=int,
                        help="A number of absorbing states per episode to add.")
    args = parser.parse_args()
    return args
