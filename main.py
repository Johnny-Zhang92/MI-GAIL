from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=g-import-not-at-top,g-bad-import-order
import platform

if int(platform.python_version_tuple()[0]) < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

import os
import random
import torch
import zlib
import csv
import datetime
from tqdm import tqdm, trange
from common.compute_mutual_information import get_mutual_information

from absl import logging
# import ddpg_td3

import gym

import numpy as np
from common.replay_buffer import ReplayBuffer
from common.replay_buffer import ExpertReplayBuffer
from common.replay_buffer import TimeStep
# import tensorflow.compat.v1 as tf
from common.utils import do_rollout
from common import lfd_envs
from common import gail
from common.TD3 import TD3
from common.sac.sac import SAC
import common.mi_reward as vd
from torch.utils.tensorboard import SummaryWriter
from mi_model import wrappers
from mi_model import data_utils
import tensorflow.compat.v2 as tf


def set_tf_gpu(gpu_num):
    """GPU相关设置"""

    # 打印变量在那个设备上
    tf.debugging.set_log_device_placement(True)
    # 获取物理GPU个数
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('物理GPU个数为：', len(gpus))
    # 设置内存自增长
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('-------------已设置完GPU内存自增长--------------')

    # 设置哪个GPU对设备可见，即指定用哪个GPU
    tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
    # 获取逻辑GPU个数
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('逻辑GPU个数为：', len(logical_gpus), "使用gpu：", gpu_num)


def select_algoritm(train_MI_GAIL_with_expert, train_DAC, train_MI_GAIL_no_expert):
    if train_MI_GAIL_with_expert:
        algoritm = [1, 1, 0]
        algorithm_name = "MI-GAIL_with_expert"
        print("Warning! Training MI_GAIL now!")
    elif train_DAC:
        algoritm = [0, 1, 0]
        algorithm_name = "DAC"
        print("Warning! Training DAC now!")
    elif train_MI_GAIL_no_expert:
        algoritm = [1, 0, 1]
        algorithm_name = "MI_GAIL_no_expert"
        print("Warning! Training MI_GAIL_no_expert now!")
    else:
        print("train_MI_GAIL_with_expert:{}, train_DAC:{}, train_MI_GAIL_no_expert:{}"
              .format(train_MI_GAIL_with_expert, train_DAC, train_MI_GAIL_no_expert))
        raise ["No matching algoritm"]
    return algoritm, algorithm_name


def main(args):
    algoritm, algorithm_name = select_algoritm(train_MI_GAIL_with_expert=args.train_MI_GAIL,
                                               train_DAC=args.train_DAC,
                                               train_MI_GAIL_no_expert=args.train_MI_GAIL_no_expert)
    # set_tf_gpu(gpu_num=args.GPU_num)
    # Tensorboard
    writer = SummaryWriter(
        'runs/{}_{}_{}_seed[{}]_exp_traj_num[{}]_mi_weight_[{}]'.format(algorithm_name, datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"), args.env_name, args.seed, args.num_expert_trajectories, args.mi_weight))

    # Environment
    env = gym.make(args.env_name)
    # 为应用 DAC 算法中关于吸收态的改进，创建环境，对于非吸收态状态，在状态最后增加一个维度(值为 0)；
    # 对于吸收态状态，在状态最后增加一个维度(值为 1)。
    if args.DAC_learn_absorbing:
        # env = lfd_envs.AbsorbingWrapper(env)
        (expert_states, expert_actions, expert_next_states,
         expert_dones, expert_index) = data_utils.load_expert_data(args.expert_path, num_traj=40)

        (expert_states, expert_actions, expert_next_states,
         expert_dones, expert_index) = data_utils.subsample_trajectories(expert_states,
                                                                         expert_actions,
                                                                         expert_next_states,
                                                                         expert_dones,
                                                                         args.VD_num_trajectories,
                                                                         expert_index)
        print('# of demonstraions: {}'.format(expert_states.shape[0]))

        if args.VD_normalize_states:
            shift = -np.mean(expert_states, 0)
            scale = 1.0 / (np.std(expert_states, 0) + 1e-3)
            expert_states = (expert_states + shift) * scale
            expert_next_states = (expert_next_states + shift) * scale

            shift1 = np.concatenate([shift, [0]], axis=0)
            scale1 = np.concatenate([scale, [0]], axis=0)
        else:
            shift = None
            scale = None
        env = wrappers.create_il_env(args.env_name, args.seed, shift, scale)
        # env.step()

    if args.env_name in ['HalfCheetah-v2', 'Ant-v2']:
        rand_actions = int(1e4)
    else:
        rand_actions = int(1e3)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action,
              "discount": args.TD3_discount, "tau": args.TD3_tau}
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Discriminator：Instantiation GAIL class
    # Default: FLAGS.trajectory_size = 50
    subsampling_rate = env._max_episode_steps // args.DAC_trajectory_size  # pylint: disable=protected-access
    lfd = gail.GAIL(state_dim + action_dim, subsampling_rate=subsampling_rate,
                    gail_loss=args.DAC_gail_loss,
                    gail_lr=args.DAC_gail_lr,
                    device=device, lambd=args.DAC_lambd)

    # Generator：Instantiation TD3 class
    if args.DAC_algo == 'td3':
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.TD3_policy_noise * max_action
        kwargs["noise_clip"] = args.TD3_noise_clip * max_action
        kwargs["policy_freq"] = args.TD3_policy_freq
        kwargs["actor_lr"] = args.TD3_actor_lr
        kwargs["critic_lr"] = args.TD3_critic_lr
        kwargs["expl_noise"] = args.TD3_expl_noise
        kwargs["device"] = device
        kwargs["learn_absorbing"] = args.DAC_learn_absorbing
        kwargs["shift"] = shift1
        kwargs["scale"] = scale1
        kwargs["critic_for_expert"] = algoritm[1]
        kwargs["critic_for_no_expert"] = algoritm[2]
        generator = TD3(**kwargs)
    elif args.DAC_algo == 'sac':
        generator = SAC(env.observation_space.shape[0], env.action_space, args)
    else:
        raise NameError("can't find algorithm that is dac_args.algo")

    imitator, expert_dataset_iter, policy_replay_buffer_iter, vd_env, vd_eval_env, \
    policy_replay_buffer, expert_states_instance \
        = vd.value_dice_prepare(args, args.expert_path)

    # Instantiation class：sampling state,action from environment
    # random_reward, _ = do_rollout(env, generator.actor, None, num_trajectories=10, sample_random=True)

    # Load expert data and construct expert replay buffer
    expert_replay_buffer = ExpertReplayBuffer(num_trajectories=args.num_expert_trajectories,
                                              expert_path=args.expert_path,
                                              target_interval=args.MI_target_interval,
                                              seed=args.seed, mi_num_traj=int(5000 / 1000))

    # Subsample after adding absorbing states, because otherwise we can lose
    # final states.
    # print('Original dataset size {}'.format(len(expert_replay_buffer)))
    # expert_replay_buffer.subsample_transitions(subsampling_rate)
    # print('Subsampled dataset size {}'.format(len(expert_replay_buffer)))
    reward_scale = expert_replay_buffer.expert_data.avg_ret

    # 对于能跑满1000步的专家样本，这一步没有用
    if args.DAC_learn_absorbing:
        expert_replay_buffer.add_absorbing_states(env)
    print("expert_replay_buffer_len:", expert_replay_buffer.__len__())
    # Test
    # time_step = expert_replay_buffer.sample(batch_size=dac_args.batch_size)
    # expert_batch = TimeStep(*zip(*time_step))
    # lfd.update(expert_batch, expert_batch)
    # Test

    replay_buffer = ReplayBuffer(capacity=args.replay_buffer_size, seed=args.seed)
    total_numsteps = 0
    iters = 0
    id, mi_iter = 0, 0
    save_interval = 1000
    prev_print_timestep, write_tensorboard_timestep = 0, 0
    prev_eval_save_timestep, prev_eval_save_timestep_mi = 0, 0
    get_mutual_information_timestep = 0
    critic_loss, actor_loss = 0, 0
    train_value_dice, mi_reward, dis_reward = 0, 0, 0
    avg_reward = 0
    training_imitator, training_generator_discriminator = 1, 1
    expert_loss, gene_loss, total_loss, gen_accuracy, expert_accuracy = 0, 0, 0, 0, 0

    reward_upper_bound, mi_reward_weight = get_reward_upper_bound_and_mi_reward_weight(args.env_name)

    while total_numsteps < args.max_timesteps:
        # Decay helps to make the model more stable.
        # TODO(agrawalk): Use tf.train.exponential_decay
        # generator.change_learning_rate(total_time_steps=total_numsteps, initial_actor_lr=td3_args.actor_lr)
        rollout_reward, rollout_timesteps = do_rollout(env,
                                                       generator,
                                                       replay_buffer,
                                                       noise_scale=args.DAC_exploration_noise,  # 0.1
                                                       num_trajectories=1,
                                                       rand_actions=rand_actions,  # 1e3 or 1e4
                                                       sample_random=(generator.actor_step == 0),
                                                       add_absorbing_state=args.DAC_learn_absorbing,
                                                       mi_model=imitator,
                                                       expert_dataset_iter=expert_dataset_iter,
                                                       policy_replay_buffer_iter=policy_replay_buffer_iter,
                                                       mi_replay_buffer=policy_replay_buffer,
                                                       shift=shift1, scale=scale1,
                                                       expert_states_instance=expert_states_instance,
                                                       args=args)
        total_numsteps += rollout_timesteps
        iters += rollout_timesteps

        # if total_numsteps - train_value_dice >= 1000:
        #     average_returns, imitator, env, eval_env, expert_dataset_iter, \
        #     policy_replay_buffer_iter, policy_replay_buffer \
        #         = vd.update_value_dice(vd_args, imitator, env, eval_env,
        #                             expert_dataset_iter, policy_replay_buffer_iter,
        #                             policy_replay_buffer)
        #
        #     train_value_dice = total_numsteps

        # Update imitator algoritm
        if (total_numsteps >= 1000) and training_imitator and algoritm[0]:
            for _ in tqdm(range(5 * rollout_timesteps), desc='steps:{} training MI_model'.format(total_numsteps)):
                imitator.update(expert_dataset_iter, policy_replay_buffer_iter,
                                0.99, replay_regularization=0.1, expert_states_instance=expert_states_instance)

        if len(replay_buffer) >= args.DAC_min_samples_to_start:  # 1000
            if training_generator_discriminator:
                for _ in tqdm(range(rollout_timesteps),
                              desc='steps:{} training Discriminator'.format(total_numsteps)):  # Update Discriminator
                    time_step = replay_buffer.sample(batch_size=args.DAC_batch_size)
                    batch = TimeStep(*zip(*time_step))

                    time_step = expert_replay_buffer.sample(batch_size=args.DAC_batch_size)
                    expert_batch = TimeStep(*zip(*time_step))

                    expert_loss, gene_loss, total_loss, gen_accuracy, expert_accuracy \
                        = lfd.update(batch, expert_batch)

            # Update Generator by the Discriminator reward
            if training_generator_discriminator:
                for _ in tqdm(range(args.TD3_updates_per_step * rollout_timesteps),
                              desc="steps:{} training Generator0".format(total_numsteps)):  # Update Generator
                    time_step = replay_buffer.sample(batch_size=args.TD3_batch_size)
                    batch = TimeStep(*zip(*time_step))
                    if algoritm[1]:
                        time_step = expert_replay_buffer.sample(batch_size=args.DAC_batch_size)
                        expert_batch = TimeStep(*zip(*time_step))
                        critic_loss, actor_loss, mi_reward, dis_reward \
                            = generator.train(batch_samples=batch, expert_samples=expert_batch,
                                              dis_reward=lfd.get_reward,
                                              mi_model=imitator,
                                              total_number_step=total_numsteps,
                                              train_by_mi_reward=True, mi_weight=args.mi_weight)
                    elif algoritm[2]:
                        critic_loss, actor_loss, mi_reward, dis_reward \
                            = generator.old_train(batch_samples=batch,
                                                  dis_reward=lfd.get_reward,
                                                  mi_model=imitator,
                                                  total_number_step=total_numsteps)
                    else:
                        raise ["Please specify which algorithm you want to train? MI-GAIL or TD3?"]
                    # critic_loss, actor_loss, mi_reward, dis_reward \
                    #     = generator.old_train(batch_samples=batch,
                    #                           dis_reward=lfd.get_reward,
                    #                           mi_model=imitator,
                    #                           total_number_step=total_numsteps)
                print('mi_reward:***|{}, dis_reward:***|{}'.format(round(mi_reward, 4), round(dis_reward, 4)))

            # Evaluate Generator
            if total_numsteps - prev_eval_save_timestep >= args.DAC_eval_save_interval:  # 1000
                avg_reward = eval_policy(policy=generator, env_name=args.env_name, seed=args.seed,
                                         eval_episodes=5, learn_absorbing=args.DAC_learn_absorbing, args=args)
                prev_eval_save_timestep = total_numsteps
                if avg_reward >= reward_upper_bound:
                    training_generator_discriminator = 0

            # Print information
            if total_numsteps - prev_print_timestep >= 1000:
                # logging.info('Training: total timesteps:{}, episode reward.{}'.format(total_numsteps, round(rollout_reward, 4)))
                print('Training: total timesteps:{}, episode reward:{}, critic_loss:{}, actor_loss:{}, '
                      'expert_loss:{}, gene_loss:{}, total_loss:{}, gen_accuracy:{}, expert_accuracy:{}'.format
                      (total_numsteps, round(rollout_reward, 2), round(critic_loss, 4), round(actor_loss, 4),
                       round(expert_loss, 4), round(gene_loss, 4), round(total_loss, 4),
                       round(gen_accuracy, 4), round(expert_accuracy, 4)))
                prev_print_timestep = total_numsteps

            # Write to tensorboard
            if total_numsteps - write_tensorboard_timestep >= 1000:
                # Writing data to tensorboard
                writer.add_scalar("verified_episode_return", round(rollout_reward, 4), total_numsteps)
                writer.add_scalar("average_return", round(avg_reward, 4), total_numsteps)
                writer.add_scalar("critic_loss", round(critic_loss, 4), total_numsteps)
                writer.add_scalar("actor_loss", round(actor_loss, 4), total_numsteps)

            write_tensorboard_timestep = total_numsteps

            # if total_numsteps - get_mutual_information_timestep >= 1000:
            #     mi_scor, mi_scor_per_dim = get_mutual_information(policy=generator, env_name=args.env_name,
            #                                                       seed=args.seed,
            #                                                       learn_absorbing=args.DAC_learn_absorbing,
            #                                                       expert_replay_buffer=expert_replay_buffer,
            #                                                       mi_batch_size=5000)
            #     mi_data = [mi_scor, mi_iter, total_numsteps]
            #     mi_data_dim = [mi_scor_per_dim, mi_iter, total_numsteps]
            #     print("Write csv data: {} to {}".format(mi_data, csv_path_mi))
            #     with open(csv_path_mi, "a+", newline='') as mi_f:
            #         csv_writer_mi = csv.writer(mi_f)
            #         csv_writer_mi.writerow(mi_data)
            #     with open(csv_path_mi_dim, "a+", newline='') as mi_dim_f:
            #         csv_writer_mi_dim = csv.writer(mi_dim_f)
            #         csv_writer_mi_dim.writerow(mi_data_dim)
            #
            #     get_mutual_information_timestep, mi_iter = total_numsteps, mi_iter + 1


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, learn_absorbing=1, args=None):
    eval_env = gym.make(env_name)
    if learn_absorbing:
        eval_env = lfd_envs.AbsorbingWrapper(eval_env)
    eval_env.seed(seed)

    avg_reward = 0.
    avg_timesteps = 0

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            if args.DAC_algo == 'td3':
                action = policy.select_action(np.array(state))
            elif args.DAC_algo == 'sac':
                action = policy.select_action(state, evaluate=True)
            else:
                raise ["No matching algorithm to evaluate policy!"]
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            avg_timesteps += 1

    avg_reward /= eval_episodes
    avg_timesteps /= eval_episodes

    # print("---------------------------------------")
    print(f"Ge:{eval_episodes} episodes, Average_return: {avg_reward:.3f}, "
          f"Average_steps:{avg_timesteps:.3f}")
    print("---------------------------------------")
    return avg_reward


def get_env_parser(env_name):
    if env_name == "Hopper-v2":
        from env_parser.Hopper_Parser import get_parser
    elif env_name == "HalfCheetah-v2":
        from env_parser.HalfCheetah_Parser import get_parser
    elif env_name == "Ant-v2":
        from env_parser.Ant_Parser import get_parser
    elif env_name == "Walker2d-v2":
        from env_parser.Walker2d_Parser import get_parser
    elif env_name == "Humanoid-v2":
        from env_parser.Humanoid_Parser import get_parser
    else:
        raise Exception("No matching environment name in function get_env_parser!")
    args = get_parser()

    return args


def get_reward_upper_bound_and_mi_reward_weight(env_name):
    # reward_upper_bound, mi_reward_weight = 0, 0
    if env_name == "Hopper-v2":
        reward_upper_bound, mi_reward_weight = 3300, 0.9999
    elif env_name == "HalfCheetah-v2":
        reward_upper_bound, mi_reward_weight = 11000, 0.9999
    elif env_name == "Ant-v2":
        reward_upper_bound, mi_reward_weight = 4900, 0.9999
    elif env_name == "Walker2d-v2":
        reward_upper_bound, mi_reward_weight = 4600, 0.9999
    elif env_name == "Humanoid-v2":
        reward_upper_bound, mi_reward_weight = 5500, 0.9999
    else:
        print("env_name:{}".format(env_name))
        raise Exception("No matching environment name in function get_env_parser!")
    return reward_upper_bound, mi_reward_weight


def train_main(env_name):
    import sys
    print("*" * 40)
    print("sys.argv:", sys.argv)
    if len(sys.argv) > 1:
        print("sys.argv[0]:", sys.argv[2])
        for i in range(len(sys.argv)):
            if sys.argv[i] == "--env_name":
                env_name = sys.argv[i + 1]
                print("Find env_name in list sys.argv!!!!!!!!!!!!")
                break
            if i == len(sys.argv) - 1:
                raise Exception("Please input as the format: python main.py --env_name Hopper-v2")
    else:
        print("+" * 60)
        print("change env_name by list variable: env_names")
        print("+" * 60)
    args = get_env_parser(env_name)
    # Cuda or CPU
    print("args.GPU_num:{}".format(args.GPU_num))
    if args.cuda and torch.cuda.is_available():
        with torch.cuda.device(args.GPU_num):
            main(args=args)
    else:
        main(args=args)


if __name__ == '__main__':
    env_names = ["Ant-v2", "Hopper-v2", "HalfCheetah-v2", "Walker2d-v2", "Humanoid-v2",
                 "InvertedPendulum-v2", "InvertedDoublePendulum-v2"]
    train_main(env_name=env_names[1])
