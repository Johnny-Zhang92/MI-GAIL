import numpy as np
import torch
import gym
import random
import sklearn.metrics as skm
from common import lfd_envs


def compute_mutual_information(x, y):
    # 互信息具有对称性，因此 x 和 y 的顺序不影响结果
    return skm.mutual_info_score(x, y)


def compute_mutual_information_per_dimension(x, y):
    # x = np.reshape(x, [-1, x.shape[-1]])
    # y = np.reshape(y, [-1, y.shape[-1]])
    mi = 0
    for i in range(x.shape[-1]):
        x_temp = x[:, i]
        y_temp = y[:, i]
        temp_mi = skm.mutual_info_score(x_temp.reshape(-1), y_temp.reshape(-1))
        mi = mi + temp_mi
    mi = mi / x.shape[-1]
    return mi


def sample_from_current_policy(policy, env_name, seed, learn_absorbing):
    eval_env = gym.make(env_name)
    if learn_absorbing:
        eval_env = lfd_envs.AbsorbingWrapper(eval_env)
    eval_env.seed(seed)

    avg_reward = 0.
    avg_timesteps = 0
    eval_episodes = 0

    state_list, steps = [], 0
    state, done = eval_env.reset(), False
    while True:
        action = policy.select_action(np.array(state))
        state_action = np.concatenate([state[:-1], action], axis=-1)
        if steps < 5000:
            state_list.append(state_action)
            steps += 1
        else:
            break

        action = policy.select_action(np.array(state))
        state, reward, done, _ = eval_env.step(action)
        avg_reward += reward
        avg_timesteps += 1
        if done:
            state, done = eval_env.reset(), False
            eval_episodes += 1

    avg_reward /= eval_episodes
    avg_timesteps /= eval_episodes

    # print("---------------------------------------")
    print(f"Mutual Information: Ge:{eval_episodes} episodes, Average_return: {avg_reward:.3f}, "
          f"Average_steps:{avg_timesteps:.3f}")
    print("---------------------------------------")
    return np.array(state_list)


# def sample_from_expert_demonstrations(expert_replay_buffer, mi_batch_size):
#     expert_traj_num = expert_replay_buffer.expert_data.state.shape[0]
#     idx = np.arange(0, expert_traj_num)
#     np.random.shuffle(idx)
#     # 专家样本每行100个state，取5行也就是5000个state
#     expert_state = expert_replay_buffer.expert_data.state[idx[0:int(mi_batch_size/1000)], :]
#     return expert_state.reshape(mi_batch_size, -1)

def sample_from_expert_demonstrations(expert_replay_buffer, num_trajectories):
    e_sa = expert_replay_buffer.MI_expert_sa
    idx = np.arange(0, e_sa.shape[0])
    np.random.shuffle(idx)
    mi_expert_sa = e_sa[idx[0:num_trajectories]]
    mi_expert_sa = mi_expert_sa.reshape([-1, mi_expert_sa.shape[-1]])
    return mi_expert_sa


def get_mutual_information(policy, env_name, seed, learn_absorbing, expert_replay_buffer, mi_batch_size):
    current_state = sample_from_current_policy(policy, env_name, seed, learn_absorbing)
    expert_state = sample_from_expert_demonstrations(expert_replay_buffer, int(mi_batch_size / 1000))

    # 判断数据类型和维度是否满足要求
    assert isinstance(current_state, np.ndarray), "current_state is not a np.ndarray"
    assert isinstance(expert_state, np.ndarray), "expert_state is not a np.ndarray"
    assert current_state.shape == expert_state.shape, "current_state.shape:{} isn't equal to expert_state.shape:{}".format(
        current_state.shape, expert_state.shape)

    mutual_information_per_dimension = compute_mutual_information_per_dimension(expert_state, current_state)
    current_state, expert_state = np.reshape(current_state, -1), np.reshape(expert_state, -1)
    mutual_information = compute_mutual_information(expert_state, current_state)
    return mutual_information, mutual_information_per_dimension
