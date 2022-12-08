# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Various functions used for TD3 and DDPG implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from common.replay_buffer import Mask
import numpy as np
import torch
import torch.nn as nn


# from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe
def tensor_wrapper(x, device='cpu') -> torch.Tensor:
    x = torch.FloatTensor(x)
    if len(x.shape) < 2:
        x.unsqueeze_(0)
    return x.to(device)


def TensorWrapper(x, device='cpu') -> torch.Tensor:
    x = torch.FloatTensor(x)
    if len(x.shape) < 2:
        x.unsqueeze_(-1)
    return x.to(device)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


def weights_init(m: nn.Module, gains: list = None):
    if gains is None:
        gains = [1.0 for _ in range(len(m))]
    for i, layer in enumerate(m):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gains[i])
            nn.init.constant_(layer.bias, 0.)


def soft_update(vars_, target_vars, tau=1.0):
    """Performs soft updates of the target networks.

  Args:
    vars_: a list of parameters of a source network ([tf.Variable]).
    target_vars: a list of parameters of a target network ([tf.Variable]).
    tau: update parameter.
  """
    for var, var_target in zip(vars_, target_vars):
        var_target.assign((1 - tau) * var_target + tau * var)


def do_rollout(env,
               actor,
               replay_buffer,
               noise_scale=0.1,
               num_trajectories=1,
               rand_actions=0,
               sample_random=False,
               add_absorbing_state=False,
               mi_model=None,
               expert_dataset_iter=None,
               policy_replay_buffer_iter=None,
               mi_replay_buffer=None,
               shift=None, scale=None,
               expert_states_instance=None,
               args=None):
    """Do N rollout.

  Args:
      scale: std of expert states
      shift: mean of expert states
      env: environment to train on.
      actor: policy to take actions.
      replay_buffer: replay buffer to collect samples.
      noise_scale: std of gaussian noise added to a policy output.
      num_trajectories: number of trajectories to collect.
      rand_actions: number of random actions before using policy.
      sample_random: whether to sample a random trajectory or not.
      add_absorbing_state: whether to add an absorbing state.
  Returns:
    An episode reward and a number of episode steps.
  """
    total_reward = 0
    total_timesteps = 0

    for _ in range(num_trajectories):
        obs = env.reset()
        # print(obs.shape, shift.shape)
        obs_mi = (obs + shift) * scale
        episode_timesteps = 0
        while True:
            if (replay_buffer is not None and len(replay_buffer) < rand_actions) \
                    or sample_random:
                action = env.action_space.sample()
            else:
                if args.DAC_algo == 'td3':
                   # action = actor.select_noised_action(state=obs)
                    state1 = torch.FloatTensor(obs).to("cuda")
                    action = actor.actor(state1)
                    action = action.detach().cpu().numpy()
                    if noise_scale > 0:
                        action += np.random.normal(size=action.shape) * noise_scale
                    action = action.clip(-1, 1)
                elif args.DAC_algo == 'sac':
                    action = actor.select_action(obs)

            next_obs, reward, done, _ = env.step(action)
            next_obs_mi = (next_obs + shift) * scale
            # Extremely important, otherwise Q function is not stationary!
            # Taken from: https://github.com/sfujim/TD3/blob/master/main.py#L123
            if not done or episode_timesteps + 1 == env._max_episode_steps:  # pylint: disable=protected-access
                done_mask = Mask.NOT_DONE.value  # 1.0
                # print("do_rollout,not_done_mask:{}".format(done_mask))
            else:
                done_mask = Mask.DONE.value  # 0.0
                # print("do_rollout,done_mask:{}".format(done_mask))

            # add absorbing samples to mi_replay_buffer
            done_mi = done and episode_timesteps + 1 == env._max_episode_steps
            if done and done_mi:
                next_obs_mi = env.get_absorbing_state()

            add_samples_to_replay_buffer(mi_replay_buffer, obs_mi, action,
                                         next_obs_mi, expert_states_instance[episode_timesteps],
                                         expert_states_instance[episode_timesteps+1])

            # add absorbing samples to mi_replay_buffer
            if done and not done_mi:
                for i_mi in range(10):
                    if i_mi + episode_timesteps < env._max_episode_steps:
                        obs_mi = env.get_absorbing_state()
                        action_mi = env.action_space.sample()
                        next_obs_mi = env.get_absorbing_state()

                        add_samples_to_replay_buffer(mi_replay_buffer, obs_mi, action_mi,
                                                     next_obs_mi, obs_mi, obs_mi)

            total_reward += reward
            episode_timesteps += 1
            total_timesteps += 1

            if replay_buffer is not None:
                if (add_absorbing_state and done and
                        episode_timesteps < env._max_episode_steps):  # pylint: disable=protected-access
                    next_obs = env.get_absorbing_state()
                replay_buffer.push_back(obs, action, next_obs, [reward], [done_mask],
                                        done, expert_states_instance[episode_timesteps - 1],
                                        expert_states_instance[episode_timesteps])

            if done:
                break

            obs = next_obs

        # Add an absorbing state that is extremely important for GAIL.
        if add_absorbing_state and (replay_buffer is not None and
                                    episode_timesteps < env._max_episode_steps):  # pylint: disable=protected-access
            action = np.zeros(env.action_space.shape)
            absorbing_state = env.get_absorbing_state()

            # done=False is set to the absorbing state because it corresponds to
            # a state where gym environments stopped an episode.

            replay_buffer.push_back(absorbing_state, action, absorbing_state, [0.0],
                                    [Mask.ABSORBING.value], False, absorbing_state, absorbing_state)
            # print("utils_115_mask.absorbing:{}".format(done_mask))

    return total_reward / num_trajectories, total_timesteps // num_trajectories


def add_samples_to_replay_buffer(replay_buffer, obs, action, next_obs, target_state0, target_state1):
    """Add a transition to a replay buffer.

  Args:
    replay_buffer: a replay buffer to add samples to.
    obs: observation.
    action: action.
    next_obs: next observation.
  """
    replay_buffer.add_batch((np.array([obs.astype(np.float32)]),
                             np.array([action.astype(np.float32)]),
                             np.array([next_obs.astype(np.float32)]),
                             np.array([[0]]).astype(np.float32),
                             np.array([[1.0]]).astype(np.float32),
                            np.array([target_state0.astype(np.float32)]),
                            np.array([target_state1.astype(np.float32)])))
