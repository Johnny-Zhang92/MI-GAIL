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

"""Implementation of a local replay buffer for DDPG."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import random
from absl import logging
import numpy as np
from enum import Enum
from common.expert_data import MuJoCoExpertData

# Python内置四种基本container：list, dict, set, tuple，collections模块为其补充。
# namedtuple为collections中一个容器。一言以蔽之，是为有属性名字的tuple.

TimeStep = collections.namedtuple(
    'TimeStep',
    ('obs', 'action', 'next_obs', 'reward', 'mask',
     'done', 'target0', 'target1'))

# Separate Transition tuple to store advantages, returns (for compatibility).
# TODO(agrawalk) : Reconcile with TimeStep.
TimeStepAdv = collections.namedtuple(
    'TimeStepAdv',
    ('obs', 'action', 'next_obs', 'reward', 'mask', 'done',
     'log_prob', 'entropy', 'value_preds', 'returns', 'advantages'))


class ReplayBuffer(object):
    """A class that implements basic methods for a replay buffer."""

    def __init__(self, capacity, seed, algo='ddpg', gamma=0.99, tau=0.95):
        """Initialized a list for timesteps."""
        random.seed(seed)
        self._buffer = []
        self.algo = algo
        self.gamma = gamma
        self.tau = tau

        self.capacity = capacity
        self.position = 0

    def __len__(self):
        """Length method.

    Returns:
      A length of the buffer.
    """
        return len(self._buffer)

    def flush(self):
        """Clear the replay buffer."""
        self._buffer = []

    def buffer(self):
        """Get access to protected buffer memory for debug."""
        return self._buffer

    def push_back(self, *args):
        """Pushes a timestep.
        Args: *args: see the definition of TimeStep.
        """
        if len(self._buffer) < self.capacity:
            self._buffer.append(None)
        self._buffer[self.position] = TimeStep(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=100):
        """Uniformly samples a batch of timesteps from the buffer.

    Args:
      batch_size: number of timesteps to sample.

    Returns:
      Returns a batch of timesteps.
    """
        return random.sample(self._buffer, batch_size)

    def get_average_reward(self):
        """Returns the average reward of all trajectories in the buffer.
    """
        reward = 0
        num_trajectories = 0
        for time_step in self._buffer:
            reward += time_step.reward[0]
            if time_step.done:
                num_trajectories += 1
        return reward / num_trajectories

    def add_absorbing_states(self, env):
        """Adds an absorbing state for every final state.

    The mask is defined as 1 is a mask for a non-final state, 0 for a
    final state and -1 for an absorbing state.

    Args:
      env: environments to add an absorbing state for.
    """
        prev_start = 0
        replay_len = len(self)
        for j in range(replay_len):
            # 在默认的最大轨迹步数内，若智能体非正常结束，则所有结束的状态都指定其下一状态为吸收态，
            # 对每个状态增加一个维度，正常状态增加维度的值为 0，吸收态增加维度的值为 1
            # if j - prev_start + 1 == env._max_episode_steps:
            #     print("j-prev_start:{},prev_start:{}".format(j - prev_start, prev_start))
            if self._buffer[j].done and j - prev_start + 1 < env._max_episode_steps:  # pylint: disable=protected-access
                next_obs = env.get_absorbing_state()
            else:
                next_obs = env.get_non_absorbing_state(self._buffer[j].next_obs)
            self._buffer[j] = TimeStep(
                env.get_non_absorbing_state(self._buffer[j].obs),
                self._buffer[j].action, next_obs, self._buffer[j].reward,
                self._buffer[j].mask, self._buffer[j].done,
                env.get_non_absorbing_state(self._buffer[j].obs),
                env.get_non_absorbing_state(self._buffer[j].obs))
            # 每当在默认最大轨迹步数(1000)内有非正常结束的状态，则增加一个吸收态的样本，
            # 吸收态的下一状态也是吸收态，动作全为 0, 奖赏为0.
            # 这一设定的目的是，人为增加信息，对于所有使得智能体可能到达吸收态的动作，减小Q网络对其的估值。
            if self._buffer[j].done:
                if j - prev_start + 1 < env._max_episode_steps:  # pylint: disable=protected-access
                    action = np.zeros(env.action_space.shape)
                    absorbing_state = env.get_absorbing_state()
                    # done=False is set to the absorbing state because it corresponds to
                    # a state where gym environments stopped an episode.
                    self.push_back(absorbing_state, action, absorbing_state, [0.0],
                                   [Mask.ABSORBING.value], False)
                prev_start = j + 1

    def subsample_trajectories(self, num_trajectories):
        """Subsamples trajectories in the replay buffer.

    Args:
      num_trajectories: number of trajectories to keep.
    Raises:
      ValueError: when the replay buffer contains not enough trajectories.
    """

        trajectories = []
        trajectory = []
        for timestep in self._buffer:
            trajectory.append(timestep)
            if timestep.done:
                trajectories.append(trajectory)
                trajectory = []
        if len(trajectories) < num_trajectories:
            raise ValueError('Not enough trajectories to subsample')
        # 用来连接多个迭代器,例如：["ABC", "DEF"] ---> A B C D E F
        # 此处的作用为将包含两个元素的列表拼接为一个元素
        subsampled_trajectories = random.sample(trajectories, num_trajectories)

        self._buffer = list(itertools.chain.from_iterable(subsampled_trajectories))

    def update_buffer(self, keys, values):
        for step, transition in enumerate(self._buffer):
            transition_dict = transition._asdict()
            for key, value in zip(keys, values[step]):
                transition_dict[key] = value
                self._buffer[step] = TimeStepAdv(**transition_dict)

    def combine(self, other_buffer, start_index=None, end_index=None):
        """Combines current replay buffer with a different one.

    Args:
      other_buffer: a replay buffer to combine with.
      start_index: index of first element from the other_buffer.
      end_index: index of last element from other_buffer.
    """
        self._buffer += other_buffer._buffer[start_index:end_index]  # pylint: disable=protected-access

    def subsample_transitions(self, subsampling_rate=20):
        """Subsamples trajectories in the replay buffer.

    Args:
      subsampling_rate: rate with which subsample trajectories.
    Raises:
      ValueError: when the replay buffer contains not enough trajectories.
    """
        subsampled_buffer = []
        i = 0
        offset = np.random.randint(0, subsampling_rate)

        for timestep in self._buffer:
            i += 1
            # Never remove the absorbing transitions from the list.
            if timestep.mask == Mask.ABSORBING.value or (
                    i + offset) % subsampling_rate == 0:
                subsampled_buffer.append(timestep)

            if timestep.done or timestep.mask == Mask.ABSORBING.value:
                i = 0
                offset = np.random.randint(0, subsampling_rate)

        self._buffer = subsampled_buffer

    def compute_normalized_advantages(self):
        batch = TimeStepAdv(*zip(*self._buffer))
        advantages = np.stack(batch.advantages).squeeze()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        print('normalized advantages: %s' % advantages[:100])
        print('returns : %s' % np.stack(batch.returns)[:100])
        print('value_preds : %s' % np.stack(batch.value_preds)[:100])
        keys = ['advantages']
        values = advantages.reshape(-1, 1)
        self.update_buffer(keys, values)

    def compute_returns_advantages(self, next_value_preds, use_gae=False):
        """Compute returns for trajectory."""

        logging.info('Computing returns and advantages...')

        # TODO(agrawalk): Add more tests and asserts.
        batch = TimeStepAdv(*zip(*self._buffer))
        reward = np.stack(batch.reward).squeeze()
        value_preds = np.stack(batch.value_preds).squeeze()
        returns = np.stack(batch.returns).squeeze()
        mask = np.stack(batch.mask).squeeze()
        # effective_traj_len = traj_len - 2
        # This takes into account:
        #   - the extra observation in buffer.
        #   - 0-indexing for the transitions.
        effective_traj_len = len(reward) - 2

        if use_gae:
            value_preds[-1] = next_value_preds
            gae = 0
            for step in range(effective_traj_len, -1, -1):
                delta = (reward[step] +
                         self.gamma * value_preds[step + 1] * mask[step] -
                         value_preds[step])
                gae = delta + self.gamma * self.tau * mask[step] * gae
                returns[step] = gae + value_preds[step]
        else:
            returns[-1] = next_value_preds
            for step in range(effective_traj_len, -1, -1):
                returns[step] = (reward[step] +
                                 self.gamma * returns[step + 1] * mask[step])

        advantages = returns - value_preds
        keys = ['value_preds', 'returns', 'advantages']
        values = [list(entry) for entry in zip(  # pylint: disable=g-complex-comprehension
            value_preds.reshape(-1, 1),
            returns.reshape(-1, 1),
            advantages.reshape(-1, 1))]
        self.update_buffer(keys, values)

        self._buffer = self._buffer[:-1]


class ExpertReplayBuffer(ReplayBuffer):
    def __init__(self, num_trajectories, expert_path, target_interval, seed, mi_num_traj):
        super(ExpertReplayBuffer, self).__init__(capacity=1000 * num_trajectories, seed=seed)
        '''
        num_trajectories = 40
        {"acs":ndarray:(40,1000,3), "done":ndarray:(40,1000), "lens":ndarray:(40), "mask":ndarray:(40, 1000),
         "obs":ndarray:(40,1000,11), "obs1":ndarray:(40,1000,11), "returns":ndarray:(40), "reward":ndarray:(40,1000)}
        '''
        self.expert_data = MuJoCoExpertData(expert_path=expert_path,
                                            train_fraction=1,
                                            traj_limitation=num_trajectories,
                                            target_interval=target_interval)
        self.subsample_trajectories(num_trajectories)
        self.MI_expert_sa = self.get_expert_demonstrations(mi_num_traj)

    def subsample_trajectories(self, num_trajectories):
        """Subsamples trajectories in the replay buffer.

    Args:
      num_trajectories: number of trajectories to keep.
    Raises:
      ValueError: when the replay buffer contains not enough trajectories.
    """

        for i in range(self.expert_data.state.shape[0]):
            for j in range(self.expert_data.state.shape[1]):
                state = self.expert_data.state[i, j, :]
                next_state = self.expert_data.next_state[i, j, :]
                action = self.expert_data.action[i, j, :]
                reward = self.expert_data.reward[i, j]
                mask = Mask.NOT_DONE.value  # 1.0
                done = self.expert_data.done[i, j]
                self.push_back(state, action, next_state, [reward], [mask], done, state, state)
        trajectories = []
        trajectory = []
        for timestep in self._buffer:
            trajectory.append(timestep)
            if timestep.done:
                trajectories.append(trajectory)
                trajectory = []
        if len(trajectories) < num_trajectories:
            raise ValueError('Not enough trajectories to subsample')
        # 用来连接多个迭代器,例如：["ABC", "DEF"] ---> A B C D E F
        # 此处的作用为将包含两个元素的列表拼接为一个元素
        self.subsampled_trajectories = random.sample(trajectories, num_trajectories)
        self._buffer = list(itertools.chain.from_iterable(self.subsampled_trajectories))

    def get_expert_demonstrations(self, num_trajectories):
        state_trajectories, action_trajectories = [], []
        state_trajectory, action_trajectory = [], []

        for trajectory in self.subsampled_trajectories:
            for timestep in trajectory:
                state_trajectory.append(timestep.obs)
                action_trajectory.append(timestep.action)
                if timestep.done:
                    state_trajectories.append(state_trajectory)
                    action_trajectories.append(action_trajectory)
                    state_trajectory, action_trajectory = [], []
        if len(state_trajectories) < num_trajectories:
            raise ValueError('Not enough trajectories to subsample')
        mi_expert_sa = np.concatenate([np.array(state_trajectories), np.array(action_trajectories)], axis=-1)

        return mi_expert_sa


class Mask(Enum):
    ABSORBING = -1.0
    DONE = 0.0
    NOT_DONE = 1.0
