# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""A wrapper that scales and shifts observations."""
import gym
import numpy as np


class NormalizeStateWrapper(gym.ObservationWrapper):
    """Wraps an environment to shift and scale observations.
  """

    def __init__(self, env, shift, scale):
        super(NormalizeStateWrapper, self).__init__(env)
        self.shift = shift
        self.scale = scale
        self.shift1 = np.concatenate([self.shift, [0]], axis=0)
        self.scale1 = np.concatenate([self.scale, [0]], axis=0)

    def observation(self, observation):
        # return (observation + self.shift) * self.scale  # 返回归一化的状态
        return observation  # 返回正常的状态

    def get_normalize_state(self, obs):
        obs = (obs + self.shift1) * self.scale1
        # obs = np.concatenate([obs, [0]], -1)
        return obs


    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps  # pylint: disable=protected-access
