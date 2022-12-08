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

"""Implementations of imitation learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tqdm import tqdm
from mi_model import data_utils
from mi_model import gail
from mi_model import twin_sac
from mi_model import mutual_information
from mi_model import wrappers


def get_flag():
    FLAGS = flags.FLAGS

    flags.DEFINE_string('env_name', 'Hopper-v2',
                        'Environment for training/evaluation.')
    flags.DEFINE_integer('seed', 42, 'Fixed random seed for training.')
    flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
    flags.DEFINE_integer('actor_update_freq', 1, 'Update actor every N steps.')
    flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
    flags.DEFINE_float('replay_regularization', 0.1, 'Amount of replay mixing.')
    flags.DEFINE_float('nu_lr', 1e-3, 'nu network learning rate.')
    flags.DEFINE_float('actor_lr', 1e-5, 'Actor learning rate.')
    flags.DEFINE_float('critic_lr', 1e-3, 'Critic learning rate.')
    flags.DEFINE_float('sac_alpha', 0.1, 'SAC temperature.')
    flags.DEFINE_float('tau', 0.005,
                       'Soft update coefficient for the target network.')
    flags.DEFINE_integer('hidden_size', 256, 'Hidden size.')
    flags.DEFINE_integer('updates_per_step', 5, 'Updates per time step.')
    flags.DEFINE_integer('max_timesteps', int(1e5), 'Max timesteps to train.')
    flags.DEFINE_integer('num_trajectories', 1, 'Number of trajectories to use.')
    flags.DEFINE_integer('num_random_actions', int(2e3),
                         'Fill replay buffer with N random actions.')
    flags.DEFINE_integer('start_training_timesteps', int(1e3),
                         'Start training when replay buffer contains N timesteps.')
    flags.DEFINE_string('save_dir', './save/', 'Directory to save results to.')
    flags.DEFINE_boolean('learn_alpha', True,
                         'Whether to learn temperature for SAC.')
    flags.DEFINE_boolean('normalize_states', True,
                         'Normalize states using expert stats.')
    flags.DEFINE_integer('log_interval', int(1e3), 'Log every N timesteps.')
    flags.DEFINE_integer('eval_interval', int(1e3), 'Evaluate every N timesteps.')
    flags.DEFINE_enum('algo', 'value_dice', ['bc', 'dac', 'value_dice'],
                      'Algorithm to use to compute occupancy ration.')
    flags.DEFINE_integer('absorbing_per_episode', 10,
                         'A number of absorbing states per episode to add.')
    return FLAGS


def _update_pbar_msg(pbar, total_timesteps, FLAGS):
    """Update the progress bar with the current training phase."""
    if total_timesteps < FLAGS.start_training_timesteps:
        msg = 'not training'
    else:
        msg = 'training'
    if total_timesteps < FLAGS.num_random_actions:
        msg += ' rand acts'
    else:
        msg += ' policy acts'
    if pbar.desc != msg:
        pbar.set_description(msg)


def add_samples_to_replay_buffer(replay_buffer, obs, action, next_obs, index):
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
                             np.array([[1.0]]).astype(np.float32)),
                            np.array([index.astype(np.float32)]))


def evaluate(actor, env, num_episodes=10):
    """Evaluates the policy.

  Args:
    actor: A policy to evaluate.
    env: Environment to evaluate the policy on.
    num_episodes: A number of episodes to average the policy on.

  Returns:
    Averaged reward and a total number of steps.
  """
    total_timesteps = 0
    total_returns = 0

    for _ in range(num_episodes):
        state = env.reset()
        state = env.get_normalize_state(state)
        done = False
        while not done:
            action, _, _ = actor(np.array([state]))
            action = action[0].numpy()

            next_state, reward, done, _ = env.step(action)
            next_state = env.get_normalize_state(next_state)

            total_returns += reward
            total_timesteps += 1
            state = next_state

    print("---------------------------------------")
    print(f"MI:{num_episodes} episodes, Average_return: {total_returns / num_episodes:.3f}, "
          f"Average_steps:{total_timesteps / num_episodes:.3f}")
    # print("---------------------------------------")

    return total_returns / num_episodes, total_timesteps / num_episodes


def value_dice_prepare(FLAGS, expert_path):
    tf.enable_v2_behavior()

    tf.random.set_seed(FLAGS.VD_seed)
    np.random.seed(FLAGS.VD_seed)
    random.seed(FLAGS.VD_seed)

    # filename = os.path.join(FLAGS.VD_expert_dir, FLAGS.VD_env_name + '.npz')
    (expert_states, expert_actions, expert_next_states,
     expert_dones, expert_index) = data_utils.load_expert_data(expert_path, num_traj=40)

    (expert_states, expert_actions, expert_next_states,
     expert_dones, expert_index) = data_utils.subsample_trajectories(expert_states,
                                                                     expert_actions,
                                                                     expert_next_states,
                                                                     expert_dones,
                                                                     FLAGS.VD_num_trajectories,
                                                                     expert_index)
    print('# of demonstraions: {}'.format(expert_states.shape[0]))

    if FLAGS.VD_normalize_states:
        shift = -np.mean(expert_states, 0)
        scale = 1.0 / (np.std(expert_states, 0) + 1e-3)
        expert_states = (expert_states + shift) * scale
        expert_next_states = (expert_next_states + shift) * scale
    else:
        shift = None
        scale = None

    env = wrappers.create_il_env(FLAGS.VD_env_name, FLAGS.VD_seed, shift, scale)

    eval_env = wrappers.create_il_env(FLAGS.VD_env_name, FLAGS.VD_seed + 1, shift,
                                      scale)

    unwrap_env = env

    while hasattr(unwrap_env, 'env'):
        if isinstance(unwrap_env, wrappers.NormalizeBoxActionWrapper):
            expert_actions = unwrap_env.reverse_action(expert_actions)
            break
        unwrap_env = unwrap_env.env

    (expert_states, expert_actions, expert_next_states,
     expert_dones, expert_index) = data_utils.add_absorbing_states(expert_states,
                                                                   expert_actions,
                                                                   expert_next_states,
                                                                   expert_dones, env, expert_index)

    spec = (
        tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32, 'observation'),
        tensor_spec.TensorSpec([env.action_space.shape[0]], tf.float32, 'action'),
        tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32, 'next_observation'),
        tensor_spec.TensorSpec([1], tf.float32, 'reward'),
        tensor_spec.TensorSpec([1], tf.float32, 'mask'),
        tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32, 'target0'),
        tensor_spec.TensorSpec([env.observation_space.shape[0]], tf.float32, 'target1'))

    # We need to store at most twice more transition due to
    # an extra absorbing to itself transition.
    # replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    #     spec, batch_size=1, max_length=FLAGS.max_timesteps * 2)
    #
    # for i in range(expert_states.shape[0]):
    #     # Overwrite rewards for safety. We still have to add them to the replay
    #     # buffer to maintain the same interface. Also always use a zero mask
    #     # since we need to always bootstrap for imitation learning.
    #     add_samples_to_replay_buffer(replay_buffer, expert_states[i],
    #                                  expert_actions[i], expert_next_states[i], expert_index[i])

    policy_replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        spec, batch_size=1, max_length=FLAGS.VD_max_timesteps * 2)

    policy_replay_buffer_iter = iter(
        policy_replay_buffer.as_dataset(
            sample_batch_size=FLAGS.VD_sample_batch_size))

    # print("expert_states.shape, env.get_absorbing_state().shape",
    #       expert_states.shape, np.reshape(env.get_absorbing_state(), [1, -1]).shape)

    expert_states_instance = np.concatenate([expert_states,
                                             np.reshape(env.get_absorbing_state(), [1, -1])], axis=0)
    expert_states = tf.Variable(expert_states, dtype=tf.float32)
    expert_actions = tf.Variable(expert_actions, dtype=tf.float32)
    expert_next_states = tf.Variable(expert_next_states, dtype=tf.float32)
    expert_dones = tf.Variable(expert_dones, dtype=tf.float32)
    expert_index = tf.Variable(expert_index, dtype=tf.float32)

    expert_dataset = tf.data.Dataset.from_tensor_slices(
        (expert_states, expert_actions, expert_next_states, expert_index))
    expert_dataset = expert_dataset.repeat().shuffle(
        expert_states.shape[0]).batch(
        FLAGS.VD_sample_batch_size, drop_remainder=True)

    expert_dataset_iter = iter(expert_dataset)

    # (expert_states1, expert_actions1,
    #  expert_next_states1, expert_index1) = expert_dataset_iter.get_next()

    imitator = mutual_information.MiModel(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        nu_lr=FLAGS.VD_nu_lr,
        actor_lr=FLAGS.VD_actor_lr,
        alpha_init=FLAGS.VD_sac_alpha,
        hidden_size=FLAGS.VD_hidden_size,
        log_interval=FLAGS.VD_log_interval)

    return imitator, expert_dataset_iter, policy_replay_buffer_iter, env, eval_env, \
           policy_replay_buffer, expert_states_instance


def update_value_dice(FLAGS, imitator, env, eval_env,
                      expert_dataset_iter, policy_replay_buffer_iter,
                      policy_replay_buffer):
    episode_return = 0
    episode_timesteps = 0
    done = True

    total_timesteps = 0
    previous_time = time.time()

    eval_returns = []
    with tqdm(total=FLAGS.max_timesteps, desc='') as pbar:
        while total_timesteps < FLAGS.max_timesteps:
            _update_pbar_msg(pbar, total_timesteps, FLAGS)

            if total_timesteps % FLAGS.eval_interval == 0:
                logging.info('Performing policy eval.')
                average_returns, evaluation_timesteps = evaluate(imitator.actor, eval_env)
                logging.info('Eval: ave returns=%f, ave episode length=%f',
                             average_returns, evaluation_timesteps)

                obs = env.reset()
                episode_return = 0
                episode_timesteps = 0
                previous_time = time.time()

            if total_timesteps < FLAGS.num_random_actions:
                action = env.action_space.sample()
            else:
                mean_action, _, _ = imitator.actor(np.array([obs]))
                action = mean_action[0].numpy()
                action = (action + np.random.normal(
                    0, 0.1, size=action.shape)).clip(-1, 1)

            next_obs, reward, done, _ = env.step(action)

            # done caused by episode truncation.
            truncated_done = done and episode_timesteps + 1 == env._max_episode_steps  # pylint: disable=protected-access

            if done and not truncated_done:
                next_obs = env.get_absorbing_state()

            # Overwrite rewards for safety. We still have to add them to the replay
            # buffer to maintain the same interface. Also always use a zero mask
            # since we need to always bootstrap for imitation learning.

            add_samples_to_replay_buffer(policy_replay_buffer, obs, action, next_obs)
            if done and not truncated_done:
                # Add several absobrsing states to absorbing states transitions.
                for abs_i in range(FLAGS.absorbing_per_episode):
                    if abs_i + episode_timesteps < env._max_episode_steps:  # pylint: disable=protected-access
                        obs = env.get_absorbing_state()
                        action = env.action_space.sample()
                        next_obs = env.get_absorbing_state()

                        add_samples_to_replay_buffer(policy_replay_buffer, obs, action,
                                                     next_obs)

            episode_return += reward
            episode_timesteps += 1
            total_timesteps += 1
            pbar.update(1)

            obs = next_obs

            if total_timesteps >= FLAGS.start_training_timesteps:
                for _ in range(FLAGS.updates_per_step):
                    imitator.update(
                        expert_dataset_iter,
                        policy_replay_buffer_iter,
                        FLAGS.discount,
                        replay_regularization=FLAGS.replay_regularization)
    return average_returns, imitator, env, eval_env, expert_dataset_iter, \
           policy_replay_buffer_iter, policy_replay_buffer

#
# if __name__ == '__main__':
#     app.run(main())
