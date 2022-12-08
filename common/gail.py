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

"""An implementation of GAIL with WGAN discriminator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import tensorflow.compat.v1 as tf
# from tensorflow.contrib import summary as contrib_summary
# from tensorflow.contrib.eager.python import tfe as contrib_eager_python_tfe
# from tensorflow.contrib.gan.python.losses.python import losses_impl as contrib_gan_python_losses_python_losses_impl
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.optim import rmsprop
from torch.optim import Adam


# Initialize neural network weights
def weights_init_(m):
    # torch.nn.init.normal_(m, mean=0, std=1)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.normal_(m.weight, mean=0, std=1)
        torch.nn.init.constant_(m.bias, 0)


class DiscriminatorNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DiscriminatorNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state_action):
        x1 = torch.tanh(self.linear1(state_action))
        x1 = torch.tanh(self.linear2(x1))
        x1 = self.linear3(x1)
        return x1


class GAIL(object):
    """
    Implementation of GAIL (https://arxiv.org/abs/1606.03476).
    Instead of the original GAN, it uses WGAN (https://arxiv.org/pdf/1704.00028).
    """

    def __init__(self, input_dim, subsampling_rate, lambd=10.0, gail_loss='airl', gail_lr=1e-3, device=None):
        """
        Initializes Discriminator networks and optimizers.
        Args:
           input_dim: size of the observation space.
           subsampling_rate: subsampling rate that was used for expert trajectories.
           lambd: gradient penalty coefficient for wgan.
           gail_loss: gail loss to use.
        """
        self.subsampling_rate = subsampling_rate  # 1000 / 50
        self.lambd = lambd
        self.gail_loss = gail_loss  # airl

        self.disc_step = 0
        self.device = device
        self.discriminator = DiscriminatorNet(input_dim).to(self.device)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=gail_lr)


    def update(self, batch, expert_batch):
        """Updates the WGAN potential function or GAN discriminator.
        Args:
           batch: A batch from training policy.
           expert_batch: A batch from the expert.
        """
        obs = torch.FloatTensor(batch.obs).to(self.device)
        expert_obs = torch.FloatTensor(expert_batch.obs).to(self.device)
        expert_mask = torch.FloatTensor(expert_batch.mask).to(self.device)

        # Since expert trajectories were resampled but no absorbing state,
        # statistics of the states changes,
        # we need to adjust weights accordingly.
        expert_mask = torch.maximum(torch.FloatTensor([0]).to(self.device),
                                    -expert_mask)  # -expert_mask中每一个元素与 0 相比谁大返回谁
        expert_weight = expert_mask / self.subsampling_rate + (1 - expert_mask)

        action = torch.from_numpy(np.stack(batch.action)).type(torch.float32).to(self.device)
        expert_action = torch.from_numpy(np.stack(expert_batch.action)).type(torch.float32).to(self.device)

        inputs = torch.cat([obs, action], -1)  # shape: batch_size * (state.shape+1 + action.shape)
        expert_inputs = torch.cat([expert_obs, expert_action], -1)

        # gradient penalty, 取专家样本与当前生成样本之间的点，并限制这些点的梯度更新小于1.
        alpha = torch.from_numpy(np.random.uniform(size=(inputs.size()[0], 1))).type(torch.float32).to(
            self.device)  # 从一个均匀分布的区域中随机采样,默认区间[0,1]
        gp_input = (alpha * inputs + (1 - alpha) * expert_inputs).requires_grad_()

        # Get the output of inputs and expert_inputs in discriminator
        output = self.discriminator(inputs)
        expert_output = self.discriminator(expert_inputs)
        gp_output = self.discriminator(gp_input)

        # Compute accuracy
        gen_accuracy = np.array(output.cpu() < 0.5).mean()
        expert_accuracy = np.array(expert_output.cpu() > 0.5).mean()

        gene_loss = sigmoid_cross_entropy_with_logits(output, 0.)
        expert_loss = sigmoid_cross_entropy_with_logits(expert_output, 1.)

        # Adopting gradient penalty trick
        # to restrict the second derivative of samples between expert samples and generated samples
        # 让神经网络中的权重变化小于 1
        # 为了计算二阶导数，这里一阶导数部分必须设置：create_graph = True, retain_graph = True
        grad = autograd.grad(gp_output.sum(), gp_input, create_graph=True, retain_graph=True)[0]
        grad_penalty = torch.pow(torch.norm(grad, dim=-1) - 1, 2).mean()


        # Optimize
        self.discriminator_optimizer.zero_grad()
        total_loss = gene_loss + expert_loss + self.lambd * grad_penalty  # wgan-gp
        total_loss.backward()
        self.discriminator_optimizer.step()
        # print("total_loss:{}".format(total_loss))
        return expert_loss.item(), gene_loss.item(), total_loss.item(), gen_accuracy, expert_accuracy

    def get_reward(self, obs, action):  # pylint: disable=unused-argument
        if self.gail_loss == 'airl':
            inputs = torch.cat([obs, action], dim=-1)
            return self.discriminator(inputs)
        else:
            inputs = torch.cat([obs, action], dim=-1)
            return -torch.log(1 - torch.sigmoid(self.discriminator(inputs)) + 1e-8)


# def modified_discriminator_loss(discriminator_real_outputs, discriminator_gen_outputs,
#                                 label_smoothing=0.25, real_weights=1.0, generated_weights=1.0,
#                                 loss_collection=ops.GraphKeys.LOSSES,
#                                 reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
#                                 add_summaries=False):
#     loss_on_real = sigmoid_cross_entropy_with_logits(discriminator_real_outputs, 1.)
#     loss_on_generated = sigmoid_cross_entropy_with_logits(discriminator_gen_outputs, 0.)
#     # real_weights=1.0, label_smoothing=0.0, generated_weights=1.0,
#     # reduction=losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
#     # loss_collection=ops.GraphKeys.LOSSES

def sigmoid_cross_entropy_with_logits(logits: torch.Tensor, label, weight: torch.Tensor = None):
    """
    args:
        logits:
        label:
        weight:
    return:
    """
    zeros = torch.zeros_like(logits)
    cond = (logits >= zeros)
    relu_logits = torch.where(cond, logits, zeros)  # torch.where(x > 0, x, y),如果condition满足，则返回值对应位置取x，否则取y
    neg_abs_logits = torch.where(cond, -logits, logits)
    res = (relu_logits - logits * label) + torch.log1p(torch.exp(neg_abs_logits))
    if weight is not None:
        res *= weight
    return res.mean()

'''
class Discriminator(tf.keras.Model):
    """Implementation of a discriminator network."""

    def __init__(self, input_dim):
        """Initializes a discriminator.

    Args:
       input_dim: size of the input space.
    """
        super(Discriminator, self).__init__()
        kernel_init = tf.keras.initializers.Orthogonal(gain=1.0)

        self.main = tf.keras.Sequential([
            tf.layers.Dense(
                units=256,
                input_shape=(input_dim,),
                activation='tanh',
                kernel_initializer=kernel_init),
            tf.layers.Dense(
                units=256, activation='tanh', kernel_initializer=kernel_init),
            tf.layers.Dense(units=1, kernel_initializer=kernel_init)
        ])

    def call(self, inputs):
        """Performs a forward pass given the inputs.

    Args:
      inputs: a batch of observations (tfe.Variable).

    Returns:
      Values of observations.
    """
        return self.main(inputs)


class GAIL1(object):
    """Implementation of GAIL (https://arxiv.org/abs/1606.03476).

  Instead of the original GAN, it uses WGAN (https://arxiv.org/pdf/1704.00028).
  """

    def __init__(self, input_dim, subsampling_rate, lambd=10.0, gail_loss='airl'):
        """Initializes actor, critic, target networks and optimizers.

    Args:
       input_dim: size of the observation space.
       subsampling_rate: subsampling rate that was used for expert trajectories.
       lambd: gradient penalty coefficient for wgan.
       gail_loss: gail loss to use.
    """

        self.subsampling_rate = subsampling_rate  # 1000 / 50
        self.lambd = lambd
        self.gail_loss = gail_loss  # airl

        with tf.variable_scope('discriminator'):
            self.disc_step = contrib_eager_python_tfe.Variable(
                0, dtype=tf.int64, name='step')
            self.discriminator = Discriminator(input_dim)
            self.discriminator_optimizer = tf.train.AdamOptimizer()
            self.discriminator_optimizer._create_slots(self.discriminator.variables)  # pylint: disable=protected-access

    def update(self, batch, expert_batch):
        """Updates the WGAN potential function or GAN discriminator.

    Args:
       batch: A batch from training policy.
       expert_batch: A batch from the expert.
    """
        obs = contrib_eager_python_tfe.Variable(
            np.stack(batch.obs).astype('float32'))
        expert_obs = contrib_eager_python_tfe.Variable(
            np.stack(expert_batch.obs).astype('float32'))

        expert_mask = contrib_eager_python_tfe.Variable(
            np.stack(expert_batch.mask).astype('float32'))

        # Since expert trajectories were resampled but no absorbing state,
        # statistics of the states changes,
        # we need to adjust weights accordingly.
        expert_mask = tf.maximum(0, -expert_mask)  # -expert_mask中每一个元素与 0 相比谁大返回谁
        expert_weight = expert_mask / self.subsampling_rate + (1 - expert_mask)

        action = contrib_eager_python_tfe.Variable(
            np.stack(batch.action).astype('float32'))
        expert_action = contrib_eager_python_tfe.Variable(
            np.stack(expert_batch.action).astype('float32'))

        inputs = tf.concat([obs, action], -1)
        expert_inputs = tf.concat([expert_obs, expert_action], -1)

        # Avoid using tensorflow random functions since it's impossible to get
        # the state of the random number generator used by TensorFlow.
        alpha = np.random.uniform(size=(inputs.get_shape()[0], 1))  # 从一个均匀分布的区域中随机采样,默认区间[0,1]
        alpha = contrib_eager_python_tfe.Variable(alpha.astype('float32'))
        inter = alpha * inputs + (1 - alpha) * expert_inputs

        with tf.GradientTape() as tape:
            output = self.discriminator(inputs)
            expert_output = self.discriminator(expert_inputs)

            with contrib_summary.record_summaries_every_n_global_steps(
                    100, self.disc_step):
                gan_loss = contrib_gan_python_losses_python_losses_impl.modified_discriminator_loss(
                    expert_output,
                    output,
                    label_smoothing=0.0,
                    real_weights=expert_weight)
                contrib_summary.scalar(
                    'discriminator/expert_output',
                    tf.reduce_mean(expert_output),
                    step=self.disc_step)
                contrib_summary.scalar(
                    'discriminator/policy_output',
                    tf.reduce_mean(output),
                    step=self.disc_step)

            with tf.GradientTape() as tape2:
                tape2.watch(inter)
                output = self.discriminator(inter)
                grad = tape2.gradient(output, [inter])[0]

            grad_penalty = tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - 1, 2))

            loss = gan_loss + self.lambd * grad_penalty  # wgan-gp

        with contrib_summary.record_summaries_every_n_global_steps(
                100, self.disc_step):
            contrib_summary.scalar(
                'discriminator/grad_penalty', grad_penalty, step=self.disc_step)

        with contrib_summary.record_summaries_every_n_global_steps(
                100, self.disc_step):
            contrib_summary.scalar(
                'discriminator/loss', gan_loss, step=self.disc_step)

        grads = tape.gradient(loss, self.discriminator.variables)

        self.discriminator_optimizer.apply_gradients(
            zip(grads, self.discriminator.variables), global_step=self.disc_step)

    def get_reward(self, obs, action, next_obs):  # pylint: disable=unused-argument
        if self.gail_loss == 'airl':
            inputs = tf.concat([obs, action], -1)
            return self.discriminator(inputs)
        else:
            inputs = tf.concat([obs, action], -1)
            return -tf.log(1 - tf.nn.sigmoid(self.discriminator(inputs)) + 1e-8)

    @property
    def variables(self):
        """Returns all variables including optimizer variables.

    Returns:
      A dictionary of all variables that are defined in the model.
      variables.
    """
        disc_vars = (
                self.discriminator.variables + self.discriminator_optimizer.variables()
                + [self.disc_step])

        return disc_vars
'''