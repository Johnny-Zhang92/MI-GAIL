import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow.compat.v2 as tf
import torch.autograd as autograd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    # def forward(self, state, action):
    #     sa = torch.cat([state, action], 1)

    def forward(self, sa):
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class DACCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DACCritic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # def forward(self, sa):

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            actor_lr=1e-3,
            critic_lr=1e-3,
            expl_noise=0.1,
            device=None,
            learn_absorbing=True,
            shift=None, scale=None,
            critic_for_expert=None,
            critic_for_no_expert=None):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        if critic_for_expert:
            self.critic = Critic(state_dim, action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
        elif critic_for_no_expert:
            self.critic = DACCritic(state_dim, action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
        else:
            raise ["Please specify which algorithm you want to train? MI-GAIL or TD3?"]

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expl_noise = expl_noise
        self.action_dim = action_dim
        self.actor_step = 0
        self.critic_step = 0
        self.learn_absorbing = learn_absorbing

        self.total_it = 0
        self.device = device
        self.actor_loss, self.critic_loss = torch.tensor([0]), torch.tensor([0])
        self.bc_loss_function = nn.MSELoss()
        self.shift = torch.FloatTensor(shift).to(device)
        self.scale = torch.FloatTensor(scale).to(device)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()  # 返回一个折叠成一维的数组,该函数只能适用于numpy对象

    def select_noised_action(self, state):
        action = self.select_action(np.array(state))
        noise = np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
        noise_action = (action + noise).clip(-self.max_action, self.max_action)
        return noise_action

    def change_learning_rate(self, total_time_steps, initial_actor_lr):
        if total_time_steps % 100000 == 0:
            learning_rate = initial_actor_lr * pow(0.5, total_time_steps // 100000)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
            print('Generator actor learning rate: ', round(learning_rate, 10))
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

    def train(self, batch_samples, expert_samples, dis_reward=None,
              mi_model=None, total_number_step=None, train_by_mi_reward=False, mi_weight=None):
        self.total_it += 1

        # Get samples
        state = torch.FloatTensor(batch_samples.obs).to(self.device)
        action = torch.FloatTensor(batch_samples.action).to(self.device)
        next_state = torch.FloatTensor(batch_samples.next_obs).to(self.device)
        mask = torch.FloatTensor(batch_samples.mask).to(self.device)
        target0 = torch.FloatTensor(batch_samples.target0).to(self.device)
        # 'obs', 'action', 'next_obs', 'reward', 'mask', 'done'
        expert_state = torch.FloatTensor(expert_samples.obs).to(self.device)
        expert_action = torch.FloatTensor(expert_samples.action).to(self.device)
        expert_next_state = torch.FloatTensor(expert_samples.next_obs).to(self.device)
        expert_mask = torch.FloatTensor(expert_samples.mask).to(self.device)

        total_state = torch.cat([state, expert_state])
        total_action = torch.cat([action, expert_action])
        total_next_state = torch.cat([next_state, expert_next_state])
        total_mask = torch.cat([mask, expert_mask])

        # middle point samples
        # gradient penalty, 取专家样本与当前生成样本之间的点，并限制这些点的梯度更新小于1.
        alpha = torch.from_numpy(np.random.uniform(size=(state.size()[0], 1))).type(torch.float32).to(
            self.device)  # 从一个均匀分布的区域中随机采样,默认区间[0,1]
        gp_state = (alpha * expert_state + (1 - alpha) * state).requires_grad_()
        gp_next_state = (alpha * expert_next_state + (1 - alpha) * next_state).requires_grad_()

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            expert_next_action = (
                    self.actor_target(expert_next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            gp_next_action = (
                    self.actor_target(gp_next_state) + noise
            ).clamp(-self.max_action, self.max_action)
            gp_action = (
                    self.actor_target(gp_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Get reward
            mi_reward = torch.tensor([0]).to(torch.float32)
            if dis_reward is None:
                reward = torch.FloatTensor(batch_samples.reward).to(self.device)
            else:
                rb_reward = dis_reward(state, action)
                expert_reward = dis_reward(expert_state, expert_action)
                if train_by_mi_reward:
                    mi_reward = self.get_mi_reward(batch_samples=batch_samples,
                                                   next_action=next_action,
                                                   mi_model=mi_model)
                    # mi_weight = 0.9999**total_number_step
                    rb_reward = rb_reward + mi_weight * mi_reward
                    # reward = dis_reward
                else:
                    reward = dis_reward
                # reward = dis_reward

            # Compute the target Q value
            if self.learn_absorbing:
                # Starting from the goal state we can execute only non-actions.
                # the mask of absorbing state is 1.
                a_mask = torch.maximum(torch.tensor([0], device=self.device), mask)
                expert_a_mask = torch.maximum(torch.tensor([0], device=self.device), expert_mask)

                # change the action of absorbing state to all zero
                next_state_action = torch.cat([next_state, next_action * a_mask], 1)
                target_q1, target_q2 = self.critic_target(sa=next_state_action)
                expert_next_state_action = torch.cat([expert_next_state, expert_next_action * expert_a_mask], 1)
                expert_target_q1, expert_target_q2 = self.critic_target(sa=expert_next_state_action)

                # torch.min返回两个Tensor，一个是最小值Tensor，另一个是下标Tensor
                next_q = torch.min(torch.cat([target_q1, target_q2], -1), -1, keepdim=True)[0]
                expert_next_q = torch.min(torch.cat([expert_target_q1, expert_target_q2], -1), -1, keepdim=True)[0]

                target_q = rb_reward + self.discount * next_q
                expert_target_q = expert_reward + self.discount * expert_next_q
            else:
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + mask * self.discount * target_q

        # Get current Q estimates
        state_action = torch.cat([state, action], 1)
        current_q1, current_q2 = self.critic(sa=state_action)
        expert_state_action = torch.cat([expert_state, expert_action], 1)
        expert_current_q1, expert_current_q2 = self.critic(sa=expert_state_action)

        gp_state_action = torch.cat([gp_state, gp_action], 1)
        gp_current_q1, gp_current_q2 = self.critic(sa=gp_state_action)
        grad1 = autograd.grad(gp_current_q1.mean(), gp_state_action, create_graph=True, retain_graph=True)[0]
        grad_penalty1 = torch.pow(torch.norm(grad1, dim=-1) - 1, 2).mean()
        grad2 = autograd.grad(gp_current_q2.mean(), gp_state_action, create_graph=True, retain_graph=True)[0]
        grad_penalty2 = torch.pow(torch.norm(grad2, dim=-1) - 1, 2).mean()

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        expert_critic_loss = F.mse_loss(expert_current_q1, expert_target_q) + F.mse_loss(expert_current_q2,
                                                                                         expert_target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        total_critic_loss = critic_loss + expert_critic_loss + 10.0 * grad_penalty1 + 10.0 * grad_penalty2
        # total_critic_loss = critic_loss + 10.0 * grad_penalty1 + 10.0 * grad_penalty2
        total_critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_step += 1
        self.critic_loss = critic_loss.item()
        # Delayed policy updates

        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            state_normal = (state + self.shift) * self.scale
            # print(state_normal.cpu().detach())
            # action_mi, _, _ = mi_model.actor(np.array(state_normal.cpu().detach()))
            # action_mi = torch.FloatTensor(action_mi.numpy()).detach().to(self.device)
            # # print("action", action)
            action_td3 = self.actor(state)
            # bc_loss = self.bc_loss_function(action_td3, action_mi)

            pred_q = -self.critic.Q1(state, action_td3).mean()
            expert_pred_q = -self.critic.Q1(expert_state, self.actor(expert_state)).mean()
            # pred_mi = -mi_model.nu_net(state, action_td3, target0)

            gp_pred_q = -self.critic.Q1(gp_state, self.actor(gp_state)).mean()
            grad1 = autograd.grad(gp_pred_q.mean(), gp_state, create_graph=True, retain_graph=True)[0]
            grad_penalty_for_actor = torch.pow(torch.norm(grad1, dim=-1) - 1, 2).mean()

            if self.learn_absorbing:
                # Don't update the actor for absorbing states.
                # And skip update if all states are absorbing.
                a_mask = 1.0 - torch.maximum(torch.tensor([0], device=self.device), -mask)  # boardcast
                expert_a_mask = 1.0 - torch.maximum(torch.tensor([0], device=self.device), -expert_mask)  # boardcast
                if torch.sum(a_mask) < 1e-8:
                    return torch.tensor([0], device=self.device)
                # actor_loss = 0.3 * torch.sum(pred_q * a_mask) / torch.sum(a_mask) + 0.7 * bc_loss
                bc_weight = 0.7 * (0.99998 ** total_number_step)
                actor_loss = torch.sum(pred_q * a_mask) / torch.sum(a_mask)
                expert_actor_loss = torch.sum(expert_pred_q * a_mask) / torch.sum(expert_a_mask)
                # actor_loss = (1 - bc_weight) * torch.sum(pred_q * a_mask) / torch.sum(a_mask) \
                #              + bc_weight * bc_loss
                bc_lr = pow(0.5, (total_number_step // 10000) / 10)
                # actor_loss = 0.3 * torch.sum(pred_q * a_mask) / torch.sum(a_mask) \
                #              + bc_lr * bc_loss

                # tf_state = tf.Variable(batch_samples.obs, dtype=tf.float32)
                # tf_action = tf.Variable(batch_samples.action, dtype=tf.float32)
                # tf_target0 = tf.Variable(batch_samples.target0, dtype=tf.float32)
                #
                # tf_next_state = tf.Variable(batch_samples.next_obs, dtype=tf.float32)
                # tf_next_action = tf.Variable(next_action.cpu().detach().numpy(), dtype=tf.float32)
                # tf_target1 = tf.Variable(batch_samples.target1, dtype=tf.float32)

                # dis_reward = dis_reward(state, action)
                # mi_inputs = tf.concat([tf_state, tf_action, tf_target0], 1)
                # mi_next_inputs = tf.concat([tf_next_state, tf_next_action, tf_target1], 1)
                # out_put = mi_model.nu_net(mi_inputs)
                # next_out_put = mi_model.nu_net(mi_next_inputs)
                # mi_reward = out_put - 0.99 * next_out_put
                # mi_reward = torch.FloatTensor(mi_reward.numpy()).to(self.device) * -1e-1
            else:
                actor_loss = torch.mean(pred_q)
                # actor_loss = torch.mean(pred_q) + bc_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            total_actor_loss = actor_loss + expert_actor_loss + 1.0 * grad_penalty_for_actor
            # total_actor_loss = actor_loss + expert_actor_loss
            # total_actor_loss = actor_loss + 1.0 * grad_penalty_for_actor
            total_actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_step += 1
            self.actor_loss = actor_loss.item()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # return self.critic_loss, self.actor_loss, mi_reward.mean().item(), reward.mean().item()
        return self.critic_loss, self.actor_loss, mi_reward.mean().item(), rb_reward.mean().item()

    def get_mi_reward(self, batch_samples, next_action, mi_model):
        tf_state = tf.Variable(batch_samples.obs, dtype=tf.float32)
        tf_action = tf.Variable(batch_samples.action, dtype=tf.float32)
        tf_target0 = tf.Variable(batch_samples.target0, dtype=tf.float32)

        tf_next_state = tf.Variable(batch_samples.next_obs, dtype=tf.float32)
        tf_next_action = tf.Variable(next_action.cpu().detach().numpy(), dtype=tf.float32)
        tf_target1 = tf.Variable(batch_samples.target1, dtype=tf.float32)

        mi_inputs = tf.concat([tf_state, tf_action, tf_target0], 1)
        mi_next_inputs = tf.concat([tf_next_state, tf_next_action, tf_target1], 1)
        out_put = mi_model.nu_net(mi_inputs)
        next_out_put = mi_model.nu_net(mi_next_inputs)
        mi_reward = out_put - 0.99 * next_out_put
        mi_reward = np.exp(mi_reward.numpy())
        mi_reward = torch.FloatTensor(mi_reward).to(self.device)

        condition = (mi_reward <= 1.0622)
        # torch.where(x > 0, x, y),如果condition满足，则返回值对应位置取x，否则取y
        mi_reward = torch.where(condition, 1., 0.)
        return mi_reward

    def old_train(self, batch_samples, dis_reward=None, mi_model=None, total_number_step=None):
        self.total_it += 1

        # Get samples
        state = torch.FloatTensor(batch_samples.obs).to(self.device)
        action = torch.FloatTensor(batch_samples.action).to(self.device)
        next_state = torch.FloatTensor(batch_samples.next_obs).to(self.device)
        mask = torch.FloatTensor(batch_samples.mask).to(self.device)
        target0 = torch.FloatTensor(batch_samples.target0).to(self.device)
        # 'obs', 'action', 'next_obs', 'reward', 'mask', 'done'

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Get reward
            mi_reward = torch.tensor([0]).to(torch.float32)
            if dis_reward is None:
                reward = torch.FloatTensor(batch_samples.reward).to(self.device)
            else:
                # tf_state = tf.Variable(batch_samples.obs, dtype=tf.float32)
                # tf_action = tf.Variable(batch_samples.action, dtype=tf.float32)
                # tf_target0 = tf.Variable(batch_samples.target0, dtype=tf.float32)
                #
                # tf_next_state = tf.Variable(batch_samples.next_obs, dtype=tf.float32)
                # tf_next_action = tf.Variable(next_action.cpu().detach().numpy(), dtype=tf.float32)
                # tf_target1 = tf.Variable(batch_samples.target1, dtype=tf.float32)
                #
                dis_reward = dis_reward(state, action)
                # mi_inputs = tf.concat([tf_state, tf_action, tf_target0], 1)
                # mi_next_inputs = tf.concat([tf_next_state, tf_next_action, tf_target1], 1)
                # out_put = mi_model.nu_net(mi_inputs)
                # next_out_put = mi_model.nu_net(mi_next_inputs)
                # mi_reward = out_put - 0.99 * next_out_put
                # mi_reward = np.exp(mi_reward.numpy())
                # mi_reward = torch.FloatTensor(mi_reward).to(self.device)
                # reward = 1.0 * dis_reward - 0.1 * mi_reward
                reward = dis_reward

            # Compute the target Q value
            if self.learn_absorbing:
                # Starting from the goal state we can execute only non-actions.
                # the mask of absorbing state is 1.
                a_mask = torch.maximum(torch.tensor([0], device=self.device), mask)
                # change the action of absorbing state to all zero
                target_q1, target_q2 = self.critic_target(next_state, next_action * a_mask)
                # torch.min返回两个Tensor，一个是最小值Tensor，另一个是下标Tensor
                next_q = torch.min(torch.cat([target_q1, target_q2], -1), -1, keepdim=True)[0]
                target_q = reward + self.discount * next_q
            else:
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + mask * self.discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_step += 1
        self.critic_loss = critic_loss.item()
        # Delayed policy updates

        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            state_normal = (state + self.shift) * self.scale
            # print(state_normal.cpu().detach())
            action_mi, _, _ = mi_model.actor(np.array(state_normal.cpu().detach()))
            action_mi = torch.FloatTensor(action_mi.numpy()).detach().to(self.device)
            # print("action", action)
            action_td3 = self.actor(state)
            bc_loss = self.bc_loss_function(action_td3, action_mi)

            pred_q = -self.critic.Q1(state, action_td3).mean()
            # pred_mi = -mi_model.nu_net(state, action_td3, target0)

            if self.learn_absorbing:
                # Don't update the actor for absorbing states.
                # And skip update if all states are absorbing.
                a_mask = 1.0 - torch.maximum(torch.tensor([0], device=self.device), -mask)  # boardcast
                if torch.sum(a_mask) < 1e-8:
                    return torch.tensor([0], device=self.device)
                # actor_loss = 0.3 * torch.sum(pred_q * a_mask) / torch.sum(a_mask) + 0.7 * bc_loss
                bc_weight = 0.7 * (0.99998 ** total_number_step)
                actor_loss = torch.sum(pred_q * a_mask) / torch.sum(a_mask)
                # actor_loss = (1 - bc_weight) * torch.sum(pred_q * a_mask) / torch.sum(a_mask) \
                #              + bc_weight * bc_loss
                bc_lr = pow(0.5, (total_number_step // 10000) / 10)
                # actor_loss = 0.3 * torch.sum(pred_q * a_mask) / torch.sum(a_mask) \
                #              + bc_lr * bc_loss

                # tf_state = tf.Variable(batch_samples.obs, dtype=tf.float32)
                # tf_action = tf.Variable(batch_samples.action, dtype=tf.float32)
                # tf_target0 = tf.Variable(batch_samples.target0, dtype=tf.float32)
                #
                # tf_next_state = tf.Variable(batch_samples.next_obs, dtype=tf.float32)
                # tf_next_action = tf.Variable(next_action.cpu().detach().numpy(), dtype=tf.float32)
                # tf_target1 = tf.Variable(batch_samples.target1, dtype=tf.float32)

                # dis_reward = dis_reward(state, action)
                # mi_inputs = tf.concat([tf_state, tf_action, tf_target0], 1)
                # mi_next_inputs = tf.concat([tf_next_state, tf_next_action, tf_target1], 1)
                # out_put = mi_model.nu_net(mi_inputs)
                # next_out_put = mi_model.nu_net(mi_next_inputs)
                # mi_reward = out_put - 0.99 * next_out_put
                # mi_reward = torch.FloatTensor(mi_reward.numpy()).to(self.device) * -1e-1
            else:
                actor_loss = torch.mean(pred_q)
                # actor_loss = torch.mean(pred_q) + bc_loss

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_step += 1
            self.actor_loss = actor_loss.item()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # return self.critic_loss, self.actor_loss, mi_reward.mean().item(), reward.mean().item()
        return self.critic_loss, self.actor_loss, mi_reward.mean().item(), reward.mean().item()

    def train_by_mi(self, batch_samples, dis_reward=None, mi_model=None, total_number_step=None):
        # Get samples
        state = torch.FloatTensor(batch_samples.obs).to(self.device)
        mask = torch.FloatTensor(batch_samples.mask).to(self.device)

        # Compute actor losse
        state_normal = (state + self.shift) * self.scale
        action_mi, _, _ = mi_model.actor(np.array(state_normal.cpu().detach()))
        action_mi = torch.FloatTensor(action_mi.numpy()).detach().to(self.device)
        action_td3 = self.actor(state)
        bc_loss = self.bc_loss_function(action_td3, action_mi)

        a_mask = 1.0 - torch.maximum(torch.tensor([0], device=self.device), -mask)  # boardcast
        if torch.sum(a_mask) < 1e-8:
            return torch.tensor([0], device=self.device)
        bc_lr = pow(0.5, ((total_number_step // 1000) + 10) / 100)
        actor_loss = bc_lr * bc_loss
        # actor_loss = 0.8 * bc_loss

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_loss = actor_loss.item()

        # Update the frozen target models
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
