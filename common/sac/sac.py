import os
import torch
import torch.nn.functional as F
import torch.autograd as autograd
import tensorflow.compat.v2 as tf
import numpy as np
from torch.optim import Adam
from common.sac.utils import soft_update, hard_update
from common.sac.model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.SAC_gamma
        self.tau = args.SAC_tau
        self.alpha = args.SAC_alpha

        self.policy_type = args.SAC_policy
        self.target_update_interval = args.SAC_target_update_interval
        self.automatic_entropy_tuning = args.SAC_automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.SAC_hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.SAC_lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.SAC_hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        self.actor_step = 0

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.SAC_lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.SAC_hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.SAC_lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.SAC_hidden_size, action_space).to(
                self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.SAC_lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        # print("action")
        return action.detach().cpu().numpy()[0]

    def train(self, batch_samples, expert_samples, dis_reward=None,
            mi_model=None, total_number_step=None, train_by_mi_reward=False, mi_weight=None):
        # Sample a batch from memory
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        # next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # action_batch = torch.FloatTensor(action_batch).to(self.device)
        # reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        # mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

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

        # middle point samples
        # gradient penalty, 取专家样本与当前生成样本之间的点，并限制这些点的梯度更新小于1.
        alpha = torch.from_numpy(np.random.uniform(size=(state.size()[0], 1))).type(torch.float32).to(
            self.device)  # 从一个均匀分布的区域中随机采样,默认区间[0,1]
        gp_state = (alpha * expert_state + (1 - alpha) * state).requires_grad_()
        gp_next_state = (alpha * expert_next_state + (1 - alpha) * next_state).requires_grad_()

        with torch.no_grad():
            # 1. sample action for next state
            next_action, next_state_log_pi, _ = self.policy.sample(next_state)
            expert_next_action, expert_next_state_log_pi, _ = self.policy.sample(expert_next_state)
            gp_action, gp_state_log_pi, _ = self.policy.sample(gp_state)

            # 2. compute reward with MI model
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

            # 3. Compute the target Q value
            # Starting from the goal state we can execute only non-actions.
            # the mask of absorbing state is 1.
            a_mask = torch.maximum(torch.tensor([0], device=self.device), mask)

            qf1_next_target, qf2_next_target = self.critic_target(next_state, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = rb_reward + a_mask * self.gamma * (min_qf_next_target)

            expert_a_mask = torch.maximum(torch.tensor([0], device=self.device), expert_mask)
            qf1_expert_next_target, qf2_expert_next_target = self.critic_target(expert_next_state, expert_next_action)
            expert_min_qf_next_target = torch.min(qf1_expert_next_target,
                                                  qf2_expert_next_target) - self.alpha * expert_next_state_log_pi
            expert_next_q_value = expert_reward + expert_a_mask * self.gamma * (expert_min_qf_next_target)

        qf1, qf2 = self.critic(state,
                               action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        expert_qf1, expert_qf2 = self.critic(expert_state, expert_action)
        expert_qf1_loss = F.mse_loss(expert_qf1,
                                     expert_next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        expert_qf2_loss = F.mse_loss(expert_qf2,
                                     expert_next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        expert_qf_loss = expert_qf1_loss + expert_qf2_loss

        gp_state_action = torch.cat([gp_state, gp_action], 1)
        gp_current_q1, gp_current_q2 = self.critic.forward1(gp_state_action)
        grad1 = autograd.grad(gp_current_q1.mean(), gp_state_action, create_graph=True, retain_graph=True)[0]
        grad_penalty1 = torch.pow(torch.norm(grad1, dim=-1) - 1, 2).mean()
        grad2 = autograd.grad(gp_current_q2.mean(), gp_state_action, create_graph=True, retain_graph=True)[0]
        grad_penalty2 = torch.pow(torch.norm(grad2, dim=-1) - 1, 2).mean()

        self.critic_optim.zero_grad()
        # total_critic_loss = qf_loss + expert_qf_loss + 10.0 * grad_penalty1 + 10.0 * grad_penalty2
        total_critic_loss = qf_loss + expert_qf_loss
        total_critic_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state)
        qf1_pi, qf2_pi = self.critic(state, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        expert_pi, expert_log_pi, _ = self.policy.sample(expert_state)
        expert_qf1_pi, expert_qf2_pi = self.critic(expert_state, expert_pi)
        expert_min_qf_pi = torch.min(expert_qf1_pi, expert_qf2_pi)

        gp_pi, gp_log_pi, _ = self.policy.sample(gp_state)
        gp_qf1_pi, gp_qf2_pi = self.critic.forward(gp_state, gp_pi)
        gp_min_qf_pi = torch.min(gp_qf1_pi, gp_qf2_pi)
        grad1 = autograd.grad(gp_min_qf_pi.mean(), gp_state, create_graph=True, retain_graph=True)[0]
        grad_penalty_for_actor = torch.pow(torch.norm(grad1, dim=-1) - 1, 2).mean()

        # a_mask = 1.0 - torch.maximum(torch.tensor([0], device=self.device), -mask)  # boardcast
        # expert_a_mask = 1.0 - torch.maximum(torch.tensor([0], device=self.device), -expert_mask)  # boardcast
        # if torch.sum(a_mask) < 1e-8:
        #     return torch.tensor([0], device=self.device)

        sac_q = ((self.alpha * log_pi) - min_qf_pi)  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = torch.sum(sac_q * mask) / torch.sum(mask)

        expert_sac_q = ((
                                    self.alpha * expert_log_pi) - expert_min_qf_pi)  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        expert_policy_loss = torch.sum(expert_sac_q * expert_mask) / torch.sum(expert_mask)

        self.policy_optim.zero_grad()
        # total_policy_loss = policy_loss + expert_policy_loss + 1.0 * grad_penalty_for_actor
        total_policy_loss = policy_loss + expert_policy_loss
        total_policy_loss.backward()
        self.policy_optim.step()
        self.actor_step += 1
        # print("self.actor_step:{}".format(self.actor_step))

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        return qf1_loss.item(), policy_loss.item(), mi_reward.mean().item(), rb_reward.mean().item()

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

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()
