import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from common.utils import soft_update, hard_update
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class MutualInformationNetwork(nn.Module):
    def __init__(self, num_inputs, num_oupts, hidden_dim, action_space=None):
        super(MutualInformationNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.linear3 = nn.Linear(hidden_dim, num_oupts)
        # self.noise = torch.Tensor(num_oupts)

        self.apply(weights_init_)

        # output rescaling
        self.output_scale = torch.tensor(0.5)
        self.output_bias = torch.tensor(0.5)

    def forward(self, state_action):
        x = F.relu(self.linear1(state_action))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.linear3(x)) * self.output_scale + self.output_bias  # 0<= mean <=1
        return mean

    # def sample(self, state_action):
    #     mean = self.forward(state_action)
    #     noise = self.noise.normal_(0., std=0.1)
    #     noise = noise.clamp(-0.2, 0.2)
    #     output = mean + noise
    #     return output, mean

    def to(self, device):
        self.output_scale = self.output_scale.to(device)
        self.output_bias = self.output_bias.to(device)
        # self.noise = self.noise.to(device)
        return super(MutualInformationNetwork, self).to(device)


class MutualInformation(object):
    def __init__(self, num_inputs, action_space, device, exp_hidden_dim, exp_lr, num_oupts=1):  #
        self.num_inputs = num_inputs + action_space + num_inputs  # state_action_target
        self.num_oupts = num_oupts
        self.device = device
        self.network = MutualInformationNetwork(num_inputs=self.num_inputs, num_oupts=self.num_oupts,
                                                hidden_dim=exp_hidden_dim, action_space=None).to(
            self.device)
        self.network_optim = Adam(self.network.parameters(), lr=exp_lr)

    def update_mi_network(self, reach_samples, batch_samples):
        # state_action(distribution_xy): sampled from the replay buffer which save all state-action pairs
        # state_action_target: sampled from the replay buffer
        # which save the state-action pairs which reach the target state
        # distribution_xy = state_action_target

        state_batch_r = torch.FloatTensor(reach_samples.obs).to(self.device)
        action_batch_r = torch.FloatTensor(reach_samples.action).to(self.device)
        target_batch_r = torch.FloatTensor(reach_samples.target).to(self.device)

        state_batch = torch.FloatTensor(batch_samples.obs).to(self.device)
        action_batch = torch.FloatTensor(batch_samples.action).to(self.device)

        state_action = torch.cat((state_batch, action_batch), dim=1)
        reach_state_action_target = torch.cat((state_batch_r, action_batch_r, target_batch_r), dim=1)

        state_action_size = state_action.shape[1]
        # batch_size = state_action_target.shape[0]
        idx = torch.randperm(reach_state_action_target.shape[0])
        shuffled_target = reach_state_action_target[idx, state_action_size:]

        # target = state_action_target[:, state_action_size:]
        distribution_x_y = torch.cat((state_action, shuffled_target), dim=1)
        # sending distribution_xy and distribution_x_y to network
        input_data = torch.cat((reach_state_action_target, distribution_x_y), dim=0)

        # compute the loss of MutualInformationNetwork
        logits = self.network.forward(input_data)
        pred_xy = logits[:reach_state_action_target.shape[0]]
        pred_x_y = logits[reach_state_action_target.shape[0]:]
        # loss = - np.log2(np.exp(1)) * (torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y))))
        loss = - np.log2(np.exp(1)) * (torch.log(torch.mean(torch.exp(pred_xy))) - torch.mean(pred_x_y))
        # print("Update MI: pred_xy:{}, pred_x_y:{}".format(pred_xy.mean().item(), pred_x_y.mean().item()))

        self.network_optim.zero_grad()
        loss.backward()
        self.network_optim.step()

    def get_mi_reward(self, state, action, next_state, state_target):
        # state_tensor = torch.FloatTensor(state)
        # next_state_tensor = torch.FloatTensor(next_state)
        # action_tensor = torch.FloatTensor(action)
        # state_target_tensor = torch.FloatTensor(state_target)
        # next_state_target_tensor = torch.FloatTensor(next_state_target)
        state_action_target = torch.cat([state, action, state_target], dim=1).to(self.device)
        next_state_action_target = torch.cat([next_state, action, state_target], dim=1).to(self.device)

        logits_1 = self.network(state_action_target)
        logits_2 = self.network(next_state_action_target)
        # mutual_information1 = np.log2(np.exp(1)) * (torch.mean(logits_1) - torch.log(torch.mean(torch.exp(logits_1))))
        # mutual_information2 = np.log2(np.exp(1)) * (torch.mean(logits_2) - torch.log(torch.mean(torch.exp(logits_2))))

        potential_energy_reward = logits_2 - logits_1
        # print("Update MI: current_state:{}, next_state:{}, energy_reward:{}"
        #       .format(logits_1.mean().item(), logits_2.mean().item(), potential_energy_reward.mean().item()))

        return potential_energy_reward, logits_1, logits_2


class CheckTarget:
    def __init__(self, target_interval, expert_target, difference_weight):
        self.target_interval = target_interval
        self.difference_weight = difference_weight
        self.step_list0, self.step_list1, self.reach_step = [], [], []
        self.target_list = expert_target
        self.get_step(expert_target)
        print(f"step_list0 {self.step_list0}, \nstep_list1 {self.step_list1}")

        self.difference = [2]
        self.position = 0

    def get_step(self, expert_trajectory):
        for i in range(int(1000 / self.target_interval)):
            self.step_list0.append(i * self.target_interval)  # the start step of each interval
            self.step_list1.append((i + 1) * self.target_interval)  # the end step of each interval

    def check_reach_target(self, current_step, current_state):
        if (current_step + 1) in self.step_list1:
            for i in range(len(self.target_list)):
                # self.target_list.shape=[1000/interval,number_of_trajectory,shape_of_state]
                # 每个维度的差距都小于 self.difference
                if (abs((current_state - self.target_list[i][int(current_step / self.target_interval - 1)].obs).mean())
                        <= self.difference_weight * np.array(self.difference).mean()):
                    # print(
                    #     f"mean:{(current_state - self.target_list[i][int(current_step / self.target_interval - 1)].obs).mean():.4}"
                    #     f"a:{(current_state - self.target_list[i][int(current_step / self.target_interval - 1)].obs).mean() <= self.difference}")
                    # if (current_state - self.target_list[int(current_step / self.target_interval - 1), i, :] <= 1000).all():
                    return 1, self.target_list[i][int(current_step / self.target_interval - 1)].obs
                return 0, None
        else:
            return 0, None

    def current_difference(self, mean):
        if len(self.difference) <= 20:
            self.difference.append(None)
        self.difference[self.position] = mean
        self.position = (self.position + 1) % 20