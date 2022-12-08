import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from common.utils import to_numpy, weights_init
from common.utils import TensorWrapper


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
    relu_logits = torch.where(cond, logits, zeros)
    neg_abs_logits = torch.where(cond, -logits, logits)
    res = (relu_logits - logits * label) + torch.log1p(torch.exp(neg_abs_logits))
    if weight is not None:
        res *= weight
    return res.mean()


class Discriminator(nn.Module):

    def __init__(self, sa_dim, lr=0.001, device='cpu') -> None:
        hidden_dim = 256
        super(Discriminator, self).__init__()
        print("sa_dim:", sa_dim)
        self.net = nn.Sequential(nn.Linear(sa_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, 1))
        weights_init(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.device = device

    def forward(self, input):
        return self.net(input)

    def update(self,
               gene_input,
               expert_input,
               lambd=10.0,
               gene_weight=1.0,
               expert_weight=1.0,
               gene_label_smooth=.0,
               expert_label_smooth=.0,
               log_info: dict = None):
        # gene_input.shape: [batch_len, sa_dim], expert_input.shape: [batch_len, sa_dim]
        # gene_weight.shape: [batch_len, 1], expert_weight.shape: [batch_len, 1]
        # Avoid using tensorflow random functions since it's impossible to get
        # the state of the random number generator used by TensorFlow.
        alpha = np.random.uniform(size=(gene_input.shape[0], 1))
        alpha = TensorWrapper(alpha, self.device)
        inter_input = (alpha * gene_input + (1. - alpha) * expert_input).requires_grad_()  # 开启梯度
        # print("gene_input.shape:", gene_input.shape)
        gene_output = self(gene_input)
        expert_output = self(expert_input)
        inter_output = self(inter_input)

        gene_label = (gene_label_smooth) * torch.ones_like(gene_output, device=self.device)
        expert_label = (1. - expert_label_smooth) * torch.ones_like(gene_output, device=self.device)
        gene_weight = gene_weight * torch.ones_like(gene_label, device=self.device)
        expert_weight = expert_weight * torch.ones_like(expert_label, device=self.device)
        gene_loss = sigmoid_cross_entropy_with_logits(gene_output, 0.)
        expert_loss = sigmoid_cross_entropy_with_logits(expert_output, 1.)

        # ? 为了计算二阶导数，这里一阶导数部分必须设置create_graph=True, retain_graph=True
        grad = autograd.grad(inter_output.sum(), inter_input, create_graph=True, retain_graph=True)[0]
        grad_penalty = torch.pow(torch.norm(grad, dim=-1) - 1, 2).mean()

        self.optimizer.zero_grad()
        loss = gene_loss + expert_loss + lambd * grad_penalty
        loss.backward()
        self.optimizer.step()

        if log_info is not None:
            log_info['dis_gene_output'].append(to_numpy(gene_output.mean()))
            log_info['dis_expert_output'].append(to_numpy(expert_output.mean()))
            log_info['dis_gene_loss'].append(to_numpy(gene_loss))
            log_info['dis_expert_loss'].append(to_numpy(expert_loss))
            log_info['dis_grad_penalty'].append(to_numpy(grad_penalty))
