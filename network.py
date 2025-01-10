"""
建立策略网络ActorPPO和价值网络CriticPPO
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.distributions.normal import Normal


class ActorPPO(nn.Module):  # 输入:Tensor 形状:(batch_size,state_dim)
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        # Tensor1 形状:(batch_size,action_dim),Tensor1 形状:(batch_size,1)
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()
        dist = Normal(action_avg, action_std)

        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)

        action = (torch.tensor([15, np.pi, 1]) * action.tanh()).detach()
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticPPO(nn.Module):  # 输入:Tensor 形状:(batch_size,state_dim)
    def __init__(self, dims: [int], state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:   # 输出:Tensor 形状:(batch_size,1)
        return self.net(state)  # advantage value


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


