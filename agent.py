"""
autor: Jiaqiang Zhao

time: 2024/9/20  20:38
"""

import torch
from torch import Tensor
from env import reward_compute
from env import read_state, env_step
from network import ActorPPO, CriticPPO


class PPOAgent:
    def __init__(self, args):
        self.state_dim = args.state_dim  
        self.action_dim = args.action_dim  

        self.last_state = None  # 计算优势函数时使用

        '''Arguments for device'''
        self.device = torch.device(
            f"cuda:{args.gpu_id}" if (torch.cuda.is_available() and (args.gpu_id >= 0)) else "cpu")

        '''Arguments for model'''
        self.learning_rate = args.learning_rate
        self.soft_update_tau = args.soft_update_tau

        self.cri = CriticPPO(args.cri_net_dims, args.state_dim, args.action_dim)#.to(self.device)
        self.act = ActorPPO(args.act_net_dims, args.state_dim, args.action_dim)#.to(self.device)
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate)
        self.criterion = torch.nn.SmoothL1Loss()

        '''Arguments for training'''
        self.repeat_times = args.repeat_times
        self.batch_size = args.batch_size  # 每轮训练的样本数量
        self.horizon = args.horizon_len

        self.reward_scale = args.reward_scale
        self.gamma = args.gamma
        self.ratio_clip = args.ratio_clip  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = args.lambda_gae_adv  # could be 0.80~0.99
        self.lambda_entropy = args.lambda_entropy  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

        '''Arguments for save/load of model'''
        self.cwd = args.cwd

        '''Arguments for serial communication'''
        self.serial = args.serial

    @staticmethod
    def optimizer_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    def explore_env(self, horizon_len) -> (Tensor, Tensor, Tensor, Tensor):
        """ 存储经验数据 """
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)

        """ 状态读取和动作获取 """
        get_actions = self.act.get_action
        arduino = self.serial

        ary_state = self.last_state  # 防止没有得到ary_state

        """ 和环境交互，收集数据 """
        for i in range(horizon_len):
            state = read_state(arduino, (1, self.state_dim)) #.to(self.device)  # shape == (1, state_dim)
            print(f'得到状态{state},形状{state.shape}\n')
            action, logprob = get_actions(state) #.to(self.device)   # shape1==(1, action_dim)
            print(f'输出动作{action},形状{action.shape}\n')
            reward = reward_compute(state)#.to(self.device)  # shape==(1,)
            print(f'奖励值{reward},形状{reward.shape}\n')

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward

            ary_action = action.cpu()  # 用于和环境交互，转移到cpu上更快
            ary_state = env_step(arduino, ary_action)

        self.last_state = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        return states, actions, logprobs, rewards

    def update_net(self, buffer):
        with torch.no_grad():
            states, actions, logprobs, rewards = buffer
            buffer_size = states.shape[0]

            '''get advantages(每一状态下的优势函数) reward_sums'''
            bs = 20
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]  # 小批次处理，发挥gpu的优势
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)  # 1e-5是为了防止除以0

        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1

        for _ in range(update_times):  # 小批次更新，一次喂给self.batch_size个数据为一个批次
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)  # 选择batch_size个索引
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            # 先更新价值网络
            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)  # 损失值，单个值
            self.optimizer_update(self.cri_optimizer, obj_critic)  # 有没有必要转移到GPU？

            # 再更新策略网络
            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item(), rewards

    def get_advantages(self, rewards, values):
        advantages = torch.empty_like(values)  # advantage value

        horizon_len = rewards.shape[0]

        next_state = torch.tensor(self.last_state, dtype=torch.float32).to(self.device)
        next_value = self.cri(next_state.unsqueeze(0)).detach().squeeze(1).squeeze(0)

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = advantage = delta + self.gamma * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages  # size==(buffer_size,)

    def save_model(self):
        torch.save(self.act, f'{self.cwd}/actor.pt')
        torch.save(self.act.state_dict(), f'{self.cwd}/actor_params.pt')
        print("model saved successfully!!!")


if __name__ == '__main__':
    pass
