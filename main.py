"""
----------------------------------------------------------------------
项目名称/Project          : 强化学习的机械实现
程序名称/Program name     : Python-side implementation
作者/Author              : 赵佳强
日期/Date                : 2024/09/26
程序目的/Purpose          : 此程序为“强化学习的机械实现”项目的Python端的代码。
该项目应用强化学习算法实现对振动系统的振动控制。
Arduino UNO板完成系统状态的读取与舵机控制，python完成智能决策(根据系统状态，输出舵机转动的角度)的过程。
python端和Arduino端的数据传输通过串口通信的方式实现。

-----------------------------------------------------------------------
该程序主要参考自深度强化学习算法库ElegantRL(开源)：
https://github.com/AI4Finance-Foundation/ElegantRL
-----------------------------------------------------------------------
"""

import os
import itertools
import torch
import numpy as np
from agent import PPOAgent
from config import Config
from env import read_state, env_step
from matplotlib import pyplot as plt
from plot import draw


def train_agent(args):
    """ 训练准备 """
    args.init_before_training()  # set cwd (current working directory) for saving model
    agent = PPOAgent(args=args)
    agent.last_state = None

    logging_rewards = []
    logging_obj_critics = []
    logging_obj_actors = []
    total_step = 0
    episode = 0

    """ 开始训练 """
    torch.set_grad_enabled(False)
    while True:
        buffer_items = agent.explore_env(args.horizon_len)  # 探索环境，收集数据

        torch.set_grad_enabled(True)
        logging = agent.update_net(buffer_items)  # 进行训练和更新
        torch.set_grad_enabled(False)

        # 每个episode收集horizon_len条数据，更新update_times次网络，得到一个return_episode
        logging_obj_critics.append(logging[0])  # 记录每个episode的回报
        logging_obj_actors.append(logging[1])  # 记录每个episode的回报
        logging_rewards.append(logging[3])  # 记录每个step的奖励值
        total_step += args.horizon_len

        print(f'完成第{episode+1}轮训练\n')
        episode += 1
        if total_step >= args.train_break_step:
            break

    """ 保存模型与训练数据 """
    agent.save_model()

    # _rewards = [item for sublist in logging_rewards for item in sublist]
    _rewards = list(itertools.chain(*logging_rewards))
    __rewards = [t.item() for t in _rewards]
    print(f'__rewards长度{len(__rewards)}')
    print(__rewards)
    data_reward = np.column_stack(([i for i in range(len(__rewards))], __rewards))
    file_path = os.path.join(args.cwd, 'data_reward.txt')
    np.savetxt(file_path, data_reward)
    # data_obj_critics = np.column_stack(([i for i in range(len(logging_obj_critics))], logging_obj_critics))
    # np.savetxt(args.cwd, data_obj_critics)
    # data_obj_actors = np.column_stack(([i for i in range(len(logging_obj_actors))], logging_obj_actors))
    # np.savetxt(args.cwd, data_obj_actors)

    print("————————————————————————————————————智能体模型和训练数据已保存!!!———————————————————————————————————")

    return logging_rewards, logging_obj_critics, logging_obj_actors


def evaluator(act_model, args):
    rewards = []
    while len(rewards) >= args.evl_break_step:
        state = read_state(arduino=args.serial, state_size=(args.state_dim,))
        reward = -torch.sqrt(torch.pow(state[0][0], 2) + torch.pow(state[0][1], 2))  # torch.float型
        rewards.append(reward)

        action = act_model(state)
        next_state = env_step(arduino=args.serial, action=action)
    return rewards


# def draw(path):
    # x_data = []
    # y_data = []

    # 获取文件夹中所有的 txt 文件
    # files = [f for f in os.listdir(path) if f.endswith('.txt')]
    # for file in files:
        # file_path = os.path.join(path, file)
        # data = np.loadtxt(file_path)

        # 假设第一列是横轴，第二列是纵轴
        # x_values = data[:, 0]
        # y_values = data[:, 1]

        # x_data = x_values
        # y_data.append(y_values)

    # y_data = np.array(y_data)
    # mean_y = np.mean(y_data, axis=0)
    # std_y = np.std(y_data, axis=0)

    # 绘制
    # plt.plot(x_data, mean_y, label='reward', color='b')
    # plt.fill_between(x_data, mean_y - std_y, mean_y + std_y, color='b', alpha=0.2)
    # plt.xlabel('step')
    # plt.ylabel('reward')
    # plt.title('Reward Curve')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    """ 参数设置 """
    argparse = Config()  # 超参数、文件路径
    arduino = argparse.serial  # 串口

    """ 训练与评估 """
    print("————————————————————————————————————Start training!!!——————————————————————————————————————")
    train_logging = train_agent(argparse)  # train_logging=(logging_rewards,logging_obj_critics,logging_obj_actors)
    print("——————————————————————————————————The training is over!!!——————————————————————————————————")
    draw(argparse.cwd)   # 绘制训练曲线

    print("———————————————————————————————————Start evaluating!!!——————————————————————————————————————")
    actor = torch.load(f'{argparse.cwd}/actor.pt')
    evl_logging = evaluator(act_model=actor, args=argparse)
    print("————————————————————————————————The evaluation is over!!!————————————————————————————————————")
    # draw(argparse.cwd)  # 绘制评估曲线
