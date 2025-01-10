import os
from env import link_serial, read_state, env_step


class Config:  # for on-policy
    def __init__(self):
        '''Arguments for state and action'''
        self.n = 10  # range of time
        self.state_dim = 2 * self.n  # vector dimension (feature number) of state
        self.action_dim = 3  # vector dimension (feature number) of action

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256
        self.ratio_clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_gae_adv = 0.95  # could be 0.80~0.99
        self.lambda_entropy = 0.01  # could be 0.00~0.10

        '''Arguments for training'''
        self.cri_net_dims = (64, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.act_net_dims = (64, 32)
        self.learning_rate = 6e-5  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3，未使用

        self.batch_size = int(128)  # num of transitions sampled from replay buffer.单个训练批次的大小，未使用
        self.horizon_len = int(200)  # collect horizon_len step while exploring, then update network.3个里只用到了这个
        self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.在线算法用不到
        self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
        self.train_break_step = int(800)

        '''Arguments for device and distributed training'''
        self.gpu_id = int(-1)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`，未使用
        self.random_seed = int(0)  # initialize random seed in self.init_before_training()，未使用

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.evl_break_step = int(200)

        '''Arguments for serial communication'''
        self.port = 'COM8'
        self.rate = 9600
        self.times = 1
        self.serial = link_serial(port=self.port, rate=self.rate, times=self.times)

        '''Arguments for interact with the environment,要不算了，直接从env.py导入引用即可'''
        # self.read_state = read_state(arduino=self.serial, state_size=(self.state_dim,))

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./model and data'
        os.makedirs(self.cwd, exist_ok=True)
