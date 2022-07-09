# -*-  coding=utf-8 -*-
# @Time : 2022/6/22 14:51
# @Author : Scotty1373
# @File : DDPG.py
# @Software : PyCharm
import datetime as dt
import itertools
import random
import time
import copy
from collections import deque

import numpy as np
import torch
from torch.distributions import Normal

from utils_tools.utils import RunningMeanStd
from models.net import Actor, Critic
from utils_tools.utils import process_image
from utils_tools.unity import decode

time_Feature = round(time.time())
Noise = {}


class DDPG:
    def __init__(self, state_length, frame_overlay, action_dim, device_, sock_udp=None):
        self.t = 0
        self.max_Q = 0
        self.trainingLoss = 0
        self.train = True
        self.train_from_checkpoint = False

        # Get size of state and action
        self.state_length = state_length
        self.frame_overlay = frame_overlay
        self.action_dim = action_dim
        self.device = device_

        self.discount_factor = 0.97
        self.batch_size = 128
        self.train_start = 2000
        self.train_from_checkpoint_start = 3000
        self.tua = 0.99
        # 初始化history存放参数 ！！！可以不使用，直接使用train_replay返回值做
        self.history_loss_actor = 0.0
        self.history_loss_critic = 0.0

        # exploration noise
        self.acc_noise = Normal(torch.zeros(1), torch.ones(1)*0.5)
        self.ori_noise = Normal(torch.zeros(1), torch.ones(1)*0.2)

        # state rms
        self.pixel_rms = RunningMeanStd(shape=(1, self.frame_overlay, 80, 80))
        self.vect_rms = RunningMeanStd(shape=(1, self.frame_overlay*self.state_length))
        self.rwd_rms = RunningMeanStd()

        # Create replay memory using deque
        self.memory = deque(maxlen=32000)

        # For Unity Connection
        self.sock = sock_udp
        self.remoteHost = None
        self.remotePort = None
        assert self.sock is not None

        # module first init
        self._init()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        self.opt_actor = torch.optim.Adam(self.actor_model.parameters(), lr=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic_model.parameters(), lr=1e-5)
        self.loss_critic = torch.nn.MSELoss()

        self.hard_update_target_model(self.actor_model, self.actor_target_model)
        self.hard_update_target_model(self.critic_model, self.critic_target_model)

    def _init(self):
        # Create main model and target model
        self.actor_model = Actor(state_length=self.state_length,
                                 frame_overlay=self.frame_overlay,
                                 action_dim=self.action_dim).to(self.device)
        self.actor_target_model = Actor(state_length=self.state_length,
                                        frame_overlay=self.frame_overlay,
                                        action_dim=self.action_dim).to(self.device)

        self.critic_model = Critic(state_length=self.state_length,
                                   frame_overlay=self.frame_overlay,
                                   action_dim=self.action_dim).to(self.device)
        self.critic_target_model = Critic(state_length=self.state_length,
                                          frame_overlay=self.frame_overlay,
                                          action_dim=self.action_dim).to(self.device)
        for param in self.actor_target_model.parameters():
            param.requires_grad = False
        for param in self.critic_target_model.parameters():
            param.requires_grad = False

    # Get action from model using epsilon-greedy policy
    def get_action(self, Input, noise_added=None):
        self.actor_model.eval()
        act = self.actor_model(Input[0], Input[1])
        if noise_added is not None:
            # noise_added = torch.from_numpy(noise_added.__call__()).to(self.device)
            acc_noise_adder = torch.FloatTensor(self.acc_noise.sample()).to(self.device)
            ori_noise_adder = torch.FloatTensor(self.ori_noise.sample()).to(self.device)
            act[..., 0] += acc_noise_adder
            act[..., 1] += ori_noise_adder
        self.actor_model.train()
        act[..., 0].clamp_(-1, 1)
        act[..., 1].clamp_(-1, 1)
        return act

    def replay_memory(self, pixel, vect, action, reward, next_pixel, next_vect, done):
        self.memory.append((pixel, vect, action, reward, next_pixel, next_vect, done, self.t))

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        elif self.train_from_checkpoint:
            if len(self.memory) < self.train_from_checkpoint_start:
                return
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        '''
        torch.float64对应torch.DoubleTensor
        torch.float32对应torch.FloatTensor
        '''
        # from_numpy会对numpy数据类型转换成双精度张量，而torch.Tensor不存在这种问题，torch.Tensor将数组转换成单精度张量
        pixel, vect, action_t, reward_t, next_pixel, next_vect, terminal, step = zip(*minibatch)
        pixel = torch.Tensor(pixel).squeeze().to(self.device)
        next_pixel = torch.Tensor(next_pixel).squeeze().to(self.device)
        vect = torch.Tensor(vect).squeeze().to(self.device)
        next_vect = torch.Tensor(next_vect).squeeze().to(self.device)
        action_t = torch.Tensor(action_t).reshape(-1, 2).to(self.device)
        terminal = torch.BoolTensor(terminal).reshape(-1, 1).to(self.device)
        terminal = terminal.float()

        reward_t = np.array(reward_t).reshape(-1, 1)
        mean, std, count = reward_t.mean(), reward_t.std(), reward_t.shape[0]
        self.rwd_rms.update_from_moments(mean, std**2, count)
        reward_t = (reward_t - self.rwd_rms.mean) / np.sqrt(self.rwd_rms.var)
        reward_t = torch.Tensor(reward_t).to(self.device)

        # Critic loss
        self.opt_critic.zero_grad()
        # pi'(si+1)
        action_target = self.actor_target_model(next_pixel, next_vect)

        # Ｑ(si+1, pi'(si+1)) for calculate td target
        td_target = reward_t + self.discount_factor * (1 - terminal) * self.critic_target_model(next_pixel, next_vect, action_target.detach_())

        # Q(si, ai)
        critic_loss_cal = self.loss_critic(self.critic_model(pixel, vect, action_t), td_target.detach())
        critic_loss_cal.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), 1)
        self.opt_critic.step()
        self.history_loss_critic = critic_loss_cal.item()

        # Actor loss
        if self.t % 3 == 0:
            # 重置critic，actor优化器参数, 否则会影响下次训练
            self.opt_actor.zero_grad()
            policy_actor = self.critic_model(pixel, vect, self.actor_model(pixel, vect))
            policy_actor = -policy_actor.mean()
            policy_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.5)
            self.opt_actor.step()
            self.history_loss_actor = policy_actor.item()
            with torch.no_grad():
                self.soft_update_target_model(self.actor_model, self.actor_target_model)
        with torch.no_grad():
            self.soft_update_target_model(self.critic_model, self.critic_target_model)

    def load_model(self, name):
        checkpoints = torch.load(name)
        self.critic_model.load_state_dict(checkpoints['model_critic'])
        self.actor_model.load_state_dict(checkpoints['model_actor'])
        self.opt_critic.load_state_dict(checkpoints['optimizer_critic'])
        self.opt_actor.load_state_dict(checkpoints['optimizer_actor'])

    # Save the model which is under training
    def save_model(self, name):
        torch.save({'model_actor': self.actor_model.state_dict(),
                    'model_critic': self.critic_model.state_dict(),
                    'optimizer_actor': self.opt_actor.state_dict(),
                    'optimizer_critic': self.opt_critic.state_dict()}, name)

    # target model硬更新
    @staticmethod
    def hard_update_target_model(model, target_model):
        # 解决state_dict浅拷贝问题
        with torch.no_grad():
            weight_model = copy.deepcopy(model.state_dict())
            target_model.load_state_dict(weight_model)

    # target model软更新
    def soft_update_target_model(self, source_model, target_model):
        for target_param, source_param in zip(target_model.parameters(),
                                              source_model.parameters()):
            target_param.data.copy_(self.tua * target_param + (1 - self.tua) * source_param)

    def Recv_data_Format(self, _done, pixel=None, vect=None, action=None, ep_lens=None):
        # 第一次初始化接收
        if _done != 0:
            revcData, (self.remoteHost, self.remotePort) = self.sock.recvfrom(65535)
            image, _, _, x_gap, v_ego, v_lead, z_gap, angle_ego, angle_lead = decode(revcData)
            pixel_single_frame = process_image(image)
            vect_single_frame = np.array((x_gap, v_ego, z_gap))

            pixel = np.ones((1, self.frame_overlay, 80, 80)) * pixel_single_frame
            vect = np.ones((1, self.frame_overlay, self.state_length)) * vect_single_frame[None, ...]
            vect = vect.reshape(1, -1)
            return pixel, vect

        else:
            revcData, _ = self.sock.recvfrom(65535)

            image, reward, done, x_gap, v_ego, v_lead, z_gap, angle_ego, angle_lead = decode(revcData, vect[..., 0], vect[..., 1], action[..., 0], action[..., 1], ep_lens)

            next_pixel_frame = process_image(image)[None, ...]
            pixel = np.append(next_pixel_frame, pixel[:, :3, :, :], axis=1)
            next_vect_frame = np.array((x_gap, v_ego, z_gap))[None, ...]
            vect = np.append(next_vect_frame, vect[:, :self.state_length*(self.frame_overlay-1)], axis=1)
            info = {}
            return pixel, vect, reward, done, info

    def Send_data_Format(self, pixel=None, vect=None, episode_len=None, UnityResetFlag=None, skip=False):
        if skip:
            pred_time_pre = dt.datetime.now()
            # 为了防止tanh输出达到边界值，导致神经元失活，对输出数据做clip固定在（-0.5， 0.5）之后再做重映射
            action_send = np.zeros((1, 2))

            if UnityResetFlag == 1:
                strr = str(4) + ',' + str(action_send[..., 0].item()) + ',' + str(action_send[..., 1].item())
                UnityResetFlag = 0
            else:
                strr = str(1) + ',' + str(action_send[..., 0].item()) + ',' + str(action_send[..., 1].item())

            sendDataLen = self.sock.sendto(strr.encode(), (self.remoteHost, self.remotePort))  # 0.06s later receive
            return UnityResetFlag
        else:
            pred_time_pre = dt.datetime.now()
            episode_len = episode_len + 1
            # Get action for the current state and go one step in environment
            pixel = torch.Tensor(pixel).to(self.device)
            vect = torch.Tensor(vect).to(self.device)
            action = self.get_action([pixel, vect], True)
            # 为了防止tanh输出达到边界值，导致神经元失活，对输出数据做clip固定在（-0.5， 0.5）之后再做重映射
            action_send = action.detach().cpu().numpy()

            if UnityResetFlag == 1:
                strr = str(4) + ',' + str(action_send[..., 0].item()) + ',' + str(action_send[..., 1].item())
                UnityResetFlag = 0
            else:
                strr = str(1) + ',' + str(action_send[..., 0].item()) + ',' + str(action_send[..., 1].item())

            sendDataLen = self.sock.sendto(strr.encode(), (self.remoteHost, self.remotePort))  # 0.06s later receive
            return episode_len, action_send, UnityResetFlag

    def reset(self):
        strr = str(3) + ',' + '0.0' + ',' + '0.0'
        sendDataLen = self.sock.sendto(strr.encode(), (self.remoteHost, self.remotePort))

