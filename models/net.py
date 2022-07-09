# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import itertools
from torch.utils.tensorboard import SummaryWriter

_HIDDEN_UNIT_EXTRACTOR = 512
_HIDDEN_UNIT_VECT = 100

def _uniform_init_(layer, a=-0.1, b=0.1):
    torch.nn.init.uniform_(layer.weight, a, b)
    torch.nn.init.constant_(layer.bias, 0.0)
    return layer


class Actor(nn.Module):
    def __init__(self, state_length, frame_overlay, action_dim):
        super(Actor, self).__init__()
        self.state_length = state_length
        self.frame_overlay = frame_overlay
        self.action_dim = action_dim
        assert isinstance(self.action_dim, int)
        self.conv1 = nn.Conv2d(in_channels=self.frame_overlay, out_channels=16,
                               kernel_size=(8, 8), stride=(4, 4))
        self.conv_actv1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(4, 4), stride=(2, 2))
        self.conv_actv2 = nn.ELU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1))
        self.conv_actv3 = nn.ELU(inplace=True)
        self.down_vect = nn.Linear(2304, _HIDDEN_UNIT_EXTRACTOR)

        self.inputDense1 = nn.Linear(self.state_length*self.frame_overlay, _HIDDEN_UNIT_VECT)
        self.down_vect_dense = nn.Linear(_HIDDEN_UNIT_EXTRACTOR + _HIDDEN_UNIT_VECT, 300)

        self.Dense1 = nn.Linear(300, 300)
        self.actv1 = nn.ELU(inplace=True)

        # layer_acc = [_uniform_init_(nn.Linear(300+_HIDDEN_UNIT_VECT, 128), 0, 3e-3),
        #              nn.ELU(inplace=True),
        #              _uniform_init_(nn.Linear(128, 1), 0, 3e-2),
        #              nn.Tanh()]
        layer_acc = [_uniform_init_(nn.Linear(300+_HIDDEN_UNIT_VECT, 1), 0, 5e-2),
                     nn.Tanh()]
        self.layer_acc = nn.Sequential(*layer_acc)

        # layer_ori = [_uniform_init_(nn.Linear(300+_HIDDEN_UNIT_VECT, 128), -3e-3, 3e-3),
        #              nn.ELU(inplace=True),
        #              _uniform_init_(nn.Linear(128, 1), -3e-3, 3e-3),
        #              nn.Tanh()]
        layer_ori = [_uniform_init_(nn.Linear(300+_HIDDEN_UNIT_VECT, 1), -3e-3, 3e-3),
                     nn.Tanh()]
        self.layer_ori = nn.Sequential(*layer_ori)

    def forward(self, x1, x2):
        # extractor
        feature = self.conv1(x1)
        feature = self.conv_actv1(feature)
        feature = self.conv2(feature)
        feature = self.conv_actv2(feature)
        feature = self.conv3(feature)
        feature = self.conv_actv3(feature)
        feature = torch.flatten(feature, start_dim=1, end_dim=-1)
        extractor = self.down_vect(feature)

        out = self.inputDense1(x2)

        output = torch.cat([extractor, out], dim=1)
        output = torch.nn.functional.elu(output)
        output = self.down_vect_dense(output)
        output = torch.nn.functional.elu(output)

        # actor
        output = self.Dense1(output)
        output = self.actv1(output)

        out = torch.nn.functional.elu(out)
        common_output = torch.cat((output, out), dim=-1)

        acc_out = self.layer_acc(common_output)
        ori_out = self.layer_ori(common_output)
        return torch.cat([acc_out, ori_out], dim=-1)


class Critic(nn.Module):
    def __init__(self, state_length, frame_overlay, action_dim):
        super(Critic, self).__init__()
        self.state_length = state_length
        self.frame_overlay = frame_overlay
        self.action_dim = action_dim
        assert isinstance(action_dim, int)
        self.conv1 = nn.Conv2d(in_channels=self.frame_overlay, out_channels=16,
                               kernel_size=(8, 8), stride=(4, 4))
        self.conv_actv1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=(4, 4), stride=(2, 2))
        self.conv_actv2 = nn.LeakyReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1))
        self.conv_actv3 = nn.LeakyReLU(inplace=True)
        self.down_vect = nn.Linear(2304, _HIDDEN_UNIT_EXTRACTOR)
        self.down_vect_dense = nn.Linear(_HIDDEN_UNIT_EXTRACTOR+_HIDDEN_UNIT_VECT, 300)

        self.inputDense1 = nn.Linear(self.state_length * self.frame_overlay, _HIDDEN_UNIT_VECT)

        action_embedding = 200
        self.inputDense2 = nn.Linear(self.action_dim, action_embedding)
        self.inputactv = nn.LeakyReLU(inplace=True)
        self.Dense1 = nn.Linear(300 + action_embedding, 300)
        self.actv3 = nn.LeakyReLU(inplace=True)
        self.Dense2 = nn.Linear(300 + action_embedding, 128)
        self.actv4 = nn.LeakyReLU(inplace=True)
        self.Dense3 = nn.Linear(128, 1)

    def forward(self, x1, x2, action):
        feature = self.conv1(x1)
        feature = self.conv_actv1(feature)
        feature = self.conv2(feature)
        feature = self.conv_actv2(feature)
        feature = self.conv3(feature)
        feature = self.conv_actv3(feature)
        feature = torch.flatten(feature, start_dim=1, end_dim=-1)
        extractor = self.down_vect(feature)

        out = self.inputDense1(x2)

        output = torch.cat([extractor, out], dim=1)
        output = torch.nn.functional.elu(output)
        output = self.down_vect_dense(output)
        output = torch.nn.functional.elu(output)

        input_action = self.inputDense2(action)
        input_action = self.inputactv(input_action)

        output = torch.cat([output, input_action], dim=1)
        critic = self.Dense1(output)
        critic = self.actv3(critic)
        critic_concat = torch.cat([critic, input_action], dim=1)
        critic_concat = self.Dense2(critic_concat)
        critic_concat = self.actv4(critic_concat)
        critic_concat = self.Dense3(critic_concat)
        return critic_concat


if __name__ == '__main__':
    actor_net = Actor(action_dim=2)
    critic_net = Critic(action_dim=2)
    loss = torch.nn.MSELoss()
    opt_common = torch.optim.SGD(common_net.parameters(), 0.001)
    opt_actor = torch.optim.SGD(actor_net.parameters(), 0.001)
    opt_actor_fusion = torch.optim.SGD(itertools.chain(common_net.parameters(), actor_net.parameters()), 0.001)

    x = torch.randn((10, 4, 80, 80))
    y = torch.randn((10, 2*4))

    common = common_net(x, y)
    out1 = actor_net(common)

    tgt = torch.rand(10, 2)
    # tgt_critic = torch.randn((10, 1))

    loss_scale = loss(out1, tgt)
    opt_actor_fusion.zero_grad()
    loss_scale.backward()

    opt_actor_fusion.step()



