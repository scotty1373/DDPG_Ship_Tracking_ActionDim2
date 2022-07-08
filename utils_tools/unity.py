# -*-  coding=utf-8 -*-
# @Time : 2022/6/22 14:48
# @Author : Scotty1373
# @File : unity.py
# @Software : PyCharm
import base64
import io
from utils_tools.utils import normailze

import numpy as np
from PIL import Image

Distance_EF = 70
Return_Time = 3.5
Variance = 0.5
HorizontalOffset = 12


def decode(recv_data, x_gap_old=0, v_ego=0, acc=0, ori=0, ep_lens=0):
    # received data processing
    # 用逗号将接收的数据分隔，分隔为4个不同的部分
    rec_list = str(recv_data).split(',', 7)
    x_gap = rec_list[0][2:]  # 纵向
    z_gap = rec_list[1]  # 横向
    v_ego1 = rec_list[2]  # speed of egoVehicle
    v_lead1 = rec_list[3]  # acceleration of egoVehicle
    angle_ego = rec_list[4]
    done = int(rec_list[5])
    img = base64.b64decode(rec_list[6])  # image from mainCamera
    angle_lead = rec_list[7]
    image = Image.open(io.BytesIO(img))
    # BILINEAR 双线性变换
    image = image.resize((80, 80), resample=Image.BILINEAR)
    image = np.array(image)
    image = normailze(image)
    x_gap = float(x_gap) - 12  # -车身
    leadspeed_real = (x_gap - float(x_gap_old)) / 2 + float(v_ego1)

    # 最终输出的_reward有1 0.5 0三个值
    gap = x_gap  # -车身
    _reward = cal_reward(gap, float(v_ego), leadspeed_real, acc)
    reward_z = cal_reward_steer(float(z_gap), float(angle_ego), ori)
    if done == 1:
        reward_z = -5

    if gap <= 3 or gap >= 300:
        done = 1
        _reward = -5.0
    elif ep_lens > 480:
        done = 2

    R = _reward + reward_z
    if isinstance(R, np.ndarray):
        print("R is numpy array")
    return image, R, done, gap / 55, float(v_ego1), float(leadspeed_real), \
           float(z_gap), float(angle_ego), float(angle_lead[:-1])


# 修改正态分布中的方差值，需要重新将正态分布归一化
def CalReward(action_relative_, action_best_):
    try:
        reward_recal = (np.exp(-(action_relative_ - action_best_) ** 2 / (2 * (Variance ** 2))))
    except FloatingPointError as e:
        reward_recal = 0
    return reward_recal


def cal_reward(_gap, _v_ego, _v_lead, _force):
    # reward for vertical gap
    if isinstance(_force, np.ndarray):
        _force = _force.item()
    if _force > 0:
        a = 3.5 * _force
    else:
        a = _force
    # reward for gap
    if 40 <= _gap <= 100:
        if _v_lead - _v_ego > 3.5:
            rd = (1 / (_v_lead - _v_ego + 1e-4)) * a
        elif _v_ego - _v_lead > 3.5:
            rd = (1 / (_v_lead - _v_ego + 1e-4)) * a
        else:
            rd = 1 - abs(_gap - 70) / 30
    elif 30 <= _gap < 40:
        rd = 0.25
    elif _gap < 30:
        rd = -0.5
    elif 100 < _gap <= 150:
        rd = -0.5
    else:
        rd = -1
    return rd


def cal_reward_steer(_gap, _angle_ego, _angle):
    # reward for horizontal gap
    rd = - abs(_gap / HorizontalOffset)
    return rd
