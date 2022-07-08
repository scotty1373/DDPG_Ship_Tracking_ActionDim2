# -*- coding: utf-8 -*-
import random
import torch
import itertools
from skimage import color
import numpy as np
import copy
import time
import os

time_Feature = round(time.time())

# 单目标斜对角坐标
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xyxy2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2]
    y[:, 3] = x[:, 3]
    y = y.type(torch.IntTensor)
    return y


def print_out(file, text):
    file.write(text + '\n')
    file.flush()


def log_File_path(path):
    # date = str(dt.date.today()).split('-')
    # date_concat = date[1] + date[2]
    date_concat = time_Feature
    train_log_ = open(os.path.join(path, 'train_log_{}.txt'.format(date_concat)), 'w')
    # test_log_ = open(os.path.join(path, 'test_log_{}.txt'.format(date_concat)), 'w')
    del date_concat
    return train_log_


def Model_save_Dir(PATH, time):
    path_to_return = os.path.join(PATH, 'save_model_{}'.format(time)) + '/'
    if not os.path.exists(path_to_return):
        os.mkdir(path_to_return)
    return path_to_return


def process_image(obs):
    obs = color.rgb2gray(obs)
    return obs.reshape(-1, obs.shape[0], obs.shape[1])

def normailze(pixel):
    return (pixel - pixel.min()) / (pixel.max() - pixel.min())

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def seed_torch(seed=2331):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)         # 为当前CPU 设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)        # 为当前的GPU 设置随机种子
        torch.cuda.manual_seed_all(seed)        # 当使用多块GPU 时，均设置随机种子
        torch.backends.cudnn.deterministic = True       # 设置每次返回的卷积算法是一致的
        torch.backends.cudnn.benchmark = True      # cuDNN使用的非确定性算法自动寻找最适合当前配置的高效算法，设置为False
        torch.backends.cudnn.enabled = True        # pytorch使用cuDNN加速