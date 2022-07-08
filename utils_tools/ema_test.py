# -*-  coding=utf-8 -*-
# @Time : 2022/6/22 16:28
# @Author : Scotty1373
# @File : ema_test.py
# @Software : PyCharm
import time

import numpy as np
import matplotlib.pyplot as plt
import copy

tua = 0.89

def soft_update(source, target):
    return source * tua + target * (1 - tua)


if __name__ == '__main__':
    x = [0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 10, 20, 30, 5]
    x = np.array(x, dtype=np.float32)
    y = copy.deepcopy(x)
    for idx, data in enumerate(x[1:]):
        y[idx + 1] = soft_update(y[idx], data)
    time.time()

    plt.plot(np.arange(len(x)), x)
    plt.plot(np.arange(len(x)), y)
    plt.show()




