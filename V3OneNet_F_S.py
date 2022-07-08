import base64
import io
import random
import socket
import sys
import time
from collections import deque

import numpy as np
import skimage
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense, Conv2D, Flatten, concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from skimage import color
import matplotlib.pyplot as plt

EPISODES = 1000000000
img_rows, img_cols = 80, 80
# Convert image into gray scale
img_channels = 4  # We stack 8 frames, 0.06*8 sec


class DQNAgent:
    def __init__(self, _state_size, _action_size):
        self.t = 0
        self.max_Q = 0
        self.trainingLoss = 0
        self.train = True
        # Get size of state and action
        self.state_size = _state_size
        self.action_size = _action_size

        # These are hyper parameters(超参数) for the DQN(强化学习)
        self.discount_factor = 0.99  # 折现系数(听着像金融里面的) 折扣系数
        self.learning_rate = 1e-4  # 学习率
        if self.train:  # 如果是训练模型
            self.epsilon = 1.0  # ε等于1，初始化的ε等于1( ε 是什么来着) 贪婪系数
            self.initial_epsilon = 1.0  # 否则两个为0
        else:
            self.epsilon = 0
            self.initial_epsilon = 0
        self.epsilon_min = 0.01  # ε 最小值为0.01
        self.batch_size = 64  # 一次读取64张图片
        self.train_start = 100  # 训练开始为100...100?
        self.explore = 4000  # 探索为4000...emmmmmm探索?

        # Create replay memory using deque(双端队列，元素个数为32000)
        self.memory = deque(maxlen=32000)

        # Create main model and target model(终于找到了我们的模型了，调用了下面的创建模型方法)
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Copy the model to target model
        # --> initialize the target model so that the parameters of model & target model to be same
        # 模型和目标模型的参数值是一样的
        # 升级模型0.0
        self.update_target_model()

    def build_model(self):
        print("Now we build the model")

        # define two sets of inputs
        # 这里为什么不用全局的变量来做呢?
        # input_a = Input(shape=(80, 80, img_channels))
        #
        # input_b = Input(shape=(img_channels,))
        # input_c = Input(shape=(img_channels,))
        # # the first branch operates on the first input
        # # keras里面的2维卷积，卷积卷积卷积卷积，激活函数用relu，最后输出...输出多少维的来着...
        # # 第一层输出个数为16 卷积窗口为8*8 卷积沿宽度和高度的步长郡为4
        # # (80 - 8 - 2 * 0) / 4 + 1 = 19     x = [19 19 16]
        # x = Conv2D(16, (8, 8), strides=(4, 4), activation="relu")(input_a)
        # # 第二层输出个数为32 卷积窗口为4*4 卷积步长为4 padding(补零策略) valid不填充 same填充
        # # (19 - 4 + 1) / 4 + 1 = 5          x = [5 5 32]
        # x = Conv2D(32, (4, 4), strides=(4, 4), padding='same', activation="relu")(x)
        #
        # x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation="relu")(x)
        #
        # x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation="relu")(x)
        # # 将128个元素展平为1维的向量方便全连接
        # x = Flatten()(x)
        # # 128个节点，激活函数为relu
        # x = Dense(1024, activation="relu")(x)
        # # 最后出来一个21位向量
        # x = Dense(512)(x)
        # x = Model(inputs=input_a, outputs=x)
        #
        # vect_concat = concatenate([input_b, input_c], axis=-1)
        #
        # # the second branch operates on the second input
        # y = Dense(100)(vect_concat)
        # y = Dense(300)(y)
        # y = Model(inputs=vect_concat, outputs=y)
        #
        # # combine the output of the two branches
        # feature_fusion = concatenate([x.output, y.output])
        # action = Dense(21*21)(feature_fusion)
        # action_model = Model(inputs=feature_fusion, outputs=action)

        model = DDQN_Net()
        # 使用亚当优化器优化
        adam = Adam(lr=self.learning_rate)
        # 损失函数为mean_squared_error-->均方误差
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        print("We finished building the model")
        return model

    @staticmethod
    def process_image(obs):
        obs = skimage.color.rgb2gray(obs)
        return obs
        # camera_info = CamInfo({
        #     "f_x": 500/5*8,         # focal length x
        #     "f_y": 500/5*8,         # focal length y
        #     "u_x": 200,             # optical center x
        #     "u_y": 200,             # optical center y
        #     "camera_height": 1400,  # camera height in `mm`
        #     "pitch": 90,            # rotation degree around x
        #     "yaw": 0                # rotation degree around y
        # })
        # ipm_info = CamInfo({
        #     "input_width": 400,
        #     "input_height": 400,
        #     "out_width": 80,
        #     "out_height": 80,
        #     "left": 0,
        #     "right": 400,
        #     "top": 200,
        #     "bottom": 400
        # })
        # ipm_img = IPM(camera_info, ipm_info)
        # out_img = ipm_img(obs)
        # if gap < 10:
        #     scikit-image.io.im_save('out_image_' + str(gap) + '.png', out_img)

        # return out_img

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, _input):
        if np.random.rand() <= self.epsilon:
            # print("Return Random Value")
            # return random.randrange(self.action_size)
            return np.random.uniform(-1, 1, 2)
        else:
            # print("Return Max Q Prediction")
            q_value = self.model.predict(_input)

            # Convert q array to steering value
            return linear_unbin(q_value[0])

    def replay_memory(self, state, _v_ego, _angle_ego, _action, steer, _reward, next_state, _next_v_ego, next_angle_ago,
                      _done):
        self.memory.append(
            (state, _v_ego, _angle_ego, _action, steer, _reward, next_state, _next_v_ego, next_angle_ago, _done))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.initial_epsilon - self.epsilon_min) / self.explore

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        state_t, _v_ego_t, _angle_ego_t, action_t, steer_t, reward_t, state_t1, _v_ego_t1, _angle_ego_t1, terminal = zip(
            *mini_batch)

        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)

        _v_ego_t = np.concatenate(_v_ego_t)
        _v_ego_t1 = np.concatenate(_v_ego_t1)

        _angle_ego_t = np.concatenate(_angle_ego_t)
        _angle_ego_t1 = np.concatenate(_angle_ego_t1)

        targets = self.model.predict([state_t, _v_ego_t, _angle_ego_t])
        self.max_Q = np.max(targets[0])
        target_val = self.model.predict([state_t1, _v_ego_t1, _angle_ego_t1])
        target_val_ = self.target_model.predict([state_t1, _v_ego_t1, _angle_ego_t1])

        for i in range(batch_size):
            if terminal[i] == 1:
                targets[i][action_t[i]] = reward_t[i]
                targets[i][steer_t[i]] = reward_t[i]
            else:
                # a = np.argmax(target_val[i])
                force, steer = np.unravel_index(target_val[i].argmax(), target_val[i].shape)
                targets[i][action_t[i]] = reward_t[i] + self.discount_factor * (target_val_[i][force][steer])

        loss = self.model.train_on_batch([state_t, _v_ego_t, _angle_ego_t], targets)
        self.trainingLoss = loss
        time.time()

    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


class DDQN_Net(keras.models.Model):
    def __init__(self):
        super(DDQN_Net, self).__init__()
        layers = [Conv2D(16, (8, 8), (4, 4), activation='relu'),
                  Conv2D(32, (4, 4), (4, 4), activation='relu'),
                  Conv2D(64, (3, 3), (1, 1), activation='relu'),
                  Conv2D(128, (3, 3), (1, 1), activation='relu'),
                  Flatten(),
                  Dense(1024, activation='relu'),
                  Dense(512, activation='relu')]

        self.extractor = keras.models.Sequential(*layers)

        vect_layer = [Dense(100, activation="relu"),
                      Dense(300, activation='relu')]
        self.vect = keras.models.Sequential(*vect_layer)

        self.fusion_fc1 = Dense(400, activation='relu')
        self.fusion_fc2 = Dense(21*21)

    def call(self, input):
        feature = self.extractor(input[0])
        state_vect = np.concatenate((input[1], input[2]), axis=-1)
        state_vect = self.vect(state_vect)
        fusion_feature = np.concatenate((feature, state_vect), axis=-1)
        fusion_feature = self.fusion_fc1(fusion_feature)
        fusion_feature = self.fusion_fc2(fusion_feature)
        return fusion_feature

# Utils Functions #

def linear_bin(a):
    """
    Convert a value to a categorical array.
    Parameters
    ----------
    a : int or float
        A value between -1 and 1
    Returns
    -------
    list of int
        A list of length 21 with one item set to 1, which represents the linear value, and all other items set to 0.
    """
    a = a + 1
    b = round(a / (2 / 20))
    arr = np.zeros(21)
    arr[int(b)] = 1
    return arr


def linear_unbin(arr):
    """
    Convert a categorical array to value.
    See Also
    --------
    linear_bin
    """
    if not len(arr) == 21:
        raise ValueError('Illegal array length, must be 21')

    force, steer = np.unravel_index(arr.argmax(), arr.shape)
    force = force * 2 / 20 - 1
    steer = steer * 2 / 20 - 1
    return force, steer


# def observe():
#   _rec_data, (remoteHost, remotePort) = sock.recv_from(65536)

def decode(_rec_data, _x_gap=0, _v_ego=0, _force=0, _angle_ego=0, _steer=0, _episode_len=0):
    # received data processing
    # 用逗号将接收的数据分隔，分隔为4个不同的部分
    rec_list = str(_rec_data).split(',', 7)
    x_gap = rec_list[0][2:]  # 纵向
    z_gap = rec_list[1]  # 横向
    v_ego1 = rec_list[2]  # speed of egoVehicle
    v_lead1 = rec_list[3]  # acceleration of egoVehicle
    angle_ego = rec_list[4]
    done = int(rec_list[5])
    img = base64.b64decode(rec_list[6])  # image from mainCamera
    angle_lead = rec_list[7]
    _image = Image.open(io.BytesIO(img))
    # BILINEAR 双线性变换
    _image = _image.resize((80, 80), resample=Image.BILINEAR)
    _image = tf.image.per_image_standardization(_image)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(_image)
    # plt.ion()
    # plt.show()
    # plt.pause(0)
    x_gap = float(x_gap) - 12  # -车身

    leadspeed_real = (x_gap - float(_x_gap)) / 2 + float(v_ego1)
    _image = np.array(_image)
    # _done = 0
    # 最终输出的_reward有1 0.5 0三个值
    gap = x_gap  # -车身
    _reward = cal_reward(gap, float(_v_ego), _force)
    reward_z = cal_reward_steer(float(z_gap), float(_angle_ego), _steer)
    if done == 1:
        reward_z = -1.0

    if gap <= 3 or gap >= 300:
        done = 1
        _reward = -1.0
    elif _episode_len > 480:
        done = 2
        # reward = CalReward(float(gap), float(v_ego), float(v_lead), force)
    R = _reward * reward_z
    return _image, R, done, gap, float(v_ego1), float(leadspeed_real), float(z_gap), float(angle_ego), float(
        angle_lead[:-1])


def cal_reward(_gap, _v_ego, _force):
    # if _force > 0:
    #     a = 3.5 * _force * 1.5
    # else:
    #     a = 5.5 * _force * 1.5

    if _force > 0:
        a = 3.5 * _force
    else:
        a = _force
    # reward for gap
    if 40 <= _gap <= 60:
        rd = 1
    elif 30 <= _gap < 40:
        rd = 0.5
    elif 60 < _gap <= 100:
        rd = 0.5
    else:
        rd = 0
    # return rp * rd / 195.0
    return rd


def cal_reward_steer(_gap, _angle_ego, _angle):
    # if _force > 0:
    #     a = 3.5 * _force * 1.5
    # else:
    #     a = 5.5 * _force * 1.5

    # reward for gap
    if -0.5 <= _gap <= 0.5:
        rd = 1
    elif -1 <= _gap < -0.5 or -0.5 <= _gap < 1:
        rd = 0.5
    else:
        rd = 0
    if 89 <= _angle_ego <= 91:
        rp = 1
    elif 85 <= _angle_ego < 89 or 91 < _angle_ego <= 95:
        rp = 0.5
    else:
        rp = 0
    # return rp * rd / 195.0
    return rd * rd


def reset():
    # if done == 1:
    _str_r = str(3) + ',' + '0.0' + ',' + '0.0'
    # sendDataLen =
    sock.sendto(_str_r.encode(), (remoteHost, remotePort))
    # if done == 2:
    #     _str_r = str(4) + ',' + str(action)


def print_out(file, _text):
    file.write(_text + '\n')
    file.flush()
    print(_text)
    sys.stdout.flush()


if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 8001))
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    # k.set_session(sess)

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)  # 图像长和宽为80， 有4个通道
    action_size = 21  # env.action_space.n # Steering and Throttle

    agent = DQNAgent(state_size, action_size)

    episodes = []
    train_log = open('train_log.txt', 'w')

    # 如果不是训练，直接载入模型
    if not agent.train:
        print("Now we load the saved model")
        agent.load_model("./v3_save_model/save_model_330.h5")

    t = time.time()
    done = 0
    only_reset_loc = 0
    for e in range(EPISODES):
        print("Episode: ", e)
        if done == 2:
            print("new continued episode!")
            done = 0
            episode_len = 0
            only_reset_loc = 1
        else:
            print("new fresh episode!")
            # 返回接收的数据 和地址端口
            recData, (remoteHost, remotePort) = sock.recvfrom(65536)
            # 对返回的数据进行解码
            image, _, _, x_gap, v_ego, v_lead1, z_gap, angle_ego, angle_lead = decode(recData)

            done = 0
            episode_len = 0

            # 彩色图转灰度图
            x_t = agent.process_image(image)

            # 将4个x_t依照第二个维度堆叠
            s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
            v_ego_t = np.array((v_ego, v_ego, v_ego, v_ego))
            angle_ego_t = np.array((angle_ego, angle_ego, angle_ego, angle_ego))
            # In Keras, need to reshape
            s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4
            v_ego_t = v_ego_t.reshape(1, v_ego_t.shape[0])  # 1*8
            angle_ego_t = angle_ego_t.reshape(1, angle_ego_t.shape[0])
        while done == 0:
            if agent.t % 1000 == 0:
                rewardTot = []

            episode_len = episode_len + 1
            # Get action for the current state and go one step in environment
            force, steer = agent.get_action([s_t, v_ego_t, angle_ego_t])

            if only_reset_loc == 1:
                str_r = str(4) + ',' + str(force) + ',' + str(steer)
                only_reset_loc = 0
            else:
                str_r = str(1) + ',' + str(force) + ',' + str(steer)

            # 在这里将数据发送给unity
            sendDataLen = sock.sendto(str_r.encode(), (remoteHost, remotePort))  # 0.06s later receive
            # 再次接收，这里接收的是什么？
            recData, (remoteHost, remotePort) = sock.recvfrom(65536)

            image, reward, done, x_gap, v_ego1, v_lead1, z_gap, angle_ego1, angle_lead1 = decode(recData,
                                                                                                 x_gap,
                                                                                                 v_ego,
                                                                                                 force,
                                                                                                 angle_ego,
                                                                                                 steer,
                                                                                                 episode_len)
            rewardTot.append(reward)

            x_t1 = agent.process_image(image)
            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)  # 1x80x80x4
            v_ego_1 = np.array(v_ego1)
            v_ego_1 = np.expand_dims(v_ego_1, -1)
            v_ego_1 = np.expand_dims(v_ego_1, -1)
            v_ego_t1 = np.append(v_ego_1, v_ego_t[:, :3], axis=1)  # 1x8

            angle_ego_1 = np.array(angle_ego1)
            angle_ego_1 = np.expand_dims(angle_ego_1, -1)
            angle_ego_1 = np.expand_dims(angle_ego_1, -1)
            angle_ego_t1 = np.append(angle_ego_1, angle_ego_t[:, :3], axis=1)  # 1x8

            if agent.train:
                # Save the sample <s, a, r, s'> to the replay memory
                agent.replay_memory(s_t, v_ego_t, angle_ego_t, np.argmax(linear_bin(force)),
                                    np.argmax(linear_bin(steer)), reward, s_t1, v_ego_t1, angle_ego_t1, done)
                agent.train_replay()

            s_t = s_t1
            v_ego_t = v_ego_t1
            v_ego = v_ego_1
            agent.t = agent.t + 1

            print("EPISODE", e, "TIME_STEP", agent.t, "/ Force", force, "/ REWARD", reward, "Avg REWARD:",
                  "/ Steer", steer,
                  "/ EPISODE LENGTH", episode_len, "/ v_ego", v_ego1, "/ angle_ego", angle_ego1,
                  "/ v_lead ", v_lead1, "/ angle_lead ", angle_lead1)
            format_str = ('EPISODE: %d TIMESTAMP: %d EPISODE_LENGTH: %d FORCE: %.4f REWARD: %.4f '
                          'Avg_REWARD: %.4f training_Loss: %.4f Q_MAX: %.4f gap_x: %.4f  v_ego: %.4f '
                          'v_lead: %.4f '
                          'STEER: %.4f  '
                          'gap_z: %.4f  angle_ego: %.4f angle_lead: %.4f '
                          'time: %.0f ')
            text = (format_str % (
                e, agent.t, episode_len, force, reward, sum(rewardTot) / len(rewardTot),
                agent.trainingLoss * 1e3,
                agent.max_Q, x_gap, v_ego1, v_lead1, steer,
                z_gap, angle_ego1, angle_lead1, time.time() - t))
            print_out(train_log, text)
            if done:
                # Every episode update the target model to be same with model
                agent.update_target_model()

                episodes.append(e)
                # Save model for every 2 episodes
                if agent.train and (e % 2 == 0):
                    agent.save_model("./v3_save_model/save_model_{}.h5".format(e))

                print("episode:", e, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon, " episode length:", episode_len)
                if done == 1:
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    reset()

    time.time()
