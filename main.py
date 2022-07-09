import base64
import copy
import datetime as dt
import io
import os
import random
import socket
import sys
import time
from DDPG.DDPG import DDPG
from utils_tools.utils import Model_save_Dir, log_File_path, print_out, seed_torch
import numpy as np
import torch

np.set_printoptions(precision=4)

# state vect structure:
# x_gap, v_ego, angle_ego, angle_lead, z_gap

EPISODES = 5000
img_rows, img_cols = 80, 80
state_length = 3
frame_overlay = 4
action_dim = 2

# Convert image into gray scale
# We stack 8 frames, 0.06*8 sec
img_channels = 4 
unity_Block_size = 65536
# PATH_MODEL = 'C:/dl_data/Python_Project/save_model/'
# PATH_LOG = 'C:/dl_data/Python_Project/train_log/'
CHECK_POINT_TRAIN_PATH = './save_Model/save_model_1631506899/save_model_398.h5'
PATH_MODEL = 'save_Model'
PATH_LOG = 'train_Log'
time_Feature = round(time.time())


if __name__ == "__main__":
    if not os.path.exists('./' + PATH_LOG):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_LOG))
    if not os.path.exists('./' + PATH_MODEL):
        os.mkdir(os.path.join(os.getcwd().replace('\\', '/'), PATH_MODEL))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 8001))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get size of state and action from environment
    state_size = (img_rows, img_cols, img_channels)
    action_size = (-1, 1)   # env.action_space.n # Steering and Throttle

    seed_torch(seed=25536)
    train_log = log_File_path(PATH_LOG)
    PATH_ = Model_save_Dir(PATH_MODEL, time_Feature)
    agent = DDPG(state_length=state_length,
                 frame_overlay=frame_overlay,
                 action_dim=2,
                 device_=device,
                 sock_udp=sock)
    episodes = []

    if not agent.train:
        print("Now we load the saved model")
        agent.load_model('C:/DRL_data/Python_Project/Enhence_Learning/save_Model/save_model_1627300305/save_model_398.h5')
    elif agent.train_from_checkpoint:
        agent.load_model(CHECK_POINT_TRAIN_PATH)
        print(f'Now we load checkpoints for continue training:  {CHECK_POINT_TRAIN_PATH.split("/")[-1]}')
    else:
        print('Thread Ready!!!')

    # initialize value
    done = 0
    counter_value = 0

    # state init
    pixel_state = None
    vect_state = None
    ep_history = []

    for e in range(EPISODES):      
        print("Episode: ", e)
        # ou noise重置
        # agent.acc_noise.reset()
        # agent.ori_noise.reset()

        if done == 2:
            print("new continued epicode!")
            done = 0
            UnityReset = 1
            episode_len = 0
        else:
            # 后期重置进入第一次recv
            print('done value:', done)
            print("new fresh episode!")
            UnityReset = 0
            episode_len = 0
            done = 1
            for _ in range(20):
                _, _ = agent.Recv_data_Format(done)
                UnityReset = agent.Send_data_Format(UnityResetFlag=UnityReset, skip=True)

            pixel_state, vect_state = agent.Recv_data_Format(done)
            done = 0

        while not done:
            # 跳过reset
            start_time = time.time()
            episode_len, action, UnityReset = agent.Send_data_Format(pixel_state, vect_state, episode_len, UnityReset)
            next_pixel, next_vect, reward, done, _ = agent.Recv_data_Format(done, pixel_state, vect_state, action, episode_len)
            ep_history.append(reward)
            start_count_time = int(round(time.time() * 1000))

            if agent.train:
                # s_t, v_ego_t, s_t1, v_ego_t1 = random_sample(s_t, v_ego_t, s_t1, v_ego_t1)
                agent.replay_memory(pixel_state, vect_state, action, reward, next_pixel, next_vect, done)
                agent.train_replay()

            pixel_state = next_pixel
            vect_state = next_vect
            agent.t = agent.t + 1

            vect4log = next_vect.squeeze()
            act4log = action.squeeze()
            try:
                print("EPISODE",  e, "TIMESTEP", agent.t,
                      "ACC", format(act4log[0].item(), '.2f'), " ORI", format(act4log[1].item(), '.2f'),
                      "REWARD", format(reward, '.2f'), "Avg REWARD:", sum(ep_history)/len(ep_history),
                      "EPISODE LENGTH", episode_len, 'loss_actor', format(agent.history_loss_actor, '.5f'), 'loss_critic', format(agent.history_loss_critic, '.4f'))
            except TypeError as e:
                print(TypeError, e)
            format_str = 'EPISODE: %d TIMESTEP: %d EPISODE_LENGTH: %d ACC: %.4f ORI: %.4f REWARD: %.4f Avg_REWARD: %.4f actor_loss: %.4f critic_loss: %.4f gap: %.4f  v_ego: %.4f  angle_ego: %.4f'
            text = (format_str % (e, agent.t, episode_len, act4log[0].item(), act4log[1].item(), reward, sum(ep_history)/len(ep_history),
                                  agent.history_loss_actor, agent.history_loss_critic,
                                  vect4log[4].item(),
                                  vect4log[1].item(),
                                  vect4log[2].item()))
            print_out(train_log, text)
            if done:
                episodes.append(e)
                # Save model for every 2 episode
                if agent.train and (e % 10 == 0):
                    agent.save_model(os.path.join(PATH_, "save_model_{}.h5".format(e)))
                print("episode:", e, "  memory length:", len(agent.memory), " episode length:", episode_len)
                if done == 1:
                    agent.reset()
                    time.sleep(0.5)
                ep_history.clear()
            # print('Data receive from unity, time:', int(round(time.time() * 1000) - start_count_time))

"""        
        # 迭代过程测试
        # 测试
        if e % 5 == 0 and e >= 40:
            if done == 2:
                print("!!!TESTING!!! Continued epicode for testing!!!")
                done = 0
                UnityReset = 1
                episode_len = 0
            else:
                print('done value:', done)
                print("!!! STARTING TEST !!!")
                done = 1
                s_t, v_ego_t, v_ego, v_lead, remoteHost, remotePort = Recv_data_Format(unity_Block_size, done)
                done = 0
                UnityReset = 0
                episode_len = 0

            while done == 0:
                episode_len, action, time_cost, UnityReset = Send_data_Format(remoteHost, remotePort, s_t, v_ego_t,
                                                                              episode_len, UnityReset)
                reward, done, gap, v_ego1, v_lead, a_ego1, v_ego_1, s_t1, v_ego_t1 = Recv_data_Format(
                    unity_Block_size,
                    done, v_ego,
                    v_lead, action,
                    episode_len, s_t,
                    v_ego_t)

                s_t = s_t1
                v_ego_t = v_ego_t1
                v_ego = v_ego_1
                agent.t = agent.t + 1

                print("!!!TESTING!!!", "EPISODE", e, "TIMESTEP", agent.t, "/ ACTION", action, "/ REWARD",
                      format(reward, '.4f'), "/ EPISODE LENGTH", episode_len, "/ GAP ",
                      gap, "v_ego", v_ego.item(), "v_lead", v_lead, "/ a_ego ", a_ego1, 'loss_actor',
                      agent.history_loss_actor, 'loss_critic', agent.history_loss_critic)
                format_str = f'EPISODE: {e} TIMESTEP: {agent.t} EPISODE_LENGTH: {episode_len} ' \
                             f'ACTION: {action:.4f} REWARD: {reward:.4f} ' \
                             f'Avg_REWARD: {sum(rewardTot) / len(rewardTot):.4f} gap: {gap:.4f} ' \
                             f'v_ego: {v_ego.item():.4f} v_lead: {v_lead:.4f} a_ego: {a_ego1:.4f} ' \
                             f'loss_actor: {agent.history_loss_actor:.4f} loss_critic: {agent.history_loss_critic:.4f}'
                print_out(test_log, format_str)

                if done:
                    # Save model for every 2 episode
                    print("episode:", e, "  memory length:", len(agent.memory), " episode length:", episode_len)
                    if done == 1:
                        reset()
                        time.sleep(0.5)
"""
