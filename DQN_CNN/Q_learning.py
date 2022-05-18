# # -*- coding: utf-8 -*-
# """
# Created on Wed Jan 27 21:10:06 2021
#
# @author: pang
# """
#
# import numpy as np
# import time
# import random
# from DQN_tensorflow_gpu import DQN
# import os
# import pandas as pd
# import random
# import tensorflow.compat.v1 as tf
# import Tank
# import Defender
# def self_blood_count(self_gray):
#     self_blood = 0
#     for self_bd_num in self_gray[469]:
#         # self blood gray pixel 80~98
#         # 血量灰度值80~98
#         if self_bd_num > 90 and self_bd_num < 98:
#             self_blood += 1
#     return self_blood
#
#
# def boss_blood_count(boss_gray):
#     boss_blood = 0
#     for boss_bd_num in boss_gray[0]:
#         # boss blood gray pixel 65~75
#         # 血量灰度值65~75
#         if boss_bd_num > 65 and boss_bd_num < 75:
#             boss_blood += 1
#     return boss_blood
#
#
# WIDTH = 96
# HEIGHT = 88
# window_size = (320, 100, 704, 452)  # 384,352  192,176 96,88 48,44 24,22
# # station window_size
#
# blood_window = (60, 91, 280, 562)
# # used to get boss and self blood
#
# action_size = 4
# # action[n_choose,j,k,m,r]
# # j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing
#
# EPISODES = 3000
# big_BATCH_SIZE = 16
# UPDATE_STEP = 50
# # times that evaluate the network
# num_step = 0
# # used to save log graph
# target_step = 0
# # used to update target Q network
# paused = True
# # used to stop training
#
# def train():
#     agent = DQN(WIDTH, HEIGHT, action_size, DQN_model_path, DQN_log_path)
#     # DQN init
#     # paused at the begin
#     emergence_break = 0
#     # emergence_break is used to break down training
#     # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
#     for episode in range(EPISODES):
#         station = map
#         # change graph to WIDTH * HEIGHT for station input
#         boss_blood = boss_blood_count(blood_window_gray)
#         self_blood = self_blood_count(blood_window_gray)
#         # count init blood
#         target_step = 0
#         # used to update target Q network
#         done = 0
#         total_reward = 0
#         stop = 0
#         # 用于防止连续帧重复计算reward
#         last_time = time.time()
#         while True:
#             station = np.array(station).reshape(-1, HEIGHT, WIDTH, 1)[0]
#             # reshape station for tf input placeholder
#             print('loop took {} seconds'.format(time.time() - last_time))
#             last_time = time.time()
#             target_step += 1
#             # get the action by state
#             action = agent.Choose_Action(station)
#             take_action(action)
#             next_station = np.array(next_station).reshape(-1, HEIGHT, WIDTH, 1)[0]
#             next_boss_blood = boss_blood_count(blood_window_gray)
#             next_self_blood = self_blood_count(blood_window_gray)
#             reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
#                                                                self_blood, next_self_blood,
#                                                                stop, emergence_break)
#             # get action reward
#             if emergence_break == 100:
#                 # emergence break , save model and paused
#                 # 遇到紧急情况，保存数据，并且暂停
#                 print("emergence_break")
#                 agent.save_model()
#                 paused = True
#             agent.Store_Data(station, action, reward, next_station, done)
#             if len(agent.replay_buffer) > big_BATCH_SIZE:
#                 num_step += 1
#                 # save loss graph
#                 # print('train')
#                 agent.Train_Network(big_BATCH_SIZE, num_step)
#             if target_step % UPDATE_STEP == 0:
#                 agent.Update_Target_Network()
#                 # update target Q network
#             station = next_station
#             self_blood = next_self_blood
#             boss_blood = next_boss_blood
#             total_reward += reward
#             paused = pause_game(paused)
#             if done == 1:
#                 break
#         if episode % 10 == 0:
#             agent.save_model()
#             # save model
#         print('episode: ', episode, 'Evaluation Average Reward:', total_reward / target_step)
#         restart()
#
#
#
#
#
#
#
#
#
