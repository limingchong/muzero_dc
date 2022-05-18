from init import *
import random as rd
from DQN_tensorflow_gpu import DQN

map_width = 20
map_height = 20
action_size = 5
import time

big_BATCH_SIZE = 16
num_step = 0
agent = DQN(map_width, map_height, action_size, 'model_gpu', "logs_gpu/")


class Tank:
    def __init__(self, x, y, faceState, reward):
        self.x = x
        self.y = y
        self.faceState = faceState
        self.reward = reward
        self.hp = 3
        self.survive = True
        self.initialX = x
        self.initialY = y
        # self_up = np.zeros([map_width, map_height])
        # self_down = np.zeros([map_width, map_height])
        # self_left = np.zeros([map_width, map_height])
        # self_right = np.zeros([map_width, map_height])
        # friend_up = np.zeros([map_width, map_height])
        # friend_down = np.zeros([map_width, map_height])
        # friend_left = np.zeros([map_width, map_height])
        # friend_right = np.zeros([map_width, map_height])
        # enermy_up = np.zeros([map_width, map_height])
        # enermy_down = np.zeros([map_width, map_height])
        # enermy_left = np.zeros([map_width, map_height])
        # enermy_right = np.zeros([map_width, map_height])
        # live_wall = np.zeros([map_width, map_height])
        # dead_wall = np.zeros([map_width, map_height])
        # home_pos = np.zeros([map_width, map_height])
        self.input_map = np.zeros([map_width, map_height, 15])
        self.station = self.input_map
        self.station = np.array(self.station).reshape(-1, map_height, map_width, 15)[0]
        self.agent = agent

    def take_action(self, action_state):
        # self.reward -= 1
        fo = open("recordsTankActions.csv", "a")
        fo.write(str(action_state) + " ")
        fo.close()
        if self.survive:
            if action_state == 0:
                self.moveForward()
            if action_state == 1 or action_state == 2:
                self.rotate(action_state)
            if action_state == 3:
                return
            if action_state == 4:
                self.fire()

    def moveForward(self):
        if self.faceState == 0:
            if self.y < maxY - 1:
                if map[self.x][self.y + 1] == 0:
                    map[self.x][self.y] = 0
                    map[self.x][self.y + 1] = 1
                    self.y += 1
        if self.faceState == 1:
            if self.x < maxX - 1:
                if map[self.x + 1][self.y] == 0:
                    map[self.x][self.y] = 0
                    map[self.x + 1][self.y] = 1
                    self.x += 1
        if self.faceState == 2:
            if self.y > 0:
                if map[self.x][self.y - 1] == 0:
                    map[self.x][self.y] = 0
                    map[self.x][self.y - 1] = 1
                    self.y -= 1
        if self.faceState == 3:
            if self.x > 0:
                if map[self.x - 1][self.y] == 0:
                    map[self.x][self.y] = 0
                    map[self.x - 1][self.y] = 1
                    self.x -= 1

    def rotate(self, actionState):
        if actionState == 1:
            if self.faceState != 0:
                self.faceState -= 1
            else:
                self.faceState = 3
        if actionState == 2:
            if self.faceState != 3:
                self.faceState += 1
            else:
                self.faceState = 0

    def fire(self):
        if self.faceState == 0:
            for i in range(self.y + 1, maxY):
                if map[self.x][i] == 1:
                    for k in range(len(tank)):
                        if tank[k].x == self.x & tank[k].y == i:
                            tank[k].hp -= 1
                            self.reward -= 50
                            break
                    break
                if map[self.x][i] == 2:
                    map[self.x][i] = 0
                    break
                if map[self.x][i] == 3:
                    break
                if map[self.x][i] == 4:
                    for k in range(len(defender)):
                        if defender[k].x == self.x & defender[k].y == i:
                            defender[k].hp -= 1
                            self.reward += 100
                            defender[k].reward -= 50
                            break
                    break

        if self.faceState == 1:
            for i in range(self.x + 1, maxX):
                if map[i][self.y] == 1:
                    for k in range(len(tank)):
                        if tank[k].x == i & tank[k].y == self.y:
                            tank[k].hp -= 1
                            self.reward -= 100
                            break
                    break
                if map[i][self.y] == 2:
                    map[i][self.y] = 0
                    break
                if map[i][self.y] == 3:
                    break
                if map[i][self.y] == 4:
                    for k in range(len(defender)):
                        if defender[k].y == self.y & defender[k].x == i:
                            defender[k].hp -= 1
                            self.reward += 100
                            defender[k].reward -= 50
                            break
                    break

        if self.faceState == 2:
            for i in range(self.y - 1, -1, -1):
                if map[self.x][i] == 1:
                    for k in range(len(tank)):
                        if tank[k].x == self.x & tank[k].y == i:
                            tank[k].hp -= 1
                            self.reward -= 50
                            break
                    break
                if map[self.x][i] == 2:
                    map[self.x][i] = 0
                    break
                if map[self.x][i] == 3:
                    break
                if map[self.x][i] == 4:
                    for k in range(len(defender)):
                        if defender[k].x == self.x & defender[k].y == i:
                            defender[k].hp -= 1
                            self.reward += 100
                            defender[k].reward -= 50
                            break
                    break

        if self.faceState == 3:
            for i in range(self.x - 1, -1, -1):
                if map[i][self.y] == 1:
                    for k in range(len(tank)):
                        if tank[k].x == i & tank[k].y == self.y:
                            tank[k].hp -= 1
                            self.reward -= 50
                            break
                    break
                if map[i][self.y] == 2:
                    map[i][self.y] = 0
                    break
                if map[i][self.y] == 3:
                    break
                if map[i][self.y] == 4:
                    for k in range(len(defender)):
                        if defender[k].y == self.y & defender[k].x == i:
                            defender[k].hp -= 1
                            self.reward += 100
                            defender[k].reward -= 50
                            break
                    break

    # DQN init
    # paused at the begin
    emergence_break = 0

    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    def change_map(self):

        for i in range(len(tank)):
            if tank[i].x != self.x or tank[i].y != self.y:
                if tank[i].faceState == 0:
                    # self.friend_up[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][4] = 1
                if tank[i].faceState == 1:
                    # self.friend_right[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][7] = 1
                if tank[i].faceState == 2:
                    # self.friend_down[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][5] = 1
                if tank[i].faceState == 3:
                    # self.friend_left[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][6] = 1
            else:
                if self.faceState == 0:
                    # self.self_up[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][0] = 1
                if self.faceState == 1:
                    # self.self_right[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][3] = 1
                if self.faceState == 2:
                    # self.self_down[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][1] = 1
                if self.faceState == 3:
                    # self.self_left[tank[i].x][tank[i].y] = 1
                    self.input_map[tank[i].x][tank[i].y][2] = 1
        for i in range(len(defender)):
            if defender[i].x != self.x or defender[i].y != self.y:
                if defender[i].faceState == 0:
                    # self.enermy_up[defender[i].x][defender[i].y] = 1
                    self.input_map[defender[i].x][defender[i].y][8] = 1
                if defender[i].faceState == 1:
                    # self.enermy_right[defender[i].x][defender[i].y] =1
                    self.input_map[defender[i].x][defender[i].y][11] = 1
                if defender[i].faceState == 2:
                    # self.enermy_down[defender[i].x][defender[i].y] = 1
                    self.input_map[defender[i].x][defender[i].y][9] = 1
                if defender[i].faceState == 3:
                    # self.enermy_left[defender[i].x][defender[i].y] = 1
                    self.input_map[defender[i].x][defender[i].y][10] = 1
        for i in range(maxX):
            for j in range(maxY):
                if map[i][j] == 2:
                    # self.live_wall[i][j] = 1
                    self.input_map[i][j][12] = 1
                if map[i][j] == 3:
                    # self.dead_wall[i][j] = 1
                    self.input_map[i][j][13] = 1
                if map[i][j] == 5:
                    # self.home_pos[i][j] = 1
                    self.input_map[i][j][14] = 1

        # for i in range(maxX):
        #     for j in range(maxY):
        #         if map[i][j] == 1:
        #             if i == self.x and j == self.y:
        #                 if self.faceState == 0:
        #                     self.self_up[i][j] = 1
        #                 elif self.faceState == 1:
        #                     self.self_right[i][j] = 1
        #                 elif self.faceState ==2
        #
        #
        #             else:

    # change graph to WIDTH * HEIGHT for station input
    # count init blood
    target_step = 0
    # used to update target Q network
    # 用于防止连续帧重复计算reward
    last_time = time.time()
    num_step = 0
    UPDATE_STEP = 50
    total_reward = 0
    target_step = 0

    last_time = time.time()

    def train(self):

        self.change_map()
        agent = self.agent
        # get the action by state

        action = agent.Choose_Action(self.station)
        self.take_action(action)

        for i in range(len(defender)):
            defender[i].change_map()
            action_defender = agent.Choose_Action_defender(defender[i].station)
            defender[i].take_action(action_defender)

        # next_station = map
        next_station = self.input_map
        next_station = np.array(next_station).reshape(-1, map_height, map_width, 15)[0]

        for i in range(len(defender)):
            next_station_defender = defender[i].input_map
            next_station_defender = np.array(next_station_defender).reshape(-1, map_height, map_width, 15)[0]

        # get action reward
        agent.Store_Data(self.station, action, self.reward, next_station, done)
        for i in range(len(defender)):
            agent.Store_Data_defender(defender[i].station, action_defender, defender[i].reward, next_station_defender,
                                      done)

        if len(agent.replay_buffer) > big_BATCH_SIZE:
            self.num_step += 1
            # save loss graph
            # print('train')
            agent.Train_Network(big_BATCH_SIZE, self.num_step)

        for i in range(len(defender)):
            if len(agent.replay_buffer_defender) > big_BATCH_SIZE:
                defender[i].num_step += 1
                # save loss graph
                # print('train')
                agent.Train_Network_defender(big_BATCH_SIZE, defender[i].num_step)

        if self.target_step % self.UPDATE_STEP == 0:
            agent.Update_Target_Network()
            # update target Q network
        self.station = next_station
        self.total_reward += self.reward

        for i in range(len(defender)):
            if defender[i].target_step % defender[i].UPDATE_STEP == 0:
                agent.Update_Target_Network_defender()
                # update target Q network
            defender[i].station = next_station_defender
            defender[i].total_reward += defender[i].reward
        # if episode % 10 == 0:
        # agent.save_model()
        # # save model
