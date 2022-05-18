import time
from DQN_tensorflow_gpu import DQN
from Tank import *
import tensorflow.compat.v1 as tf
from Defender import *
from init import *
import matplotlib.pyplot as plt
import tkinter as tk
import matplotlib.animation as animation

map_width = 20
map_height = 20
fig = plt.figure()
action_size = 5
tankDeath = 0
iteration = 0
exposide = 0

# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# sess1 = tf.InteractiveSession(config=config)
# sess2 = tf.Session(config=config)

for i in range(maxX):
    for j in range(maxY):
        if initialMap[i][j] == 1:
            # agent = DQN(map_width, map_height, action_size, "model_gpu", "logs_gpu/", sess1)
            tank.append(Tank(i, j, 2, 0))
        if initialMap[i][j] == 4:
            # agent1 = DQN(map_width, map_height, action_size, "model_gpu1", "logs_gpu1/", sess2)
            defender.append(Defender(i, j, 0, 0))


def judgeSurvive(object):
    if object.hp <= 0:
        return False
    return True


def initialGame():
    for i in range(maxX):
        for j in range(maxY):
            map[i][j] = initialMap[i][j]
    for k in range(len(tank)):
        tank[k].x = tank[k].initialX
        tank[k].y = tank[k].initialY
        tank[k].survive = True
        tank[k].hp = 3
    for k in range(len(defender)):
        defender[k].x = defender[k].initialX
        defender[k].y = defender[k].initialY
        defender[k].survive = True
        defender[k].hp = 3
    fo1 = open("recordsDefenderActions.csv", "a")
    fo1.write("\n")
    fo1.close()
    fo2 = open("recordsTankActions.csv", "a")
    fo2.write("\n")
    fo2.close()

UNIT = 40  # 每个迷宫单元格的宽度
MAZE_H = maxY  # 迷宫的高度
MAZE_W = maxX  # 迷宫的宽度


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ["up", "down", "left", "right"]  # 定义动作列表，有四种动作，分别为上下左右
        self.n_actions = len(self.action_space)
        self.title("Maze")
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))  # 设置迷宫大小
        self.allObject = list()
        self._build_maze()

    def _build_maze(self):
        """构建迷宫
        """
        # 设置迷宫界面的背景
        self.canvas = tk.Canvas(self, bg="white",
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # 划分迷宫单元格，即根据坐标位置划线
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

    def render(self):
        self.update()
        for child in self.allObject:
            self.canvas.delete(child)
        self.allObject.clear()

        # 起点位置
        origin = np.array([20, 20])

        for i in range(maxX):
            for j in range(maxY):
                if map[i][j] == 1:
                    trap1_center = origin + np.array([UNIT * i, UNIT * j])
                    for k in range(len(tank)):
                        if tank[k].x == i and tank[k].y == j:
                            if tank[k].faceState == 0:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] - 15, trap1_center[1] - 15,
                                    trap1_center[0] + 15, trap1_center[1] - 15,
                                    trap1_center[0] + 0, trap1_center[1] + 15,
                                    fill='blue')
                            elif tank[k].faceState == 1:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] - 15, trap1_center[1] - 15,
                                    trap1_center[0] - 15, trap1_center[1] + 15,
                                    trap1_center[0] + 15, trap1_center[1] + 0,
                                    fill='blue')
                            elif tank[k].faceState == 2:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] - 15, trap1_center[1] + 15,
                                    trap1_center[0] + 15, trap1_center[1] + 15,
                                    trap1_center[0] + 0, trap1_center[1] - 15,
                                    fill='blue')
                            else:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] + 15, trap1_center[1] - 15,
                                    trap1_center[0] + 15, trap1_center[1] + 15,
                                    trap1_center[0] - 15, trap1_center[1] + 0,
                                    fill='blue')
                            self.allObject.append(self.trap1)
                elif map[i][j] == 2:
                    trap1_center = origin + np.array([UNIT * i, UNIT * j])
                    self.trap1 = self.canvas.create_rectangle(
                        trap1_center[0] - 15, trap1_center[1] - 15,
                        trap1_center[0] + 15, trap1_center[1] + 15,
                        fill='yellow')
                    self.allObject.append(self.trap1)
                elif map[i][j] == 3:
                    trap1_center = origin + np.array([UNIT * i, UNIT * j])
                    self.trap1 = self.canvas.create_rectangle(
                        trap1_center[0] - 15, trap1_center[1] - 15,
                        trap1_center[0] + 15, trap1_center[1] + 15,
                        fill='black')
                    self.allObject.append(self.trap1)
                elif map[i][j] == 4:
                    trap1_center = origin + np.array([UNIT * i, UNIT * j])
                    for k in range(len(defender)):
                        if defender[k].x == i and defender[k].y == j:
                            if defender[k].faceState == 0:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] - 15, trap1_center[1] - 15,
                                    trap1_center[0] + 15, trap1_center[1] - 15,
                                    trap1_center[0] + 0, trap1_center[1] + 15,
                                    fill='green')
                            elif defender[k].faceState == 1:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] - 15, trap1_center[1] - 15,
                                    trap1_center[0] - 15, trap1_center[1] + 15,
                                    trap1_center[0] + 15, trap1_center[1] + 0,
                                    fill='green')
                            elif defender[k].faceState == 2:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] - 15, trap1_center[1] + 15,
                                    trap1_center[0] + 15, trap1_center[1] + 15,
                                    trap1_center[0] + 0, trap1_center[1] - 15,
                                    fill='green')
                            else:
                                self.trap1 = self.canvas.create_polygon(
                                    trap1_center[0] + 15, trap1_center[1] - 15,
                                    trap1_center[0] + 15, trap1_center[1] + 15,
                                    trap1_center[0] - 15, trap1_center[1] + 0,
                                    fill='green')
                            self.allObject.append(self.trap1)
                elif map[i][j] == 5:
                    trap1_center = origin + np.array([UNIT * i, UNIT * j])
                    self.trap1 = self.canvas.create_oval(
                        trap1_center[0] - 15, trap1_center[1] - 15,
                        trap1_center[0] + 15, trap1_center[1] + 15,
                        fill='black')
                    self.allObject.append(self.trap1)

        # 组合所有元素
        self.canvas.pack()


env = Maze()

while True:

    if done:
        initialGame()
        tankDeath = 0

        for i in range(len(tank)):
            print('\033[0;36;40m', end="")
            print("the tank's average_reward", i, tank[i].reward / iteration)
            fo = open("recordsSame2.csv", "a")
            fo.write(str(tank[i].reward) + " ")
            print('\033[0m', end="")
            tank[i].reward = 0
            tank[i].station = np.array(tank[i].station).reshape(-1, map_height, map_width, 15)[0]
            # reshape station for tf input placeholder
            print('loop took {} seconds'.format(time.time() - tank[i].last_time))
            tank[i].last_time = time.time()
            tank[i].target_step += 1
            tank[i].agent.save_model()

        for i in range(len(defender)):
            print('\033[0;37;40m', end="")
            print("the defender's average_reward", i, defender[i].reward / iteration)
            fo.write(str(defender[i].reward) + " " + str(iteration) + "\n")
            fo.close()
            print('\033[0m', end="")
            defender[i].reward = 0
            defender[i].station = np.array(defender[i].station).reshape(-1, map_height, map_width, 15)[0]
            # reshape station for tf input placeholder
            print('loop took {} seconds'.format(time.time() - defender[i].last_time))
            defender[i].last_time = time.time()
            defender[i].target_step += 1
            defender[i].agent.save_model_defender()

        # for i in range(len(defender)):
        #     print("the %d's defender's average_reward", i, defender[i].reward / iteration)
        iteration = 0
        done = False
    for i in range(len(tank)):
        if not judgeSurvive(tank[i]):
            if tank[i].survive:
                tank[i].survive = False
                map[tank[i].x][tank[i].y] = 0
                tankDeath += 1
    for i in range(len(defender)):
        if not judgeSurvive(defender[i]):
            if defender[i].survive:
                defender[i].survive = False
                map[defender[i].x][defender[i].y] = 0
                tankDeath += 1

    if tankDeath >= 1:
        done = True
    if iteration >= 1000:
        done = True
        print("timeUp")
        print("tank:"+str(tank[0].hp)+",defender:"+str(defender[0].hp))
        tank[0].reward -= 500
        defender[0].reward -= 500
    iteration += 1
    for i in range(len(tank)):
        tank[i].train()
    # for i in range(len(defender)):
    # defender[i].train(exposide)
    env.render()
