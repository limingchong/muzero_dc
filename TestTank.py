import time

import numpy as np

from Tank.DeadWall import DeadWall
from Tank.Empty import Empty
from Tank.GUI import GUI
import Tank.INIT_STATES as INIT_STATES
from Tank.LiveWall import LiveWall
from Tank.Tank import Tank

END_SHOW = True
PAUSE_TIME = 0
UNIT_SIZE = 15
EPOCH = 1000

gui = GUI(22, 22, 30, UNIT_SIZE)
actions0 = []
actions1 = []
ends = []

for epoch in range(EPOCH):

    states = []
    tanks = []
    game_time = 0
    row0 = []
    row1 = []

    for i in range(22):
        row = []
        for j in range(22):
            if INIT_STATES.INIT_STATES[i][j] == 1:
                row.append(LiveWall(i, j, UNIT_SIZE))

            elif INIT_STATES.INIT_STATES[i][j] == 2:
                row.append(DeadWall(i, j, UNIT_SIZE))

            elif INIT_STATES.INIT_STATES[i][j] > 2:
                tank = Tank(i, j, UNIT_SIZE, INIT_STATES.INIT_STATES[i][j])
                tanks.append(tank)
                row.append(tank)
            else:
                row.append(Empty(i, j, UNIT_SIZE))
        states.append(row)

    while True:
        # print(game_time)
        game_time += 1
        if PAUSE_TIME >= 0:
            gui.render(states)
            time.sleep(PAUSE_TIME)

        if type(states[tanks[0].x][tanks[0].y]) == Tank:
            row0.append(tanks[0].random_act(states))
        else:
            ends.append(1)
            print("Epoch ", epoch, " Tank 1 win.")
            break

        if type(states[tanks[1].x][tanks[1].y]) == Tank:
            row1.append(tanks[1].random_act(states))
        else:
            ends.append(0)
            print("Epoch ", epoch, "Tank 0 win.")
            break

    actions0.append(row0)
    actions1.append(row1)

lens = []
for i in range(EPOCH):
    lens.append(len(actions0[i]))

print(max(lens), sum(lens)/EPOCH, min(lens))

while END_SHOW:
    gui.render(states)
