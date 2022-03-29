'''
    tank_battle.py
'''
import muzero
import time
from games.Tank.DeadWall import DeadWall
from games.Tank.Empty import Empty
from games.Tank.GUI import GUI
import games.Tank.INIT_STATES as INIT_STATES
from games.Tank.LiveWall import LiveWall
from games.Tank.Tank import Tank
import tkinter

END_SHOW = True
PAUSE_TIME = 0.01
UNIT_SIZE = 15
EPOCH = 1000


class tank_battle():
    def __init__(self, root):
        self.root = root
        root.clear_all()

    def train(self):
        muzero.MuZero("tank_battle")

    def test(self):
        self.states = []
        self.board = INIT_STATES.INIT_STATES
        tanks = []
        game_time = 0
        self.root = GUI(22, 22, 30, UNIT_SIZE)
        self.root.bind_all("<Key>",self.key_press)

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
            self.states.append(row)

        self.me = tanks[0]
        while True:
            game_time += 1
            if PAUSE_TIME >= 0:
                self.root.render(self.states)
                time.sleep(PAUSE_TIME)

            if type(self.states[tanks[0].x][tanks[0].y]) != Tank:
                print(" Tank 1 win.")
                break

            if type(self.states[tanks[1].x][tanks[1].y]) == Tank:
                tanks[1].random_act(self.states)
            else:
                print("Tank 0 win.")
                break

    def key_press(self, e):
        if e.keysym == "space":
            self.me.shoot(self.states, self.board)

        if e.keysym == "Up" and self.me.forwardTest(self.states):
            self.me.forward(self.states, self.board)

        if e.keysym == "Left":
            self.me.rotate(-1, self.board)

        if e.keysym == "Right":
            self.me.rotate(1, self.board)


