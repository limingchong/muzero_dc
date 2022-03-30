'''
    tank_battle.py
'''
import threading
import time

from tkinter import *
import games.Tank.INIT_STATES as INIT_STATES
import muzero
from games.Tank.DeadWall import DeadWall
from games.Tank.Empty import Empty
from games.Tank.GUI import GUI
from games.Tank.LiveWall import LiveWall
from games.Tank.Tank import Tank

END_SHOW = True
PAUSE_TIME = 0.1
UNIT_SIZE = 15
EPOCH = 1000


class tank_battle():
    def __init__(self, root):
        self.root = root
        self.last_press_time = 0
        root.clear_all()

    def train(self):
        muzero.MuZero("tank_battle")

    def test(self):
        self.states = []
        self.board = INIT_STATES.INIT_STATES
        tanks = []
        game_time = 0
        self.root.bind_all("<Key>", self.key_press)

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

        self.canvas = GUI(self.root, self.states, 22, 22, 30, UNIT_SIZE)
        self.me = tanks[0]
        self.testing = True
        worker = Worker(PAUSE_TIME, tanks[1], self.states)
        worker.start()

        while self.testing:
            self.canvas.render(self.states)

    def key_press(self, e):
        if time.time() - self.last_press_time> PAUSE_TIME:
            if e.keysym == "space":
                self.me.shoot(self.states, self.board)

            if e.keysym == "Up" and self.me.forwardTest(self.states):
                self.me.forward(self.states, self.board)

            if e.keysym == "Left":
                self.me.rotate(-1, self.board)

            if e.keysym == "Right":
                self.me.rotate(1, self.board)

            if e.keysym == "Escape":
                self.testing = False
                print("close")
                self.root.setObjects()

            self.last_press_time = time.time()


class Worker(threading.Thread):
    def __init__(self, delay, tank, states):
        threading.Thread.__init__(self)
        threading.Thread()
        self.tank = tank
        self.states = states
        self.delay = delay

    def run(self):
        while type(self.states[self.tank.x][self.tank.y]) == Tank:
            self.tank.random_act(self.states)
            time.sleep(self.delay)
        else:
            print("Tank 0 win.")
