"""
pong.py
"""
import random
import threading
import time

from games.pong_dic.Ball import Ball
from games.pong_dic.GUI import GUI
from games.pong_dic.Pat import Pat

PAUSE_TIME = 0.5


class pong:
    def __init__(self, root):
        self.testing = True
        self.root = root
        self.name = "pong"
        root.clear_all()

    def train(self):
        pass

    def test(self):
        # player, ball, AI
        self.states = [Pat(20, 300, 20), Ball(300, 300, 20), Pat(580, 300, 20)]
        self.canvas = GUI(self.root, self.states)

        self.last_press_time = 0
        self.root.games_frame.unbind_all("<Button>")
        self.root.bind_all("<Key>", self.key_press)
        self.states[1].f = 0
        self.states[1].v_x = -10

        worker = Worker(PAUSE_TIME, self.states[2], self.states, self)
        worker.start()

        while self.testing:
            self.bounce()
            if time.time() - self.last_press_time > PAUSE_TIME:
                self.states[0].a_y = 0
            self.canvas.render(self.states)

        self.root.clear_all()
        self.root.setObjects()

    def key_press(self, e):
        if time.time() - self.last_press_time > PAUSE_TIME:
            if e.keysym == "Up":
                self.states[0].a_y = -3

            if e.keysym == "Down":
                self.states[0].a_y = 3

            if e.keysym == "Escape":
                self.testing = False

            self.last_press_time = time.time()

    def bounce(self):
        if self.states[1].x < 50:
            if self.states[0].y - 3 * self.states[0].size < self.states[1].y < \
                    self.states[0].y + 3 * self.states[0].size:
                self.states[1].v_y += self.states[1].y - self.states[0].y
                self.states[1].v_x = -self.states[1].v_x

        if self.states[1].x > 550:
            if self.states[2].y - 3 * self.states[2].size < self.states[1].y < \
                    self.states[0].y + 3 * self.states[0].size:
                self.states[1].v_y += self.states[1].y - self.states[0].y
                self.states[1].v_x = -self.states[1].v_x


class Worker(threading.Thread):
    def __init__(self, delay, pat, states, root):
        threading.Thread.__init__(self)
        threading.Thread()
        self.pat = pat
        self.states = states
        self.delay = delay
        self.root = root

    def run(self):
        while self.root.testing:
            action = random.randint(0, 2)
            if action == 0:
                self.pat.a_y = -3

            if action == 2:
                self.pat.a_y = 3