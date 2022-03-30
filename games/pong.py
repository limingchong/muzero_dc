'''
pong.py
'''
import time

from games.pong_dic.Ball import Ball
from games.pong_dic.GUI import GUI
from games.pong_dic.Pat import Pat

PAUSE_TIME = 0.05

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

        while self.testing:
            self.canvas.render(self.states)

        self.root.clear_all()
        self.root.setObjects()

    def key_press(self, e):
        if time.time() - self.last_press_time > PAUSE_TIME:
            print((self.states[0].x, self.states[0].y))
            if e.keysym == "Up":
                self.states[0].y -= 20

            if e.keysym == "Down":
                self.states[0].y += 20

            if e.keysym == "Escape":
                self.testing = False

            self.last_press_time = time.time()