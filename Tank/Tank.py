import random
import numpy as np

from Tank.DeadWall import DeadWall
from Tank.Empty import Empty
from Tank.Item import Item


class Tank(Item):
    x = 0
    y = 0
    orientation = 0
    team = 0
    states = []

    def __init__(self, x, y, size, me):
        Item.__init__(self, x, y, size)
        self.obj = None
        self.x = x
        self.y = y
        self.team = 0 if me < 7 else 1
        self.orientation = me - 4 * self.team
        self.size = size

    def rend(self, gui):
        origin = np.array([gui.height, gui.width])
        obj_center = origin + np.array([gui.unit * self.x, gui.unit * self.y])

        if self.team == 0:
            color = 'blue'
        else:
            color = 'red'

        if self.orientation == 3:
            arr = [-1, -1, 1, -1, 0, 1]
        elif self.orientation == 4:
            arr = [-1, -1, -1, 1, 1, 0]
        elif self.orientation == 5:
            arr = [-1, 1, 1, 1, 0, -1]
        else:
            arr = [1, -1, 1, 1, -1, 0]

        self.obj = gui.canvas.create_polygon(
            obj_center[0] + arr[0] * self.size, obj_center[1] + arr[1] * self.size,
            obj_center[0] + arr[2] * self.size, obj_center[1] + arr[3] * self.size,
            obj_center[0] + arr[4] * self.size, obj_center[1] + arr[5] * self.size,
            fill=color)
        gui.allObject.append(self.obj)
        return self.obj

    def random_act(self, states):
        action = 3
        none = np.zeros((22,22))
        while action == 3:
            action = random.randint(0, 3)
            if action == 0:
                self.shoot(states, none)

            if action == 1:
                self.rotate(1, none)

            if action == 2:
                self.rotate(-1, none)

            if action == 3:
                if self.forwardTest(states):
                    action = 0
                    self.forward(states, none)

        return action

    def forward(self, states, board):
        dx = 0
        dy = 0
        if self.orientation == 3:
            dy = 1

        if self.orientation == 4:
            dx = 1

        if self.orientation == 5:
            dy = -1

        if self.orientation == 6:
            dx = -1

        states[self.x][self.y] = Empty(self.x, self.y, 15)
        board[self.x][self.y] = 0
        self.x += dx
        self.y += dy
        states[self.x][self.y] = self
        board[self.x][self.y] = self.orientation + self.team * 4

    def forwardTest(self, states):
        dx = 0
        dy = 0
        if self.orientation == 3:
            dy = 1

        if self.orientation == 4:
            dx = 1

        if self.orientation == 5:
            dy = -1

        if self.orientation == 6:
            dx = -1

        return type(states[self.x + dx][self.y + dy]) == Empty

    def rotate(self, do, board):
        self.orientation += do
        if self.orientation == 2:
            self.orientation = 6

        if self.orientation == 7:
            self.orientation = 3

        board[self.x, self.y] = self.orientation + self.team * 4

    def shoot(self, states, board):
        dx = 0
        dy = 0
        if self.orientation == 3:
            dy = 1

        if self.orientation == 4:
            dx = 1

        if self.orientation == 5:
            dy = -1

        if self.orientation == 6:
            dx = -1

        x = self.x + dx
        y = self.y + dy

        while type(states[x][y]) == Empty:
            x += dx
            y += dy

        if type(states[x][y]) != DeadWall:
            states[x][y] = Empty(self.x, self.y, 15)
            board[x][y] = 0