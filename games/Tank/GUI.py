import threading
import time
import tkinter as tk
import numpy as np
from games.Tank.Item import Item
from tkinter import *


class GUI(threading.Thread):
    unit = 40
    height = 22
    width = 22
    size = 15

    def __init__(self, r, states, a=22, b=22, c=40, d=15, delay=0.01):
        threading.Thread.__init__(self)
        threading.Thread()
        self.delay = delay
        self.canvas = Canvas(r, bg="white",
                             height=a * c,
                             width=b * c)

        self.root = r
        self.states = states
        self.height = a
        self.width = b
        self.unit = c
        self.size = d
        self.canvas.allObject = list()

        r.geometry('{0}x{1}'.format(self.height * self.unit, self.height * self.unit))
        self.start()

    def run(self):
        while True:
            self.render(self.states)
            time.sleep(self.delay)

    def render(self, states=(0 * np.arange(22 * 22).reshape(22, 22))):
        self.root.update()
        for child in self.canvas.allObject:
            self.canvas.delete(child)
        self.canvas.allObject.clear()

        # origin = np.array([self.height, self.width])

        item = Item(0, 0, 0)

        for i in range(22):
            for j in range(22):
                item = states[i][j]
                item.rend(self)

        self.canvas.pack()
