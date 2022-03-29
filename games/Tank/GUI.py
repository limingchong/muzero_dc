import tkinter as tk
import numpy as np
from games.Tank.Item import Item
from tkinter import *


class GUI():
    unit = 40
    height = 22
    width = 22
    size = 15

    def __init__(self, r, a=22, b=22, c=40, d=15):
        self.canvas = Canvas(r, bg="white",
                         height=a * c,
                         width=b * c)

        self.root = r
        self.height = a
        self.width = b
        self.unit = c
        self.size = d
        self.canvas.allObject = list()

        r.geometry('{0}x{1}'.format(self.height * self.unit, self.height * self.unit))

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
