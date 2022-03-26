import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

from Tank.Item import Item


class GUI(tk.Tk, object):
    unit = 40
    height = 22
    width = 22
    size = 15

    def __init__(self, a=22, b=22, c=40, d=15):
        super(GUI, self).__init__()
        self.height = a
        self.width = b
        self.unit = c
        self.size = d
        self.allObject = list()
        self.canvas = tk.Canvas(self, bg="white",
                                height=a * c,
                                width=b * c)
        self.action_space = ["up", "down", "left", "right"]
        self.n_actions = len(self.action_space)
        self.title("Battle")
        self.geometry('{0}x{1}'.format(self.height * self.unit, self.height * self.unit))

    def render(self, states=(0 * np.arange(22 * 22).reshape(22, 22))):
        self.update()
        for child in self.allObject:
            self.canvas.delete(child)
        self.allObject.clear()

        # origin = np.array([self.height, self.width])

        item = Item(0, 0, 0)

        for i in range(22):
            for j in range(22):
                item = states[i][j]
                item.rend(self)

        self.canvas.pack()
