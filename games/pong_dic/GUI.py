from tkinter import *
import numpy
from games.pong_dic.Item import Item

class GUI:
    unit = 40
    height = 11
    width = 11
    size = 15

    def __init__(self, r, states, width=600, height=600):
        self.canvas = Canvas(r, bg="black",
                             height=height,
                             width=width)

        self.root = r
        self.states = states
        self.height = width
        self.width = height
        self.canvas.allObject = list()

        r.geometry('{0}x{1}'.format(self.width, self.height))

    def render(self, states):
        self.root.update()
        for child in self.canvas.allObject:
            self.canvas.delete(child)
        self.canvas.allObject.clear()

        for i in range(30):
            self.canvas.create_rectangle(295, i * 30, 305, 20 + i * 30, fill='white')

        for i in range(3):
            states[i].rend(self)

        self.canvas.pack()