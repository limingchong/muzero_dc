from tkinter import *
import numpy
from games.pong_dic.Item import Item

class GUI:
    unit = 40
    height = 11
    width = 11
    size = 15

    def __init__(self, r, states, a=15, b=15, c=40, d=15):
        self.canvas = Canvas(r, bg="black",
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

    def render(self, states=(0 * numpy.arange(11 * 11).reshape(11, 11))):
        self.root.update()
        for child in self.canvas.allObject:
            self.canvas.delete(child)
        self.canvas.allObject.clear()

        # origin = np.array([self.height, self.width])

        for i in range(30):
            self.canvas.create_rectangle(295, i * 30, 305, 20 + i * 30, fill='white')

        item = Item(0, 0, 0)

        for i in range(3):
            states[i].rend(self)

        self.canvas.pack()