from tkinter import *
from games.gomoku_dic.Piece import Piece
import numpy


class GUI:
    unit = 40
    height = 11
    width = 11
    size = 15

    def __init__(self, r, a=3, b=3, c=40, d=15):
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

    def render(self, states):
        self.root.update()
        for child in self.canvas.allObject:
            self.canvas.delete(child)
        self.canvas.allObject.clear()
        # origin = np.array([self.height, self.width])

        for x in range(0, self.width * self.unit, self.unit):
            x0, y0, x1, y1 = x, 0, x, self.height * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for y in range(0, self.height * self.unit, self.unit):
            x0, y0, x1, y1 = 0, y, self.width * self.unit, y
            self.canvas.create_line(x0, y0, x1, y1)

        for i in range(self.height):
            for j in range(self.width):
                piece = states[i][j]
                piece.rend(self, i, j)

        self.canvas.pack()