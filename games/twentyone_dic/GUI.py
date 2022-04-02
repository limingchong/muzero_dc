from tkinter import *
from games.gomoku_dic.Piece import Piece
import numpy


class GUI:
    unit = 40
    height = 11
    width = 11
    size = 15

    def __init__(self, r, states, width=600, height=300):
        self.canvas = Canvas(r, bg="white",
                             height=height,
                             width=width)

        self.root = r
        self.states = states
        self.height = height
        self.width = width
        self.canvas.allObject = list()

        r.geometry('{0}x{1}'.format(self.width, self.height))

    def render(self, states):
        self.root.update()
        for child in self.canvas.allObject:
            self.canvas.delete(child)
        self.canvas.allObject.clear()

        for obj in states:
            obj.rend(self)

        self.canvas.pack()
