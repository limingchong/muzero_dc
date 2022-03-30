"""
gomoku.py
"""

import random

import numpy

import games.gomoku_dic.GUI as GUI
import games.gomoku_dic.Piece as Piece
import games.gomoku_dic.Empty as Empty

class gomoku:
    def __init__(self, root):
        self.testing = True
        self.root = root
        self.name = "gomoku"
        self.play = False
        root.clear_all()

    def train(self):
        pass

    def test(self):
        self.states = []
        self.board = 0 * numpy.arange(11 * 11).reshape(11, 11)

        for i in range(11):
            row = []
            for j in range(11):
                row.append(Empty.Empty(0, ""))
            self.states.append(row)

        self.canvas = GUI.GUI(self.root, self.states)

        self.root.games_frame.unbind_all("<Button>")
        self.root.bind_all("<Button>", self.button_press)
        self.root.bind_all("<Key>", self.key_press)

        while self.testing:
            self.canvas.render(self.states)
            if not self.play:
                x = random.randint(0, 10)
                y = random.randint(0, 10)
                while type(self.states[x][y]) is not Empty.Empty:
                    x = random.randint(0, 10)
                    y = random.randint(0, 10)
                self.new_piece(x, y, 'white')
                self.play = True

        self.root.clear_all()
        self.root.setObjects()

    def button_press(self, e):
        x = int(11 * e.x / self.canvas.width / self.canvas.unit)
        y = int(11 * e.y / self.canvas.height / self.canvas.unit)
        if self.play and type(self.states[x][y]) is Empty.Empty:
            self.new_piece(x, y, 'black')
            self.play = False

    def new_piece(self, x, y, color):
        self.states[x][y] = Piece.Piece(10, color)
        print(color, self.judge_all(x, y))

    def judge_all(self, x0, y0):
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i != 0 or j != 0:
                    if self.judge(x0, y0, i, j):
                        return True

        return False

    def judge(self, x0, y0, dx, dy):
        for i in range(1, 5):
            if x0 + dx * i > 10 or y0 + dy * i > 10 or x0 + dx * i < 0 or y0 + dy * i < 0:
                return False
            if self.states[x0 + dx * i][y0 + dy * i].color != self.states[x0][y0].color:
                return False

        return True

    def key_press(self, e):
        if e.keysym == "Escape":
            self.testing = False