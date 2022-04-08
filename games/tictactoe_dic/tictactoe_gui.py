"""
gomoku.py
"""
import datetime
import math
import pathlib
import random

import numpy
import torch

import games.tictactoe_dic.Empty as Empty
import games.tictactoe_dic.GUI as GUI
import games.tictactoe_dic.Piece as Piece

class tictactoe_gui:
    def __init__(self, root):
        self.testing = True
        self.root = root
        self.name = "tictactoe"
        self.play = False

    def train(self):
        pass

    def test(self):
        self.root.clear_all()
        self.states = []
        self.board = 0 * numpy.arange(3 * 3).reshape(3, 3)

        for i in range(3):
            row = []
            for j in range(3):
                row.append(Empty.Empty(0, ""))
            self.states.append(row)

        self.canvas = GUI.GUI(self.root, self.states)

        self.root.games_frame.unbind_all("<Button>")
        self.root.bind_all("<Button>", self.button_press)
        self.root.bind_all("<Key>", self.key_press)

        while self.testing:
            self.canvas.render(self.states)
            if not self.play:
                x = random.randint(0, 2)
                y = random.randint(0, 2)
                while type(self.states[x][y]) is not Empty.Empty:
                    x = random.randint(0, 2)
                    y = random.randint(0, 2)
                self.new_piece(x, y, 'white')
                self.play = True

        self.root.clear_all()
        self.root.setObjects()

    def button_press(self, e):
        x = int(3 * e.x / self.canvas.width / self.canvas.unit)
        y = int(3 * e.y / self.canvas.height / self.canvas.unit)
        if self.play and type(self.states[x][y]) is Empty.Empty:
            self.new_piece(x, y, 'black')
            self.play = False

    def new_piece(self, x, y, color):
        self.states[x][y] = Piece.Piece(15, color)
        if self.judge_all(x, y):
            print(color, "win.")

    def judge_all(self, x0, y0):
        for (i, j) in ((1, 0), (0, 1), (1, -1), (1, 1)):
            if self.judge(x0, y0, i, j):
                return True

        return False

    def judge(self, x0, y0, dx, dy):
        total = 0
        for i in range(1, 3):
            if x0 + dx * i > 2 or y0 + dy * i > 2 or x0 + dx * i < 0 or y0 + dy * i < 0 or \
                    self.states[x0 + dx * i][y0 + dy * i].color != self.states[x0][y0].color:
                break
            total += 1
        for i in range(1, 3 - total):
            if x0 - dx * i > 2 or y0 - dy * i > 2 or x0 - dx * i < 0 or y0 - dy * i < 0 or \
                    self.states[x0 - dx * i][y0 - dy * i].color != self.states[x0][y0].color:
                break
            total += 1

        return total > 1

    def key_press(self, e):
        if e.keysym == "Escape":
            self.testing = False

    def get_observation(self):
        board = numpy.zeros((3, 3), dtype="int32")
        for i in range(0,3):
            for j in range(0,3):
                if type(self.states[i][j]) == Piece:
                    if self.states[i][j].color == 'black':
                        board[i][j] = -1
                    else:
                        board[i][j] = 1

        board_player1 = numpy.where(board == 1, 1, 0)               # ai have put = 1 others 0
        board_player2 = numpy.where(board == -1, 1, 0)              # human have put = 1 others 0
        board_to_play = numpy.full((3, 3), 1 if self.play else -1)  # if this is human = 1 or -1
        return numpy.array([board_player1, board_player2, board_to_play], dtype="int32")

