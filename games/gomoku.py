"""
gomoku.py
"""
from tkinter import *


import games.gomoku_dic.GUI as GUI


class gomoku():
    def __init__(self, root):
        self.root = root
        self.name = "gomoku"

    def train(self):
        pass

    def test(self):
        self.canvas = GUI.GUI(self.root)
