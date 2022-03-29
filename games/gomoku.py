'''
gomoku.py
'''
import games.gomoku_dic.GUI as GUI

class gomoku():
    def __init__(self, root):
        self.root = root
        self.name = "gomoku"

    def train(self):
        pass

    def test(self):
        GUI.GUI(self.root)



