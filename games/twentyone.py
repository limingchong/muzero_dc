'''
    twentyone.py
'''
import random
import tkinter

from games.twentyone_dic.GUI import GUI
from games.twentyone_dic.card import card


class twentyone:
    def __init__(self, root):
        self.root = root
        self.name = "twentyone"
        root.clear_all()

    def train(self):
        pass

    def test(self):
        self.states = []

        self.canvas = GUI(self.root, self.states, width=600, height=300)
        self.next = tkinter.Button(text="next", command=self.press_next)
        self.stop = tkinter.Button(text="stop", command=self.press_stop)
        self.next.place(x=500, y=200)
        self.stop.place(x=500, y=250)
        self.root.games_frame.unbind_all("<Button>")
        self.root.bind_all("<Key>", self.key_press)

        self.count_me = 0
        self.count_op = 0
        self.testing = True

        while self.testing:
            self.canvas.render(self.states)

        self.root.clear_all()
        self.root.setObjects()

    def press_next(self):
        self.states.append(card(
            20,
            'pink',
            50 + self.count_me * 30,
            200 + pow(self.count_me - 5, 2),
            random.randint(1, 10)))
        self.count_me += 1
        if self.sum(self.states) > 21:
            print("loose")

    def press_stop(self):
        self.AI()
        ai_score = self.sum(self.states[self.count_me:len(self.states)])
        my_score = self.sum(self.states[0:self.count_me])
        if ai_score > 21 or ai_score < my_score:
            print("win")
        elif ai_score == my_score:
            print("equal")
        else:
            print("loose")

    def sum(self, states):
        total = 0
        for card in states:
            total += card.num

        return total

    def AI(self):
        self.states.append(
            card(
                20,
                'white',
                50 + self.count_op * 30,
                50 + pow(self.count_op - 5, 2),
                random.randint(1, 10)))
        self.count_op += 1
        while self.sum(self.states[self.count_me:len(self.states)]) <= 21 and random.randint(0, 2) == 0:
            self.states.append(
                card(
                    20,
                    'white',
                    50 + self.count_op * 30,
                    50 + pow(self.count_op - 5, 2),
                    random.randint(1, 10)))
            self.count_op += 1

    def key_press(self, e):
        if e.keysym == "Escape":
            self.testing = False
