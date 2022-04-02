import os
import tkinter

import img
from pathlib import Path
from tkinter import *
from games.gomoku import gomoku
from games.pong import pong
from games.tank_battle import tank_battle
from games.twentyone import twentyone

'''
    Author: Mingchong Li
    Date: 2022/3/27
'''

# parameters

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540


class window(Tk):
    button1 = None

    def __init__(self):
        Tk.__init__(self)
        self.games_frame = None
        self.game = None
        self.setObjects()
        self.mainloop()

    def setObjects(self):
        self.geometry('{}x{}+{}+{}'.format(WINDOW_WIDTH,
                                           WINDOW_HEIGHT,
                                           int(self.winfo_screenwidth() / 2 - WINDOW_WIDTH / 2),
                                           int(self.winfo_screenheight() / 2 - WINDOW_HEIGHT / 2)))
        self.title("基于强化学习的攻防系统")

        self.games_frame = Frame(self)

        self.window_title = Label(self.games_frame, text="基于强化学习的攻防系统", font="黑体 25")

        global image_connect4
        global image_gomoku
        global image_pong
        global image_tank_wars
        global image_tic_tac_toe
        global image_twenty_one

        image_connect4 = tkinter.PhotoImage(file="img/connect4.png")
        image_gomoku = tkinter.PhotoImage(file="img/gomoku.png")
        image_pong = tkinter.PhotoImage(file="img/pong.png")
        image_tank_wars = tkinter.PhotoImage(file="img/tank_wars.png")
        image_tic_tac_toe = tkinter.PhotoImage(file="img/tic_tac_toe.png")
        image_twenty_one = tkinter.PhotoImage(file="img/twenty_one.png")

        image_connect4 = image_connect4.subsample(int(image_connect4.width()/200))
        image_gomoku = image_gomoku.subsample(int(image_gomoku.width()/200))
        image_pong = image_pong.subsample(int(image_pong.width()/200))
        image_tank_wars = image_tank_wars.subsample(int(image_tank_wars.width()/200))
        image_tic_tac_toe = image_tic_tac_toe.subsample(int(image_tic_tac_toe.width()/200))
        image_twenty_one = image_twenty_one.subsample(int(image_twenty_one.width()/200))

        self.connect4 = Button(self.games_frame, image=image_connect4, width=200, height=200)
        self.gomoku = Button(self.games_frame, image=image_gomoku, width=200, height=200)
        self.pong = Button(self.games_frame, image=image_pong, width=200, height=200)
        self.tank_wars = Button(self.games_frame, image=image_tank_wars, width=200, height=200)
        self.tic_tac_toe = Button(self.games_frame, image=image_tic_tac_toe, width=200, height=200)
        self.twenty_one = Button(self.games_frame, image=image_twenty_one, width=200, height=200)

        self.games_frame.bind_all("<Button>", self.click)

        self.window_title.grid(row=0, column=1)
        self.connect4.grid(row=1, column=0)
        self.gomoku.grid(row=1, column=1)
        self.pong.grid(row=1, column=2)
        self.tank_wars.grid(row=2, column=0)
        self.tic_tac_toe.grid(row=2, column=1)
        self.twenty_one.grid(row=2, column=2)

        self.games_frame.pack(padx=10, pady=10)

        options = ["训练", "测试", "修改", "详情"]
        self.rightList = Listbox(self, font="黑体 16", height=4, width=4, borderwidth=0, bg="green")
        for opt in options:
            self.rightList.insert(END, opt)

        self.update()

    def clear_all(self):
        for widget in self.winfo_children():
            widget.destroy()

    def click(self, e):
        print("e:", e, "widget:", e.widget, "root:", [e.x_root, e.y_root])

        if str(e.widget)[-7: len(str(e.widget))] == "button":
            pass
        elif str(e.widget)[-7: len(str(e.widget))] == "button2":
            self.game = gomoku(self)
        elif str(e.widget)[-7: len(str(e.widget))] == "button3":
            self.game = pong(self)
        elif str(e.widget)[-7: len(str(e.widget))] == "button4":
            self.game = tank_battle(self)
        elif str(e.widget)[-7: len(str(e.widget))] == "button6":
            self.game = twentyone(self)



        if e.num == 1:
            if self.game is None:
                self.rightList.place_forget()
            elif str(e.widget) == ".!listbox":
                if self.rightList.curselection()[0] == 0:
                    print("test", self.game.name)
                if self.rightList.curselection()[0] == 1:
                    print("train", self.game.name)
                if self.rightList.curselection()[0] == 2:
                    print("modify", self.game.name)
                if self.rightList.curselection()[0] == 3:
                    print("detail", self.game.name)
            else:
                print("test", self.game.name)
                self.clear_all()
                self.game.test()

            self.game = None

        if e.num == 3:
            self.rightList.place(x=e.x_root - self.winfo_x() - 10, y=e.y_root - self.winfo_y() - 25)


if __name__ == "__main__":
    window()
