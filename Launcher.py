import os
import tkinter

import img
from pathlib import Path
from tkinter import *
from games.gomoku import gomoku
from games.tank_battle import tank_battle

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
        self.geometry('{}x{}+{}+{}'.format(WINDOW_WIDTH,
                                           WINDOW_HEIGHT,
                                           int(self.winfo_screenwidth() / 2 - WINDOW_WIDTH / 2),
                                           int(self.winfo_screenheight() / 2 - WINDOW_HEIGHT / 2)))
        self.title("基于强化学习的攻防系统")
        self.setObjects()
        self.mainloop()

    def setObjects(self):
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

        self.connect4 =     Button(self.games_frame, image=image_connect4, width=200, height=200)
        self.gomoku =       Button(self.games_frame, image=image_gomoku, width=200, height=200)
        self.pong =         Button(self.games_frame, image=image_pong, width=200, height=200)
        self.tank_wars =    Button(self.games_frame, image=image_tank_wars, width=200, height=200)
        self.tic_tac_toe =  Button(self.games_frame, image=image_tic_tac_toe, width=200, height=200)
        self.twenty_one =   Button(self.games_frame, image=image_twenty_one, width=200, height=200)

        self.bind_all("<Button-1>", self.click)
        self.bind_all("<Button-3>", self.click)

        self.window_title.grid(row=0, column=1)
        self.connect4.grid(row=1, column=0)
        self.gomoku.grid(row=1, column=1)
        self.pong.grid(row=1, column=2)
        self.tank_wars.grid(row=2, column=0)
        self.tic_tac_toe.grid(row=2, column=1)
        self.twenty_one.grid(row=2, column=2)

        self.games_frame.pack(padx=20, pady=20)

        options = ["训练", "测试", "修改", "详情"]
        self.rightList = Listbox(self)
        for opt in options:
            self.rightList.insert(END, opt)

    def clear_all(self):
        for widget in self.games_frame.winfo_children():
            widget.destroy()

    def click(self, evt):
        print("evt:", evt)
        print("widget", evt.widget)

        if str(evt.widget) == ".!frame.!button4":
            game = tank_battle(self)
            game.test()

        if str(evt.widget) == ".!frame.!button2":
            game = gomoku(self)
            game.test()

        if evt.num == 3:
            self.rightList.place(x=evt.x, y=evt.y)
        else:
            self.rightList.place_forget()

    def press(self):
        print("command:")


if __name__ == "__main__":
    window()
