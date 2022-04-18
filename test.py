import pathlib
import pickle
import tkinter
from tkinter import *
import torch
import numpy
from matplotlib import pyplot
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


WINDOW_WIDTH = 760
WINDOW_HEIGHT = 600


root = Tk()
root.geometry('{}x{}+{}+{}'.format(
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    int(root.winfo_screenwidth() / 2 - WINDOW_WIDTH / 2),
    int(root.winfo_screenheight() / 2 - WINDOW_HEIGHT / 2)))
root.title("基于强化学习的攻防系统")
root.config(bg="#FFFFFF")

Label(root, text="Tic Tac Toe", font=("Times New Roman", "25"), bg="#FFFFFF").place(x=40,y=20)

Canvas(root, bg="#D8D8D8", height=5, width=700).place(x=30, y=70)

Canvas(root, bg="#D8D8D8", height=470, width=123).place(x=30, y=100)

Canvas(root, bg="#D8D8D8", height=470, width=552).place(x=180, y=100)

Label(root,
      text="游戏名",
      font=("microsoft yahei", "12"),
      wraplength=450,
      bg="#D8D8D8",
      justify="left").place(x=200, y=120)

Entry(root,
      textvariable="abc",
      width=15,
      borderwidth=0).place(x=270, y=125)

global image_tic_tac_toe
image_tic_tac_toe = PhotoImage(file="img/tic_tac_toe.png")
img = image_tic_tac_toe.subsample(int(image_tic_tac_toe.width() / 100))
Label(root, image=img, width=100, height=100, bd=2,relief="solid").place(x=40,y=110)

root.mainloop()