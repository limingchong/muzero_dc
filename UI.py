import json
import pathlib
import tkinter
from tkinter import *
from tkinter import ttk

from muzero import MuZero, load_model_menu, hyperparameter_search
import copy
import importlib
import json
import math
import pathlib
import pickle
import sys
import time

import nevergrad
import numpy
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

import diagnose_model
import models
import replay_buffer
import self_play
import shared_storage
import trainer

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500

def window_1(games):
    global root
    root = Tk()
    root.title("muzero")
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    width = 600
    height = 500
    x = int((screenwidth - width) / 2)
    y = int((screenheight - height) / 2)
    label = Label(root, text=" ", height=5, width=8, font='Arial 20 bold')
    label.grid(column=1, row=4)
    root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    L1 = Label(root, text="Welcome to MuZero!\nHere's a list of games:", height=5, font='Arial 20 bold')
    L1.grid(column=2, row=1)
    global combobox
    combobox = ttk.Combobox(
        master=root,  # 父容器
        height=10,  # 高度,下拉显示的条目数量
        width=20,  # 宽度
        state='readonly',  # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
        cursor='arrow',  # 鼠标移动时样式 arrow, circle, cross, plus...
        font=('', 20),  # 字体
        values=games,  # 设置下拉框的选项
    )

    combobox.grid(column=2, row=2)
    image_gomoku = tkinter.PhotoImage(file="img/gomoku.png")
    B = Button(root, text="Next", width=10, pady=20, command=chooseGame)
    B.grid(column=2, row=5)
    root.mainloop()


def chooseGame():
    global game_name
    global muzero
    game_name = combobox.get()
    muzero = MuZero(game_name)
    root.destroy()
    window_2(game_name)

def window_2(game):
    print(game)
    global root1
    root1 = Tk()
    root1.title("muzero")
    screenwidth = root1.winfo_screenwidth()  # 屏幕宽度
    screenheight = root1.winfo_screenheight()  # 屏幕高度
    width = 600
    height = 500
    x = int((screenwidth - width) / 2)
    y = int((screenheight - height) / 2)
    label = Label(root1, text=" ",height=5, width=8,font='Arial 20 bold')
    label.grid(column=1, row=4)
    root1.geometry('{}x{}+{}+{}'.format(width, height, x, y))  # 大小以及位置
    L1 = Label(root1, text="Choose an action:", height=5, font='Arial 20 bold')
    L1.grid(column=2, row=1)
    #L1.pack()
    global combobox1
    # Configure running options
    options = [
        "Train",
        "Load pretrained model",
        "Diagnose model",
        "Render some self play games",
        "Play against MuZero",
        "Test the game manually",
        "Hyperparameter search",
        "Exit",
    ]
    combobox1 = ttk.Combobox(
        master=root1,  # 父容器
        height=10,  # 高度,下拉显示的条目数量
        width=20,  # 宽度
        state='readonly',  # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
        cursor='arrow',  # 鼠标移动时样式 arrow, circle, cross, plus...
        font=('', 20),  # 字体
        values=options,  # 设置下拉框的选项
    )

    combobox1.grid(column=2, row=2)
    #combobox.pack()

    B1 = Button(root1, text="Next", width=10, pady=20, command=chooseGame1)
    B1.grid(column=2, row=5)
    root.mainloop()


def chooseGame1():
    pass


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Train directly with: python muzero.py cartpole
        muzero = MuZero(sys.argv[1])
        muzero.train()
    elif len(sys.argv) == 3:
        # Train directly with: python muzero.py cartpole '{"lr_init": 0.01}'
        config = json.loads(sys.argv[2])
        muzero = MuZero(sys.argv[1], config)
        muzero.train()
    else:
        # Let user pick a game
        games = [
            filename.stem
            for filename in sorted(list((pathlib.Path.cwd() / "games").glob("*.py")))
            if filename.name != "abstract_game.py"
        ]
        print(games)
        window_1(games)

        while False:
            # Configure running options
            options = [
                "Train",
                "Load pretrained model",
                "Diagnose model",
                "Render some self play games",
                "Play against MuZero",
                "Test the game manually",
                "Hyperparameter search",
                "Exit",
            ]
            window_2(options)
            choice = int(choice)
            if choice == 0:
                muzero.train()
            elif choice == 1:
                load_model_menu(muzero, game_name)
            elif choice == 2:
                muzero.diagnose_model(30)
            elif choice == 3:
                muzero.test(render=True, opponent="self", muzero_player=None)
            elif choice == 4:
                muzero.test(render=True, opponent="human", muzero_player=0)
            elif choice == 5:
                env = muzero.Game()
                env.reset()
                env.render()

                done = False
                while not done:
                    action = env.human_to_action()
                    observation, reward, done = env.step(action)
                    print(f"\nAction: {env.action_to_string(action)}\nReward: {reward}")
                    env.render()
            elif choice == 6:
                # Define here the parameters to tune
                # Parametrization documentation: https://facebookresearch.github.io/nevergrad/parametrization.html
                muzero.terminate_workers()
                del muzero
                budget = 20
                parallel_experiments = 2
                lr_init = nevergrad.p.Log(lower=0.0001, upper=0.1)
                discount = nevergrad.p.Log(lower=0.95, upper=0.9999)
                parametrization = nevergrad.p.Dict(lr_init=lr_init, discount=discount)
                best_hyperparameters = hyperparameter_search(
                    game_name, parametrization, budget, parallel_experiments, 20
                )
                muzero = MuZero(game_name, best_hyperparameters)
            else:
                break
            print("\nDone")

    ray.shutdown()
