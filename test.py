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

Canvas(root, bg="#D8D8D8", height=470, width=700).place(x=30, y=100)

global image_tic_tac_toe
image_tic_tac_toe = PhotoImage(file="img/tic_tac_toe.png")
img = image_tic_tac_toe.subsample(int(image_tic_tac_toe.width() / 200))
Label(root, image=img, width=200, height=200, bd=2,relief="solid").place(x=50,y=120)

Label(root,
      text="简介",
      font=("microsoft yahei", "15", "bold"), wraplength=450,bg="#D8D8D8",justify="left").place(x=265,y=113)

Label(root,
      text="是黑白棋的一种。三子棋是一种民间传统游戏，又叫九宫棋、圈圈叉叉、一条龙、井字棋等。将正方形对角线连起来，相对两边依次摆上"
           "三个双方棋子。",
      font=("microsoft yahei", "12"), wraplength=450,bg="#D8D8D8",justify="left").place(x=265,y=140)

Label(root,
      text="规则",
      font=("microsoft yahei", "15", "bold"), wraplength=450,bg="#D8D8D8",justify="left").place(x=265,y=210)

Label(root,
      text="只要将自己的三个棋子走成一条线，对方就算输了。如果两个人都掌握了技巧，那么一般来说就是平棋。一般来说，第二步下在中间最有"
           "利（因为第一步不能够下在中间），下在角上次之，下在边上再次之。",
      font=("microsoft yahei", "12"), wraplength=450,bg="#D8D8D8",justify="left").place(x=265,y=237)


checkpoint_path = "results/tictactoe/model.checkpoint"
checkpoint_path = pathlib.Path(checkpoint_path)
checkpoint = torch.load(checkpoint_path)


replay_buffer_path = "results/tictactoe/replay_buffer.pkl"

replay_buffer_path = pathlib.Path(replay_buffer_path)
with open(replay_buffer_path, "rb") as f:
    replay_buffer_infos = pickle.load(f)
    replay_buffer = replay_buffer_infos["buffer"]

# game_priority
x = []
y = []
keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]

for key in keys:
    print(key, checkpoint[key])


total_win = 0
horizontal_line = []
for i in range(checkpoint["num_played_games"]):
    x.append(i)
    steps = len(replay_buffer[i].reward_history)
    win = 1 if replay_buffer[i].to_play_history[steps - 1] == 1 else -1
    total_win += win
    y.append(total_win / (i + 1))
    horizontal_line.append(0.5)


str = "训练局数：" + str(checkpoint["num_played_games"]) + "\n" \
      "平均步数：" + str(checkpoint["num_played_steps"] / checkpoint["num_played_games"]) + "\n" \
      "平均损失：" + str(checkpoint["reward_loss"])


Label(root, text=str, font=("microsoft yahei", "12", "bold"), wraplength=450,bg="#D8D8D8",justify="left").place(x=40,y=350)
Button(root, text="重新训练", bg="#D8D8D8", borderwidth=2, fg="black", font=("microsoft yahei", "12", "bold")).place(x=40, y=450)
Button(root, text="关于我们", bg="#D8D8D8", borderwidth=2, fg="black", font=("microsoft yahei", "12", "bold")).place(x=40, y=500)

f = pyplot.figure()

pyplot.plot(x, y, ls='-', lw=1, label='win rate', color='purple')
pyplot.plot(x, horizontal_line, ls='-', lw=2, label='50%', color='red')
pyplot.legend()
pyplot.xlabel("epoch")
pyplot.ylabel("rate")

plot_show = FigureCanvasTkAgg(f, root)
plot_show.get_tk_widget().pack(side=tkinter.BOTTOM,expand=True)
plot_show.get_tk_widget().place(x=400,y=350)
plot_show.get_tk_widget().config(width=300, height=200)

root.mainloop()