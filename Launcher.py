import tkinter
from tkinter import *

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
                                           int(self.winfo_screenwidth()/2-WINDOW_WIDTH/2),
                                           int(self.winfo_screenheight()/2-WINDOW_HEIGHT/2)))
        self.title("基于强化学习的攻防系统")
        self.connect = Button(self, image="/muzero_dc/img/connect4.png", text="Next", width=10, pady=20, command=self.press)

        self.mainloop()

    def press(self):
        pass


if __name__ == "__main__":
    window()