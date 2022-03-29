from tkinter import *


class GUI(Canvas):
    def __init__(self, root, a=22, b=22, unit=40, size=15):
        self = Canvas(root, bg="black", height=a * unit, width=b * unit)
        self.place(x=100, y=100)