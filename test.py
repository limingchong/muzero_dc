from tkinter import *

def button_press(e):
    print(e)

gui = Tk()
gui.bind_all("<Button>", button_press)

gui.mainloop()


