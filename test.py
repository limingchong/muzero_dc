from tkinter import *


def round_rectangle2(x1, y1, x2, y2, r=25, **kwargs):
    points = (
    x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y1 + r, x2, y2 - r, x2, y2 - r, x2, y2,
    x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1)
    return canvas.create_polygon(points, **kwargs, smooth=True)


def round_rectangle(x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1 + radius, y1,
              x1 + radius, y1,
              x2 - radius, y1,
              x2 - radius, y1,
              x2, y1,
              x2, y1 + radius,
              x2, y1 + radius,
              x2, y2 - radius,
              x2, y2 - radius,
              x2, y2,
              x2 - radius, y2,
              x2 - radius, y2,
              x1 + radius, y2,
              x1 + radius, y2,
              x1, y2,
              x1, y2 - radius,
              x1, y2 - radius,
              x1, y1 + radius,
              x1, y1 + radius,
              x1, y1]

    return canvas.create_polygon(points, **kwargs, smooth=True)

def enter(e):
    print(e, e.widget)


root = Tk()
canvas = Canvas(root)
canvas.pack()

my_rectangle = round_rectangle(50, 50, 150, 100, radius=20, fill="blue")
rect_2 = round_rectangle2(100, 100, 200, 200, r=20, fill='pink')

root.bind_all("<Enter>", func=enter)
root.bind_all("<Button>", func=enter)

root.mainloop()
