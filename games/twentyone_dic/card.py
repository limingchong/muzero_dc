import numpy


class card:
    def __init__(self, size, color, x, y, num):
        self.size = size
        self.color = color
        self.x = x
        self.y = y
        self.num = num

    def rend(self, gui):
        obj = gui.canvas.create_rectangle(
            self.x - self.size, self.y - 2 * self.size,
            self.x + self.size, self.y + 2 * self.size,
            fill=self.color)
        txt = gui.canvas.create_text(self.x, self.y, text=self.num)
        gui.canvas.allObject.append(obj)
        gui.canvas.allObject.append(txt)