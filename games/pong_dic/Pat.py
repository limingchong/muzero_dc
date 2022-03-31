import numpy as np
from games.pong_dic.Item import Item


class Pat(Item):
    def __init__(self, x, y, size):
        Item.__init__(self, x, y, size)
        self.obj = None

    def rend(self, gui):
        self.accelerate()
        obj_center = (self.x, self.y)
        self.obj = gui.canvas.create_rectangle(
            obj_center[0] - self.size / 3, obj_center[1] - 3 * self.size,
            obj_center[0] + self.size / 3, obj_center[1] + 3 * self.size,
            fill='white')
        gui.canvas.allObject.append(self.obj)
        return self.obj
