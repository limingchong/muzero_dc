import numpy as np

from Tank.Item import Item


class DeadWall(Item):
    def __init__(self, x, y, size):
        Item.__init__(self, x, y, size)
        self.obj = None

    def rend(self, gui):
        origin = np.array([gui.height, gui.width])
        obj_center = origin + np.array([gui.unit * self.x, gui.unit * self.y])
        self.obj = gui.canvas.create_rectangle(
            obj_center[0] - self.size, obj_center[1] - self.size,
            obj_center[0] + self.size, obj_center[1] + self.size,
            fill='black')
        gui.allObject.append(self.obj)
        return self.obj
