import numpy

class Piece:
    def __init__(self, size, color):
        self.size = size
        self.color = color

    def rend(self, gui, i, j):
        origin = numpy.array([gui.height, gui.width])
        obj_center = origin + numpy.array([gui.unit * i + 7, gui.unit * j + 7])
        self.obj = gui.canvas.create_oval(
            obj_center[0] - self.size, obj_center[1] - self.size,
            obj_center[0] + self.size, obj_center[1] + self.size,
            fill=self.color)
        gui.canvas.allObject.append(self.obj)
        return self.obj
