from games.Tank.Item import Item


class Empty(Item):
    def __init__(self, x, y, size):
        Item.__init__(self, x, y, size)
        self.obj = None

    def rend(self, gui):
        return self.obj
