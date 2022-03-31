MAX_SPEED = 20


def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    if x == 0: return 0


class Item:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.a_x = 0
        self.a_y = 0
        self.v_x = 0
        self.v_y = 0
        self.f = 1
        self.size = size

    def accelerate(self):
        if self.x > 590 or 10 > self.x:
            self.v_x = -self.v_x

        if self.y > 590 or 10 > self.y:
            self.v_y = -self.v_y

        if self.v_x > MAX_SPEED:
            self.v_x = MAX_SPEED
        if self.v_y > MAX_SPEED:
            self.v_y = MAX_SPEED
        if self.v_x < - MAX_SPEED:
            self.v_x = - MAX_SPEED
        if self.v_y < - MAX_SPEED:
            self.v_y = - MAX_SPEED

        self.x += self.v_x
        self.y += self.v_y

        self.v_x += self.a_x
        self.v_x += - sign(self.v_x) * self.f
        self.v_y += self.a_y
        self.v_y += - sign(self.v_y) * self.f

    def rend(self, gui):
        pass
