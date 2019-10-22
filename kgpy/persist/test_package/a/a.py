from . import X, Y


class A:

    def __init__(self):

        self.x = X()
        self.y = Y()

    def f(self):

        self.x.f()
        self.y.f()
