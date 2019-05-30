

from . import A, B


class TestPackage:

    def __init__(self):

        self.a = A()
        self.b = B()

    def f(self):

        self.a.f()
        self.b.f()
