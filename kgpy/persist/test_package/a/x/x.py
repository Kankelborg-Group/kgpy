
from . import J, K


class X:

    def __init__(self):

        self.j = J()
        self.k = K()

    def f(self):

        self.j.f()
        self.k.f()
