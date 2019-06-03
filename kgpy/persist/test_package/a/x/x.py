
from kgpy import Persist
from . import J, K


class X(Persist):

    def __init__(self):

        self.j = J()
        self.k = K()

    def f(self):

        self.j.f()
        self.k.f()
