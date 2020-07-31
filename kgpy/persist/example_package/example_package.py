from . import A, B


class ExamplePackage:

    def __init__(self):

        self.a = A()
        self.b = B()

    def f(self):

        self.a.f()
        self.b.f()
