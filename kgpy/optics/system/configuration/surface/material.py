
__all__ = ['Material']


class Material:

    def __init__(self):

        self.name = ''


class Mirror(Material):
    
    def __init__(self):

        super().__init__()

        self.name = 'mirror'


class EmptySpace(Material):

    def __init__(self):

        super().__init__()

        self.name = 'empty space'
