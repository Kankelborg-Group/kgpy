__all__ = ['Material']


class Material:

    def __str__(self):
        return ''


class Mirror(Material):

    def __str__(self):
        return 'MIRROR'
