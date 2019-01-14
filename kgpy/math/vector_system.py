
__all__ = ['VectorSystem']


class VectorSystem:
    """
    A VectorSystem is an object comprised of a vector and a coordinate system.
    It represents a vector in a particular coordinate system
    """

    def __init__(self, X, cs):
        """
        Construct ray from coordinate system and vector
        :param X: Vector defined in the coordinate system cs
        :param cs: Coordinate system of the Ray
        :type X: kgpy.math.Vector
        :type cs: kgpy.math.CoordinateSystem
        """

        # Save input arguments as class variables
        self.X = X
        self.cs = cs
