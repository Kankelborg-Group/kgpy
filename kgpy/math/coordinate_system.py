

class CoordinateSystem:
    """
    A coordinate system is described by a 3D translation and a 3D rotation from some global coordinate system.
    """

    def __init__(self, X, Q):
        """
        Defines a new coordinate system using a vector (translation) and a quaternion (rotation)
        :param X: Vector pointing from origin of global coordinate system to the origin of this coordinate system
        :param Q: Quaternion representing the 3D rotation of this coordinate system with respect to the global
        coordinate system.
        :type X: kgpy.math.Vector
        :type Q: kgpy.math.Quaternion
        """

        # Save input arguments to class variables
        self.X = X
        self.Q = Q

