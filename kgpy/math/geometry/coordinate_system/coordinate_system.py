
import copy
import typing as tp
import numpy as np
import astropy.units as u

from kgpy.math import geometry

__all__ = ['CoordinateSystem']


class CoordinateSystem:
    """
    a coordinate system is described by a 3D translation and a 3D rotation from some global coordinate system.
    """

    def __init__(self,
                 translation=geometry.Vector([0, 0, 0, ] * u.m),
                 rotation=geometry.Vector([0, 0, 0, ] * u.deg),
                 translation_first=True):

        # Save input arguments to class variables
        self._translation = translation
        self._rotation = rotation
        self._translation_first = translation_first

        self._x_hat = geometry.Vector([1, 0, 0] * u.dimensionless_unscaled)
        self._y_hat = geometry.Vector([0, 1, 0] * u.dimensionless_unscaled)
        self._z_hat = geometry.Vector([0, 0, 1] * u.dimensionless_unscaled)

        is_inverse = not translation_first
        self._x_hat = self._x_hat.rotate(self.rotation, is_inverse)
        self._y_hat = self._y_hat.rotate(self.rotation, is_inverse)
        self._z_hat = self._z_hat.rotate(self.rotation, is_inverse)

    @classmethod
    def from_polar_coordinates(cls,
                               r: u.Quantity,
                               phi: u.Quantity,
                               origin: 'CoordinateSystem'
                               ) -> 'CoordinateSystem':

        translation = r * origin.x_hat
        rotation = geometry.quaternion.from_zyx_intrinsic_tait_bryan_angles(0, 0, phi.to(u.rad).value)
        cs = CoordinateSystem(translation, rotation, translation_first=False)
        cs = origin @ cs

        return cs

    @property
    def translation(self) -> geometry.Vector:
        return self._translation

    @property
    def translation_global(self):
        return self._translation.rotate(self.rotation, not self.translation_first)

    @property
    def rotation(self) -> geometry.Quaternion:
        return self._rotation

    @property
    def translation_first(self) -> bool:
        return self._translation_first

    @property
    def x_hat(self) -> geometry.Vector:
        return self._x_hat

    @property
    def y_hat(self) -> geometry.Vector:
        return self._y_hat

    @property
    def z_hat(self) -> geometry.Vector:
        return self._z_hat

    def __eq__(self, other: 'CoordinateSystem') -> bool:
        """
        Check if two coordinate systems are equal by checking if the translation vectors are equal and the rotation
        quaternions are equal.
        :param other: Another coordinate system to compare
        :return: True if the two coordinate systems are the same, False if not.
        """

        # Check if the translation and rotation is the same in the other coordinate system
        a = self.translation == other.translation
        b = self.rotation == other.rotation
        c = self.translation_first == other.translation_first

        # return True if the translation and rotation is the same for both coordinate systems
        return a and b and c

    def __add__(self, other: geometry.Vector) -> 'CoordinateSystem':
        """
        Add a vector to the current coordinate system.
        This operation translates the origin of the coordinate system by the Vector other.
        :param other: Vector to translate the coordinate system by
        :return: New coordinate system after translation
        """

        translation = self.translation.__add__(other)

        return type(self)(translation, self.rotation, self.translation_first)

    # Define the reverse addition operator so that both Vector + CoordinateSystem and CoordinateSystem + Vector are
    # allowed.
    __radd__ = __add__

    def __sub__(self, other: geometry.Vector) -> 'CoordinateSystem':

        return self.__add__(other.__neg__())

    def __mul__(self, other: geometry.Vector) -> 'CoordinateSystem':
        """
        Multiply the coordinate system by the quaternion other.
        This operation rotates the coordinate system by the quaternion other.
        :param other: Quaternion to rotate the coordinate system by
        :return: The rotated coordinate system
        """

        rotation = geometry.Vector(self.rotation.components + other.components)
        return type(self)(self.translation, rotation, self.translation_first)

    # Note that the reverse multiplication operation is not defined here since quaternion products do not commute

    def __matmul__(self, other: 'CoordinateSystem') -> 'CoordinateSystem':
        """
        Compute the composition of two coordinate systems.
        This function can be interpreted as translating/rotating other by self.
        :param other: The other coordinate system to be transformed
        :return: a new coordinate system representing the composition of self and other.
        """


        self_translation = self.translation

        if not self.translation_first:
            self_translation = self_translation.rotate(self.rotation, inverse=True)


        other_translation = other.translation.rotate(self.rotation)

        if not other.translation_first:
            other_translation = other_translation.rotate(other.rotation, inverse=True)

        translation = self_translation + other_translation
        rotation = self.rotation + other.rotation

        return type(self)(translation, rotation)

    def __str__(self) -> str:
        """
        :return: Human-readable string representation of this coordinate system
        """
        return 'CoordinateSystem(' + self.translation.__str__() + ', ' + self.rotation.__str__() + ')'

    __repr__ = __str__

    @property
    def inverse_translation_order_explicit(self) -> 'CoordinateSystem':

        return type(self)(self.translation, self.rotation, not self.translation_first)

    @property
    def inverse_translation_order_implicit(self) -> 'CoordinateSystem':

        translation = -self.translation.rotate(self.rotation)
        rotation = self.rotation.inverse()

        return type(self)(translation, rotation, self.translation_first)

    @property
    def inverse(self) -> 'CoordinateSystem':

        translation = -self.translation
        rotation = -self.rotation

        return type(self)(translation, rotation, not self.translation_first)

    def diff(self, other: 'CoordinateSystem') -> 'CoordinateSystem':
        """
        Calculate the difference between this coordinate system and another coordinate system.
        In other words: what coordinate system would need to be composed with other to result in this coordinate system.
        :param other: The other coordinate system to remove from this coordinate system
        :return: New coordinate system representing the difference between the two coordinate systems.
        """

        return other.inverse_translation_order_implicit.__matmul__(self)

    def isclose(self, other: 'CoordinateSystem') -> bool:
        """
        Check if CoordinateSystem other is nearly equal to self.
        This function checks if both the translation vector and the rotation quaternion in both systems are equal.
        :param other: Another CoordinateSystem to compare to
        :return: True if the CoordinateSystems are nearly equal, false otherwise
        """

        # Compare the vectors
        a = self.translation.isclose(other.translation)

        # Compare the quaternions
        b = np.isclose(self.rotation, other.rotation).all()

        return a and b

    def xy_intercept(self, v1: geometry.Vector, v2: geometry.Vector) -> tp.Union[geometry.Vector, None]:
        """
        Determine the point where the line formed by x1 and x2 intercepts the x-y plane of this coordinate system.
        This function is based off of the code provided in this answer: https://stackoverflow.com/a/18543221
        :param v1: Start-point of the line
        :param v2: End-point of the lien
        :return: Vector pointing from the origin of the global coordinate system to the point of intersection, or None
        if an intersection cannot be found.
        """

        # Change parameter names to be the same as the stackoverflow answer cited above
        unit = v1.components.unit
        epsilon = 1e-20 * unit
        p0 = v1
        p1 = v2
        p_co = self.translation_global

        p_no = self.z_hat

        # Compute vector from start to end point
        u = p1 - p0

        #
        dot = p_no.dot(u)

        if abs(dot) > epsilon:

            w = p0 - p_co
            fac = -p_no.dot(w) / dot

            # the factor of the point between p0 -> p1 (0 - 1)
            # if 'fac' is between (0 - 1) the point intersects with the segment.
            # otherwise:
            #  < 0.0: behind p0.
            #  > 1.0: infront of p1.
            if (fac >= 0.0) and (fac <= 1.0):

                u = fac * u

                return p0 + u

        return None




