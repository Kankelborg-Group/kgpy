
import astropy.units as u

__all__ = ['SurfaceType', 'Standard', 'Paraxial', 'Toroidal', 'DiffractionGrating']


class SurfaceType:

    pass


class Standard(SurfaceType):

    def __init__(self):
        super().__init__()


class Paraxial(SurfaceType):

    def __init__(self):
        super().__init__()


class Toroidal(SurfaceType):

    def __init__(self, radius_of_rotation: u.Quantity = 1 * u.mm):

        super().__init__()

        self.radius_of_rotation = radius_of_rotation


class DiffractionGrating(SurfaceType):

    def __init__(self, diffraction_order: int = 1, groove_frequency: u.Quantity = 1 / u.mm):

        super().__init__()

        self.diffraction_order = diffraction_order
        self.groove_frequency = groove_frequency
