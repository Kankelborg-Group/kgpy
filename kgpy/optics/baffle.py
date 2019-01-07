
from .component import Component

__all__ = ['Baffle']


class Baffle(Component):
    """
    Baffles are sheets of metal with cutouts for rays to pass through.
    These optical elements are calculated after the system has been optimized.
    """

    def __init__(self, zmx, ind, global_z):
        """
        Constructor for Baffle
        :param zmx: Pointer to zemax object
        :param ind: Unique index of this baffle
        :param global_z: Coordinate of this baffle along the optic axis
        :type zmx: kgpy.optics.Zemax
        :type ind: int
        :type global_z: float
        """

        # Save input arguments
        self.zmx = zmx
        self.ind = ind
        self.z = global_z

        # Calculate the indices of surfaces
        self.bad_surf_inds = self.calc_surface_intersections()

        # Calculate the number of sequential passes through the optical system
        self.passes = len(self.bad_surf_inds)


    @property
    def name(self):
        """
        :return: Human-readable name of this baffle
        :rtype: str
        """
        return 'Baffel ' + str(self.ind)

    @property
    def surface_comment_dict(self):
        """
        Automatically generate comments based off how many passes required to construct each baffle.
        :return: Dictionary of pass names and the comment on each pass.
        :rtype: dict[str, str]
        """

        # Initialize return dictionary
        d = {}

        # Loop through each ray pass and construct comment
        for p in range(self.passes):
            str = 'pass ' + str(p)
            d[str] = self.name + ', ' + str

        return d



    def insert_into_zmx(self):

        # Grab pointer to lens data editor
        lde = self.zmx.TheSystem.LDE

        for surf_ind in self.bad_surf_inds:

            surf = lde.GetSurfaceAt(surf_ind)

    def calc_surfaces_to_split(self):

        pass

# class TestBaffle(TestComponent):
#
#     def setUp(self):
#         super().setUp()
#
#         self.component = Baffle(self.zmx, ind=1, global_z=50.0)
#
#     def test__init__(self):
#
#         self.assertTrue(self.component.passes == 1)
