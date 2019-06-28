class Spider(Aperture):

    def __init__(self, arm_width: u.Quantity = 0 * u.mm, num_arms: int = 0):
        Aperture.__init__(self)

        self.num_arms = num_arms
        self.arm_width = arm_width

    def promote_to_zmx(self, surf: 'optics.ZmxSurface', attr_str: str
                       ) -> 'optics.zemax.surface.aperture.Spider':
        a = kgpy.optics.zemax.system.configuration.surface.aperture.Spider(self.arm_width, self.num_arms, surf,
                                                                           attr_str)

        a.decenter_x = self.decenter_x
        a.decenter_y = self.decenter_y

        return a
