class Circular(Aperture):

    def __init__(self, min_radius: u.Quantity = 0 * u.mm, max_radius: u.Quantity = 0 * u.mm):
        Aperture.__init__(self)

        self.min_radius = min_radius
        self.max_radius = max_radius

    def promote_to_zmx(self, surf: 'optics.ZmxSurface', attr_str: str
                       ) -> 'optics.zemax.surface.aperture.Circular':
        a = kgpy.optics.zemax.system.configuration.surface.aperture.Circular(self.min_radius, self.max_radius, surf,
                                                                             attr_str)

        a.is_obscuration = self.is_obscuration
        a.decenter_x = self.decenter_x
        a.decenter_y = self.decenter_y

        return a