class Polygon(Aperture):

    def __init__(self, points: u.Quantity):
        Aperture.__init__(self)

        self.points = points

    def promote_to_zmx(self, surf: 'optics.ZmxSurface', attr_str: str
                       ) -> 'optics.zemax.surface.aperture.Polygon':
        a = kgpy.optics.zemax.system.configuration.surface.aperture.Polygon(self.points, surf, attr_str)

        a.is_obscuration = self.is_obscuration
        a.decenter_x = self.decenter_x
        a.decenter_y = self.decenter_y

        return a


class MultiPolygon(Aperture):

    def __init__(self, polygons: List[u.Quantity]):
        Aperture.__init__(self)

        self.polygons = polygons

    def promote_to_zmx(self, surf: 'optics.ZmxSurface', attr_str: str
                       ) -> 'optics.zemax.surface.aperture.MultiPolygon':
        a = kgpy.optics.zemax.system.configuration.surface.aperture.MultiPolygon(self.polygons, surf, attr_str)

        a.is_obscuration = self.is_obscuration
        a.decenter_x = self.decenter_x
        a.decenter_y = self.decenter_y

        return a


class RegularPolygon(Polygon):

    def __init__(self, radius: u.Quantity, num_sides: int):
        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False) * u.rad  # type: u.Quantity

        # Calculate points
        x = radius * np.cos(angles)  # type: u.Quantity
        y = radius * np.sin(angles)  # type: u.Quantity
        pts = np.array([x, y])
        pts = pts.transpose() * x.unit

        Polygon.__init__(self, pts)


class Octagon(RegularPolygon):

    def __init__(self, radius: u.Quantity):
        RegularPolygon.__init__(self, radius, 8)