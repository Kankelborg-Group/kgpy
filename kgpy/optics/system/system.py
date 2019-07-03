
import typing as tp
import pickle
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
from shapely.ops import unary_union
from ezdxf.r12writer import r12writer

from kgpy import optics, math

from . import Configuration, configuration

__all__ = ['System']


class System:

    def __init__(self, name: str = '', configurations: tp.List[Configuration] = None):

        if configurations is None:
            configurations = []

        self._name = name
        self._configurations = configurations

    @property
    def name(self) -> str:
        return self._name

    @property
    def configurations(self) -> tp.List[Configuration]:
        return self._configurations







    def add_baffle(self, baffle_name: str, baffle_cs: math.CoordinateSystem,
                   pass_surfaces: tp.Union[None, tp.List[tp.Union[None, configuration.Surface]]] = None,
                   margin: u.Quantity = 1 * u.mm) -> configuration.Component:

        comp = self.add_baffle_component(baffle_name, baffle_cs)

        self.calc_baffle_aperture(comp, pass_surfaces=pass_surfaces, margin=margin)

        return comp

    def calc_baffle_aperture(self, component: configuration.Component, pass_surfaces:
                             tp.Union[None, tp.List[tp.Union[None, optics.system.configuration.Surface]]] = None,
                             margin: u.Quantity = 1 * u.mm):

        if pass_surfaces is None:
            pass_surfaces = len(component) * [None]     # type: List[Union[None, List[Surface]]]

        # n = 15
        # m = 15
        n = 5
        m = 1

        wavl = [self.wavelengths.items[0]]

        configs = list(range(self.num_configurations))

        V, _, _ = self.square_raytrace(configs, [self.image], wavl, n, m * n)

        surf_lst = []

        old_config = self.config

        plt.figure(figsize=(10,10))

        for s, surface in enumerate(component):

            surface = surface   # type: Surface

            ch_lst = []

            for c in configs:

                if pass_surfaces[s] is None:

                    _, X, Y = self.square_raytrace([c], [surface], wavl, n, m * n)

                    x = X[c, ...]
                    y = Y[c, ...]
                    v = V[c, ...]

                    x = x.flatten()
                    y = y.flatten()
                    v = v.flatten()

                    x = x[v == 0]
                    y = y[v == 0]

                    pts = np.stack([x, y]).transpose()

                else:

                    self.config = c

                    s1 = pass_surfaces[s][0]
                    s2 = pass_surfaces[s][1]

                    if s1.aperture is not None:
                        pts1 = s1.aperture.points
                    else:
                        pts1 = s1.mechanical_aperture.points

                    if s2.aperture is not None:
                        pts2 = s2.aperture.points
                    else:
                        pts2 = s2.mechanical_aperture.points

                    pts = []

                    for p1 in pts1:

                        for p2 in pts2:

                            v1 = math.Vector(np.append(p1.value, 0) * p1.unit)
                            v2 = math.Vector(np.append(p2.value, 0) * p2.unit)

                            v1 = s1.cs.X + v1.rotate(s1.cs.Q)
                            v2 = s2.cs.X + v2.rotate(s2.cs.Q)
                            
                            # c1 = gcs() + v1
                            # c2 = gcs() + v2
                            #
                            # c1 = s1.cs @ c1
                            # c2 = s2.cs @ c2

                            v3 = surface.cs.xy_intercept(v1, v2)

                            pts.append(v3.X.value[:2])

                    pts = np.stack(pts)

                    self.config = old_config

                plt.scatter(pts[:, 0], pts[:, 1], s=1)

                pts = MultiPoint(pts)
                hull = pts.convex_hull
                ch_lst.append(hull)

            surf_lst.append(unary_union(ch_lst))

        aper = unary_union(surf_lst)
        if isinstance(aper, Polygon):
            aper = MultiPolygon([aper])

        for poly in aper:
            plt.plot(*poly.exterior.xy)

        t = margin.value    # todo: fix this to convert units correctly
        n = 3
        for d in range(n):
            dilated_aperture_lst = []
            for poly in aper:
                dilated_aperture_lst.append(poly.buffer(t/n))
            aper = MultiPolygon(dilated_aperture_lst)

        aper = unary_union(aper)
        if isinstance(aper, Polygon):
            aper = MultiPolygon([aper])

        for poly in aper:
            plt.plot(*poly.exterior.xy)

        plt.title(component.name)
        plt.gca().set_aspect('equal')
        lim = 1.5 * self.entrance_pupil_radius.value
        plt.xlim((-lim, lim))
        plt.ylim((-lim, lim))
        plt.savefig(component.name + '.png', bbox_inches='tight')
        pickle.dump(plt.gcf(), open(component.name + '.pickle', 'wb'))

        for surface in component:

            aper_lst = []
            for poly in aper:

                aper_lst.append(np.stack(poly.exterior.xy).transpose() * self.lens_units)

            surface.mechanical_aperture = kgpy.optics.system.configuration.surface.aperture.aperture.MultiPolygon(aper_lst)

        # Write dxf file
        with r12writer(component.name + '.dxf') as dxf:

            for poly in aper:

                pts = np.stack(poly.exterior.xy).transpose()

                pts[:, 1] = pts[:, 1] + (3.49927 + 3.35597) / 2

                dxf.add_polyline(pts)
                # dxf.add_solid(pts)

    def add_baffle_component(self, baffle_name: str, baffle_cs: math.CoordinateSystem) -> configuration.Component:
        """
        Add a baffle to the system at the specified coordinate system across the x-y plane.
        This function automatically calculates how many times the raypath crosses the baffle plane, and constructs the
        appropriate amount of baffle surfaces
        :param baffle_name: Human-readable name of the baffle
        :param baffle_cs: Coordinate system where the baffle will be placed.
        This function assumes that the baffle lies in the x-y plane of this coordinate system.
        :return: Pointer to Baffle component
        """

        # Create new component to store the baffle
        baffle = configuration.Component(baffle_name)

        # Define variable to track how many times the system intersected the
        baffle_pass = 0

        index = 0
        
        while True:

            intercept = None

            # Make a copy of the surfaces list so we don't try to iterate over and write to the same list
            old_surfaces = self._surfaces.copy()
    
            # Loop through all surfaces in the system to see if any intersect with a baffle
            for surf in old_surfaces:

                if surf.system_index <= index:
                    continue
    
                # Compute the intersection between the thickness vector and the x-y plane of the baffle, if it exists.
                intercept = baffle_cs.xy_intercept(surf.post_cs.X, surf.back_cs.X)
    
                # If the intercept exists, insert the new baffle
                if intercept is not None:

                    # Compute the new thickness vectors for both to
                    t1 = intercept - surf.post_cs.X  # New thickness of original surface
                    t2 = surf.back_cs.X - intercept  # Thickness of new surface to be added after the baffle
    
                    # Modify the original surface to have the correct thickness
                    surf.thickness = t1.dot(surf.post_cs.z_hat)
    
                    # Calculate the tilt/decenter required to put the baffle in the correct place
                    cs = baffle_cs.diff(surf.back_cs)
                    cs.X.z = 0 * u.mm

                    # Create new baffle surface
                    baffle_thickness = t2.dot(surf.post_cs.z_hat)
                    baffle_surf = optics.Surface(baffle_name + str(baffle_pass), thickness=baffle_thickness)
                    for c in range(self.num_configurations):
                        self.config = c
                        baffle_surf._before_surf_cs_break_list.append(cs)
                        baffle_surf._after_surf_cs_break_list.append(cs.inverse)
                    self.config = 0

                    index = surf.system_index + 1
    
                    # Link the new baffle surface into the system
                    self.insert(baffle_surf, index)
    
                    # Insert new baffle surface into baffle component
                    baffle.append(baffle_surf)
                    
                    surf.update()
    
                    # Update the number of baffle passes
                    baffle_pass += 1
                    
                    break
            
            if intercept is None:
                break

        return baffle

    def square_raytrace(self, configurations: tp.List[int], surfaces: tp.List[configuration.Surface],
                        wavelengths: tp.List[configuration.Wavelength],
                        num_field, num_pupil):

        r = 1.0

        field_x = np.linspace(-r, r, num=num_field)
        field_y = np.linspace(-r, r, num=num_field)

        # Evenly spaced grid of pupil positions
        pupil_x = np.linspace(-r, r, num=num_pupil)
        pupil_y = np.linspace(-r, r, num=num_pupil)

        return self.raytrace(configurations, surfaces, wavelengths, field_x, field_y, pupil_x, pupil_y)
            
    def raytrace(self, configurations: tp.List[int], surfaces: tp.List[configuration.Surface],
                 wavelengths: tp.List[configuration.Wavelength], field_x: np.ndarray, field_y: np.ndarray,
                 pupil_x: np.ndarray, pupil_y: np.ndarray
                 ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        raise NotImplementedError







