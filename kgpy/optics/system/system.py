
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Dict, Union, Tuple
import quaternion as q
import astropy.units as u
from beautifultable import BeautifulTable
from shapely.geometry import Polygon, MultiPoint, MultiPolygon
from shapely.ops import unary_union
import pickle
from ezdxf.r12writer import r12writer

from kgpy.math import CoordinateSystem, Vector
from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy import optics
from kgpy.optics import Surface, Component
from kgpy.optics.system import wavelength, field
from kgpy.optics.surface.aperture import MultiPolygon as MultiPolygonAper

__all__ = ['System']


class System:
    """
    The System class simulates an entire optical system, and is represented as a series of Components.

    This class is intended to be a drop-in replacement for a Zemax system.
    """

    object_str = 'Object'
    stop_str = 'Stop'
    image_str = 'Image'
    main_str = 'Main'

    def __init__(self, name: str, comment: str = ''):
        """
        Define an optical system by providing a name.

        :param name: Human-readable name of the system
        :param comment: Additional information about the system.
        """

        # Save input arguments to class variables
        self.name = name
        self.comment = comment

        # Initialize attributes to be set as surfaces are added.
        self._surfaces = []     # type: List[Surface]
        
        self._config = 0
        self._num_configurations = 1

        # Create the object surface.
        obj = Surface(self.object_str, thickness=np.inf * u.mm)

        # Create stop surface.
        stop = Surface(self.stop_str)

        # Flag the surface to be the stop
        stop.is_stop = True

        # Create image surface
        image = Surface(self.image_str)

        # Add the three surfaces to the system
        self.append(obj)
        self.append(stop)
        self.append(image)
        
        self._entrance_pupil_radius = 0 * u.mm
        
        self._wavelengths = wavelength.Array()
        self._fields = field.Array()
        
    def append_configuration(self):
        
        self._num_configurations += 1
        
        for surface in self:

            prev_acs = deepcopy(surface.after_surf_cs_break)
            prev_bcs = deepcopy(surface.before_surf_cs_break)

            surface._after_surf_cs_break_list.append(prev_acs)
            surface._before_surf_cs_break_list.append(prev_bcs)
        
    @property
    def config(self) -> int:
        return self._config
    
    @config.setter
    def config(self, value: int):
        
        if value < 0:
            raise ValueError
        
        if value > self.num_configurations - 1:
            raise ValueError
        
        self._config = value

        self.object.reset_cs()
        
    @property
    def num_configurations(self):
        return self._num_configurations
        
    @property
    def fields(self) -> field.Array:
        return self._fields
    
    @fields.setter
    def fields(self, value: field.Array):
        self._fields = value
    
    @property
    def wavelengths(self) -> wavelength.Array:
        return self._wavelengths
    
    @wavelengths.setter
    def wavelengths(self, value: wavelength.Array):
        self._wavelengths = value

    @property
    def entrance_pupil_radius(self) -> u.Quantity:
        return self._entrance_pupil_radius

    @entrance_pupil_radius.setter
    def entrance_pupil_radius(self, value: u.Quantity):
        self._entrance_pupil_radius = value

    @property
    def surfaces(self) -> List[Surface]:
        """
        :return: The private list of surfaces
        """
        return self._surfaces

    @property
    def object(self) -> Surface:
        """
        :return: The object surface within the system, defined as the first surface in the list of surfaces.
        """
        return self[0]

    @property
    def image(self) -> Surface:
        """
        :return: The image surface within the system
        """
        return self[-1]

    @property
    def stop(self) -> Surface:
        """
        :return: The stop surface within the system
        """

        # Return the first surface specified as the stop surface
        try:
            return next(s for s in self if s.is_stop)

        # If there is no surface specified as the stop surface, select the first non-object surface and set it to the
        # stop surface
        except StopIteration:
            self.surfaces[1].is_stop = True

            # Recursive call to find new stop surface
            return self.stop

    @stop.setter
    def stop(self, surf: Surface):
        """
        Set the stop surface to the surface provided
        :param surf: Surface to set as the new stop
        :return: None
        """

        # Loop through all the surfaces in the system and find the provided surface
        for s in self:

            # Check if this is the provided surface
            if s is surf:

                # There can only be one surface, so make sure to update the previous stop surface
                self.stop.is_stop = False

                # Update the new stop surface
                surf.is_stop = True

                # Return once we found the stop surface so we can use the end of the loop as a control statement.
                return

        # If the loop exits the provided surface is not part of the system and this function call doesn't make sense.
        raise ValueError('Cannot set stop to surface not in system')

    @property
    def components(self) -> Dict[str, Component]:
        """
        :return: a Dictionary with all the Components in the system as values and their names as the keys.
        """

        # Allocate space to store the new dictionary
        comp = {}

        # Loop through all the surfaces in the system
        for surf in self._surfaces:

            # If this surface is associated with a component
            if surf.component is not None:

                # Add this surface's component to the dictionary if it's not already there.
                if surf.component.name not in comp:
                    comp[surf.component.name] = surf.component

        return comp

    def insert(self, surface: Surface, index: int) -> None:
        """
        Insert a surface into the specified position index the system
        :param surface: Surface object to be added to the system
        :param index: Index that we want the object to be placed at
        :return: None
        """

        # Make sure that the index is positive
        index = index % len(self)

        # Set the system pointer
        surface.sys = self

        # Set the link to the surface before the new surface
        if index > 0:
            self[index - 1].next_surf_in_system = surface
            surface.prev_surf_in_system = self[index - 1]

        # Set the link to the surface after the new surface
        if index < len(self):
            self[index].prev_surf_in_system = surface
            surface.next_surf_in_system = self[index]

        # Add the surface to the list of surfaces
        self._surfaces.insert(index, surface)

        # Reset the coordinate systems, to revaluate with the new surface
        surface.reset_cs()

    def append(self, surface: Surface) -> None:
        """
        Add a surface to the end of an optical system.
        :param surface: The surface to be added.
        :return: None
        """

        # Update link from surface to system
        surface.sys = self

        # Link to the previous surface in the system
        if len(self) > 0:
            self[-1].next_surf_in_system = surface
            surface.prev_surf_in_system = self[-1]

        # Append surface to the list of surfaces
        self._surfaces.append(surface)

    def insert_component(self, component: Component, index: int) -> None:
        """
        Add the component and all its surfaces to the end of an optical system.
        :param component: component to be added to the system.
        :return: None
        """

        index = index % len(self)

        # Link the system to the component
        component.sys = self

        # Loop through the surfaces in the component add them to the back of the system
        for surf in component:
            self.insert(surf, index)
            index = index + 1

    def add_baffle(self, baffle_name: str, baffle_cs: CoordinateSystem,
                   pass_surfaces: Union[None, List[Union[None, Surface]]] = None, margin: u.Quantity = 1 * u.mm
                   ) -> Component:

        comp = self.add_baffle_component(baffle_name, baffle_cs)

        self.calc_baffle_aperture(comp, pass_surfaces=pass_surfaces, margin=margin)

        return comp

    def calc_baffle_aperture(self, component: Component, pass_surfaces: Union[None, List[Union[None, Surface]]] = None,
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

                            v1 = Vector(np.append(p1.value, 0) * p1.unit)
                            v2 = Vector(np.append(p2.value, 0) * p2.unit)

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

            surface.mechanical_aperture = MultiPolygonAper(aper_lst)

        # Write dxf file
        with r12writer(component.name + '.dxf') as dxf:

            for poly in aper:

                pts = np.stack(poly.exterior.xy).transpose()

                pts[:, 1] = pts[:, 1] + (3.49927 + 3.35597) / 2

                dxf.add_polyline(pts)
                # dxf.add_solid(pts)


    def add_baffle_component(self, baffle_name: str, baffle_cs: CoordinateSystem) -> Component:
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
        baffle = Component(baffle_name)

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
                intercept = baffle_cs.xy_intercept(surf.front_cs.X, surf.back_cs.X)
    
                # If the intercept exists, insert the new baffle
                if intercept is not None:

                    # Compute the new thickness vectors for both to
                    t1 = intercept - surf.front_cs.X  # New thickness of original surface
                    t2 = surf.back_cs.X - intercept  # Thickness of new surface to be added after the baffle
    
                    # Modify the original surface to have the correct thickness
                    surf.thickness = t1.dot(surf.front_cs.zh)
    
                    # Calculate the tilt/decenter required to put the baffle in the correct place
                    cs = baffle_cs.diff(surf.back_cs)
                    cs.X.z = 0 * u.mm

                    # Create new baffle surface
                    baffle_thickness = t2.dot(surf.front_cs.zh)
                    baffle_surf = optics.ZmxSurface(baffle_name + str(baffle_pass), thickness=baffle_thickness)
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
                    
                    surf.reset_cs()
    
                    # Update the number of baffle passes
                    baffle_pass += 1
                    
                    break
            
            if intercept is None:
                break

        return baffle

    def square_raytrace(self, configurations: List[int], surfaces: List[Surface], wavelengths: List[wavelength.Item],
                        num_field, num_pupil):

        r = 1.0

        field_x = np.linspace(-r, r, num=num_field)
        field_y = np.linspace(-r, r, num=num_field)

        # Evenly spaced grid of pupil positions
        pupil_x = np.linspace(-r, r, num=num_pupil)
        pupil_y = np.linspace(-r, r, num=num_pupil)

        return self.raytrace(configurations, surfaces, wavelengths, field_x, field_y, pupil_x, pupil_y)
            
    def raytrace(self, configurations: List[int], surfaces: List[Surface], wavelengths: List[wavelength.Item],
                 field_x: np.ndarray, field_y: np.ndarray, pupil_x: np.ndarray, pupil_y: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        raise NotImplementedError



    @property
    def _surfaces_dict(self) -> Dict[str, Surface]:
        """
        :return: a dictionary where the key is the surface name and the value is the surface.
        """

        # Allocate space for result
        d = {}

        # Loop through surfaces and add to dict
        for surf in self:
            d[surf.name] = surf

        return d

    # @property
    # def _all_surfaces(self) -> List[Surface]:
    #     """
    #     :return: a list of all surfaces in the object, including the object and image surfaces
    #     """
    #     return [self.object] + self._surfaces + [self.image]

    def __getitem__(self, item: Union[int, str]) -> Surface:
        """
        Gets the surface at index item within the system, or the surface with the name item
        Accessed using the square bracket operator, e.g. surf = sys[i]
        :param item: Surface index or name of surface
        :return: Surface specified by item
        """

        # If the item is an integer, use it to access the surface list
        if isinstance(item, int):
            return self._surfaces.__getitem__(item)

        # If the item is a string, use it to access the surfaces dictionary.
        elif isinstance(item, str):
            return self._surfaces_dict.__getitem__(item)

        # Otherwise, the item is neither an int nor string and we throw an error.
        else:
            raise ValueError('Item is of an unrecognized type')

    def __delitem__(self, key: int):

        self[key].sys = None

        self._surfaces.__delitem__(key)

    def __iter__(self):

        return self._surfaces.__iter__()

    def __len__(self):

        return self._surfaces.__len__()

    def __str__(self) -> str:
        """
        :return: String representation of a system
        """

        # Create output table
        table = BeautifulTable(max_width=200)

        # Append lines for each surface within the component
        for surface in self._surfaces:

            # Add headers if not already populated
            if not table.column_headers:
                table.column_headers = surface.table_headers

            # Append surface to table
            table.append_row(surface.table_row)

        # Set column alignments
        table.column_alignments['Component'] = BeautifulTable.ALIGN_LEFT
        table.column_alignments['Surface'] = BeautifulTable.ALIGN_LEFT
        table.column_alignments['Thickness'] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments['X_x'] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments['X_y'] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments['X_z'] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments['T_x'] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments['T_y'] = BeautifulTable.ALIGN_RIGHT
        table.column_alignments['T_z'] = BeautifulTable.ALIGN_RIGHT

        # Don't automatically format numeric strings
        table.detect_numerics = False

        return table.__str__()
