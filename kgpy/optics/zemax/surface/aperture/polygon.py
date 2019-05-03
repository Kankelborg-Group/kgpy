
from os.path import join
from typing import List
from uuid import uuid4
import numpy as np
import astropy.units as u

from kgpy import optics
from kgpy.optics.surface import aperture
from .aperture import Aperture
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceApertureTypes, ISurfaceApertureUser

__all__ = ['Polygon', 'MultiPolygon']


class MultiPolygon(Aperture, aperture.MultiPolygon):

    def __init__(self, polygons: List[u.Quantity], surf: 'optics.ZmxSurface',
                 attr_str: str):

        self.filename = str(uuid4()) + '.uda'

        Aperture.__init__(self, surf, attr_str)

        aperture.MultiPolygon.__init__(self, polygons)

    @property
    def aperture_type(self) -> SurfaceApertureTypes:

        if self.is_obscuration:
            return SurfaceApertureTypes.UserObscuration

        else:
            return SurfaceApertureTypes.UserAperture

    @property
    def settings(self) -> ISurfaceApertureUser:

        s = Aperture.settings.fget(self)
        
        if self.is_obscuration:
            # noinspection PyProtectedMember
            return s._S_UserObscuration

        else:
            # noinspection PyProtectedMember
            return s._S_UserAperture

    @settings.setter
    def settings(self, val: ISurfaceApertureUser) -> None:
        Aperture.settings.fset(self, val)

    @property
    def points(self) -> u.Quantity:
        p = []
        
        units = self.polygons[0].unit

        for points in self.polygons:

            p.append(points.value)

        # todo: fix for non-constant units
        print(p)
        return np.concatenate(p) * units

    @property
    def polygons(self) -> List[u.Quantity]:

        return self._polygons

    @polygons.setter
    def polygons(self, val: List[u.Quantity]) -> None:

        self._polygons = val

        filepath = join(self.surf.sys.object_dir, 'Apertures', self.filename)

        self._write_uda_file(filepath, val)

        s = self.settings
        s.ApertureFile = self.filename
        self.settings = s
    
    def _write_uda_file(self, uda_file: str, polygons: List[u.Quantity]):
        
        # Open the file
        with open(uda_file, 'w') as uda:
            
            for poly in polygons:
            
                for point in poly:
                    
                    point = point.to(self.surf.sys.lens_units).value
                    
                    line = 'LIN ' + str(point[0]) + ' ' + str(point[1]) + '\n'
    
                    uda.write(line)
    
                uda.write('BRK\n')
    
    def _read_uda_file(self, uda_file: str) -> np.ndarray:
        """
        Interpret a Zemax user-defined aperture (UDA) file as a polygon.

        :param uda_file: Location of the uda file to read
        :return: A polygon representing the aperture.
        """

        # Open the file
        with open(uda_file, encoding='utf-16') as uda:

            # Allocate space for storing the list of points in the file
            polygons = []
            poly = []

            # Loop through every line in the file
            for line in uda:

                # Remove the newlines from the end of the line
                line = line.strip()

                # Split each string at spaces
                params = line.split(' ')

                # Remove any empty elements (multiple spaces between arguments)
                params = list(filter(None, params))

                # If this is a line-type entry
                if params[0] == 'LIN':

                    # If the line has three total arguments
                    if len(params) == 3:

                        # Append the x,y coordinates to the list of points
                        poly.append((float(params[1]), float(params[2])))

                    # Otherwise, the line has the incorrect number of arguments
                    else:
                        raise ValueError('Incorrect number of parameters for line')

                elif params[0] == 'BRK':
                    polygons.append(poly * self.surf.sys.lens_units)
                    poly = []

                else:
                    raise ValueError('Unrecognized user aperture command. Not all commands have been implemented yet.')

            # Construct new polygon from the list of points
            aper = np.array(poly)

            return aper


class Polygon(MultiPolygon):
    
    def __init__(self, points: u.Quantity, surf: 'optics.ZmxSurface', attr_str: str):
        
        self.points = points
        
        MultiPolygon.__init__(self, [points], surf, attr_str)

    @property
    def points(self) -> u.Quantity:
        return self._points

    @points.setter
    def points(self, value: u.Quantity):
        self._points = value
