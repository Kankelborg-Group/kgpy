
from os.path import join
from uuid import uuid4
import numpy as np
import astropy.units as u

from kgpy import optics
from kgpy.optics.surface import aperture
from .aperture import Aperture
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import SurfaceApertureTypes, ISurfaceApertureUser

__all__ = ['Polygon']


class Polygon(Aperture, aperture.Polygon):

    def __init__(self, points: u.Quantity, surf: 'optics.ZmxSurface'):

        self.filename = str(uuid4()) + '.uda'

        Aperture.__init__(self, surf)

        aperture.Polygon.__init__(self, points)

    @property
    def aperture_type(self) -> SurfaceApertureTypes:

        if self.is_obscuration:
            return SurfaceApertureTypes.UserObscuration

        else:
            return SurfaceApertureTypes.UserAperture

    @property
    def settings(self) -> ISurfaceApertureUser:

        s = self.surf.main_row.ApertureData.CurrentTypeSettings

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

        return self._points

    @points.setter
    def points(self, val: u.Quantity) -> None:

        self._points = val

        filepath = join(self.surf.sys.object_dir, 'Apertures', self.filename)

        self._write_uda_file(filepath, val.to(self.surf.sys.lens_units).value)

        s = self.settings
        s.ApertureFile = self.filename
        self.settings = s
    
    @staticmethod
    def _write_uda_file(uda_file: str, points: np.ndarray):
        
        # Open the file
        with open(uda_file, 'w') as uda:
            
            for point in points:
                
                line = 'LIN ' + str(point[0]) + ' ' + str(point[1]) + '\n'

                uda.write(line)

            uda.write('BRK\n')
    
    @staticmethod
    def _read_uda_file(uda_file: str) -> np.ndarray:
        """
        Interpret a Zemax user-defined aperture (UDA) file as a polygon.

        :param uda_file: Location of the uda file to read
        :return: A polygon representing the aperture.
        """

        # Open the file
        with open(uda_file, encoding='utf-16') as uda:

            # Allocate space for storing the list of points in the file
            pts = []

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
                        pts.append((float(params[1]), float(params[2])))

                    # Otherwise, the line has the incorrect number of arguments
                    else:
                        raise ValueError('Incorrect number of parameters for line')

                elif params[0] == 'BRK':
                    break

                else:
                    raise ValueError('Unrecognized user aperture command. Not all commands have been implemented yet.')

            # Construct new polygon from the list of points
            aper = np.array(pts)

            return aper
