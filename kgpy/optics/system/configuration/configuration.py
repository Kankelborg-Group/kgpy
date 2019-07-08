
import typing as tp
import numpy as np
import astropy.units as u
from beautifultable import BeautifulTable

from kgpy import optics
from . import Wavelength, Field, Surface
from . import surface

__all__ = ['Configuration']


class Configuration:

    object_str = 'Object'
    stop_str = 'Stop'
    image_str = 'Image'
    main_str = 'Main'

    def __init__(self,
                 name: str = '',
                 entrance_pupil_radius: u.Quantity = None,
                 surfaces: tp.List[Surface] = None,
                 wavelengths: tp.List[Wavelength] = None,
                 fields: tp.List[Field] = None
                 ):

        if entrance_pupil_radius is None:
            entrance_pupil_radius = 0 * u.m
        if surfaces is None:
            surfaces = []
        if wavelengths is None:
            wavelengths = []
        if fields is None:
            fields = []

        self._name = name
        self._entrance_pupil_radius = entrance_pupil_radius
        self._surfaces = surfaces
        self._wavelengths = wavelengths
        self._fields = fields

        self._object = self.surfaces[0]
        self._image = self.surfaces[-1]

        stop_surfaces = [s for s in self.surfaces if s.is_stop]
        if len(stop_surfaces) != 1:
            raise ValueError
        self._stop = stop_surfaces[0]

    @property
    def name(self) -> str:
        return self._name

    @property
    def entrance_pupil_radius(self) -> u.Quantity:
        return self._entrance_pupil_radius

    @property
    def surfaces(self) -> tp.List[Surface]:
        return self._surfaces

    @property
    def wavelengths(self) -> tp.List[Wavelength]:
        return self._wavelengths

    @property
    def fields(self) -> tp.List[Field]:
        return self._fields

    @property
    def object(self) -> Surface:
        """
        :return: The object surface within the system, defined as the first surface in the list of surfaces.
        """
        return self._object

    @property
    def image(self) -> Surface:
        """
        :return: The image surface within the system
        """
        return self._image

    @property
    def stop(self) -> Surface:
        """
        :return: The stop surface within the system
        """
        return self._stop

    def __str__(self) -> str:
        """
        :return: String representation of a system
        """

        # Create output table
        table = BeautifulTable(max_width=200)

        # Append lines for each surface within the component
        for surface in self.surfaces:

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
