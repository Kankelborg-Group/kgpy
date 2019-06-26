
import typing as tp
import collections
import numpy as np
import astropy.units as u
from beautifultable import BeautifulTable

from kgpy import optics
from . import Wavelength, WavelengthList, Field, FieldList, Surface, Component

__all__ = ['Configuration']


class Configuration(collections.UserList):

    object_str = 'Object'
    stop_str = 'Stop'
    image_str = 'Image'
    main_str = 'Main'

    def __init__(self, surfaces: tp.List[Surface] = None):

        super().__init__(surfaces)

        self.name = ''

        self.system = None

        self._object = None
        self._stop = None
        self._image = None

        self.entrance_pupil_radius = 0 * u.mm

        self.wavelengths = WavelengthList()

        self.fields = FieldList()

    @property
    def system(self) -> tp.Optional['optics.System']:
        return self._system

    @system.setter
    def system(self, value: tp.Optional['optics.System']):
        self._system = value

    @property
    def fields(self) -> FieldList:
        return self._fields

    @fields.setter
    def fields(self, value: FieldList):
        self._fields = value

    @property
    def wavelengths(self) -> WavelengthList:
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value: WavelengthList):
        self._wavelengths = value

    @property
    def entrance_pupil_radius(self) -> u.Quantity:
        return self._entrance_pupil_radius

    @entrance_pupil_radius.setter
    def entrance_pupil_radius(self, value: u.Quantity):
        self._entrance_pupil_radius = value

    @property
    def surfaces(self) -> tp.List[optics.system.configuration.Surface]:
        """
        :return: The private list of surfaces
        """
        return super().data

    @property
    def object(self) -> tp.Optional[Surface]:
        """
        :return: The object surface within the system, defined as the first surface in the list of surfaces.
        """
        return self._object

    @property
    def image(self) -> tp.Optional[Surface]:
        """
        :return: The image surface within the system
        """
        return self._image

    @property
    def stop(self) -> tp.Optional[Surface]:
        """
        :return: The stop surface within the system
        """
        return self._stop

    def insert(self, i: int, item: Surface):

        super().insert(i, item)

        item.configuration = self

        self.update()

    def append(self, item: Surface):

        super().append(item)

        # Update link from surface to system
        item.configuration = self

    def update(self):

        for surface in self:
            surface.update()

    def __getitem__(self, item) -> Surface:
        return super().__getitem__(item)

    def __setitem__(self, key: int, value: Surface):
        super().__setitem__(key, value)

    def __iter__(self) -> tp.Iterator[Surface]:
        return super().__iter__()

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
