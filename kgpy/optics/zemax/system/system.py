
from os.path import dirname, join
from win32com.client.gencache import EnsureDispatch
from win32com.client import constants
from typing import List, Union, Tuple, Dict, Iterator
from collections import OrderedDict
from numbers import Real
import numpy as np
import astropy.units as u
from shapely.geometry import Polygon

from kgpy.math.coordinate_system import GlobalCoordinateSystem as gcs
from kgpy.optics import System, Component, Surface
from kgpy.optics.zemax.surface import ZmxSurface
from kgpy.optics.zemax.system import wavelength, field
from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.ZOSAPI.Editors.LDE import ILDERow
from kgpy.optics.zemax.ZOSAPI.Analysis import AnalysisIDM

__all__ = ['ZmxSystem']


class ZmxSystem(System):
    """
    A class used to interface with a particular Zemax file


    Installation instructions:
    Make sure the Python wrappers are available for the COM client and interfaces
    EnsureModule('ZOSAPI_Interfaces', 0, 1, 0)
    Note - the above can also be accomplished using 'makepy.py' in the
    following directory:
    ::

         `{PythonEnv}/Lib/site-packages/wind32com/client/`
    Also note that the generate wrappers do not get refreshed when the COM library changes.
    To refresh the wrappers, you can manually delete everything in the cache directory:
    ::

        `{PythonEnv}/Lib/site-packages/win32com/gen_py/*.*`
    """

    def __init__(self, name: str, comment: str = ''):
        """
        Constructor for a Zemax system
        :param name: Human-readable name of this system
        """

        # Call superclass constructor
        System.__init__(self, name, comment)

        # Initialize private variables
        self._lens_units = None
        self._thickness = None
        self._cs_break = None

        # Clear the list of surfaces that was populated by the superclass constructor.
        # This list will be rebuilt from the Zemax file.
        self._surfaces = []         # type: List[ZmxSurface]

        # Allocate space for attribute row list
        self._attrs = []                # type: List[Tuple[str, ILDERow]]

        # Initialize the connection to the Zemax system
        self._init_zos_api()

        # Open a new design
        self.zos_sys.New(saveIfNeeded=False)

        # Initialize the system from the new design
        self._init_system_from_zmx()

        self.entrance_pupil_radius = self.entrance_pupil_radius

        layout = self.zos_sys.Analyses.New_Analysis(AnalysisIDM.Draw3D)

        layout.GetSettings().Load()
        
        
    @property
    def fields(self):
        return self._fields
    
    @fields.setter
    def fields(self, value: field.Array):
        self._fields = value.promote_to_zmx(self.zos_sys)

    @property
    def wavelengths(self) -> wavelength.Array:
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, value: wavelength.Array):
        
        self._wavelengths = value.promote_to_zmx(self.zos_sys.SystemData.Wavelengths)

    @property
    def entrance_pupil_radius(self):

        if self.zos_sys is not None:

            a = self.zos_sys.SystemData.Aperture

            if a.ApertureType == ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter:
                return (a.ApertureValue / 2) * self.lens_units

            else:
                raise ValueError('Aperture not defined by entrance pupil diameter')

        else:
            return self._entrance_pupil_radius

    @entrance_pupil_radius.setter
    def entrance_pupil_radius(self, value: u.Quantity):

        if self.zos_sys is not None:

            a = self.zos_sys.SystemData.Aperture

            a.ApertureType = ZOSAPI.SystemData.ZemaxApertureType.EntrancePuilDiameter

            a.ApertureValue = 2 * value.to(self.lens_units).value

        else:
            self._entrance_pupil_radius = value

    def save(self, path: str):

        self.zos_sys.SaveAs(path)

    @staticmethod
    def from_system(sys: System) -> 'ZmxSystem':

        zmx_sys = ZmxSystem(sys.name)

        for surf in sys:

            if surf.is_object:

                zmx_surf = zmx_sys.object

                zmx_surf.name = surf.name
                zmx_surf.comment = surf.comment
                zmx_surf.thickness = surf.thickness
                zmx_surf.component = surf.component

            elif surf.is_image:

                zmx_surf = zmx_sys.image

                zmx_surf.name = surf.name
                zmx_surf.comment = surf.comment
                zmx_surf.component = surf.component
                zmx_surf.before_surf_cs_break = surf.before_surf_cs_break
                zmx_surf.after_surf_cs_break = surf.after_surf_cs_break
                zmx_surf.aperture = surf.aperture
                zmx_surf.mechanical_aperture = surf.mechanical_aperture
                zmx_surf.radius = surf.radius
                zmx_surf.conic = surf.conic
                zmx_surf.material = surf.material

            elif surf.is_stop:

                zmx_surf = zmx_sys.stop

                zmx_surf.name = surf.name
                zmx_surf.comment = surf.comment
                zmx_surf.thickness = surf.thickness
                zmx_surf.before_surf_cs_break = surf.before_surf_cs_break
                zmx_surf.after_surf_cs_break = surf.after_surf_cs_break
                zmx_surf.component = surf.component
                zmx_surf.aperture = surf.aperture

            else:

                zmx_surf = ZmxSurface.from_surface(surf)

                zmx_sys.insert(zmx_surf, -1)

        zmx_sys.entrance_pupil_radius = sys.entrance_pupil_radius
        zmx_sys.wavelengths = sys.wavelengths
        zmx_sys.fields = sys.fields

        return zmx_sys

    @staticmethod
    def from_file(name: str, filename: str) -> 'ZmxSystem':

        # Load the a blank system
        sys = ZmxSystem(name)

        # Open the file
        sys.zos_sys.LoadFile(filename, saveIfNeeded=False)

        # Read the file into the system
        sys._init_system_from_zmx()

        return sys

    @property
    def num_surfaces(self) -> int:
        """
        :return: The number of surfaces in the system
        """
        return self.zos_sys.LDE.NumberOfSurfaces

    def insert(self, surf: ZmxSurface, index: int) -> None:
        """
        Insert a new surface at the specified index in the model
        :param surf: Surface to insert into the model
        :param index: Index where the surface will be placed in the model
        :return: None
        """

        # Call superclass method to insert this surface into the list of surfaces
        super().insert(surf, index)

        # Insert a main attribute into this Zemax surface.
        # This adds an ILDERow to the surface.
        surf.insert(ZmxSurface.main_str, ZOSAPI.Editors.LDE.SurfaceType.Standard)

    def delete_surface(self, surf: ZmxSurface):

        self.surfaces.remove(surf)

    def delete_all_surfaces(self) -> None:
        """
        Remove all the surfaces in the lens data editor
        :return:
        """

        lde = self.zos_sys.LDE

        lde.RemoveSurfacesAt(1, lde.NumberOfSurfaces - 2)

    def raytrace(self, surfaces: List[ZmxSurface], wavelengths: List[wavelength.Item], field_x: u.Quantity,
                 field_y: u.Quantity, pupil_x: u.Quantity, pupil_y: u.Quantity
                 ) -> Tuple[u.Quantity, u.Quantity, u.Quantity]:

        # Grab a handle to the zemax system
        sys = self.zos_sys

        # # Deal with shorthand for last surface
        # for s, surf in enumerate(surface_indices):
        #     if surf < 0:
        #         surface_indices[s] = sys.LDE.NumberOfSurfaces + surf

        # Initialize raytrace
        rt = sys.Tools.OpenBatchRayTrace()  # raytrace object
        tool = sys.Tools.CurrentTool  # pointer to active tool

        # Store number of surfaces
        num_surf = len(surfaces)

        # store number of wavelengths
        num_wavl = len(wavelengths)

        # Store length of each axis in ray grid
        num_field_x = len(field_x)
        num_field_y = len(field_y)
        num_pupil_x = len(pupil_x)
        num_pupil_y = len(pupil_y)

        # Create grid of rays
        Fx, Fy, Px, Py = np.meshgrid(field_x, field_y, pupil_x, pupil_y, indexing='ij')

        # Store shape of grid
        sh = list(Fx.shape)

        # Shape of grid for each surface and wavelength
        tot_sh = [num_surf, num_wavl] + sh

        # Allocate output arrays
        V = np.empty(tot_sh)      # Vignetted rays
        X = np.empty(tot_sh)
        Y = np.empty(tot_sh)

        # Loop over each surface and run raytrace to surface
        for s, surf in enumerate(surfaces):

            # Run raytrace for each wavelength
            for w, wavl in enumerate(wavelengths):

                # Run raytrace for each field angle
                for fi in range(num_field_x):
                    for fj in range(num_field_y):

                        # Open instance of batch raytrace
                        rt_dat = rt.CreateNormUnpol(num_pupil_x * num_pupil_y, constants.RaysType_Real,
                                                    surf.main_row.SurfaceNumber)

                        # Loop over pupil to add rays to batch raytrace
                        for pi in range(num_pupil_x):
                            for pj in range(num_pupil_y):

                                # Select next ray
                                fx = Fx[fi, fj, pi, pj]
                                fy = Fy[fi, fj, pi, pj]
                                px = Px[fi, fj, pi, pj]
                                py = Py[fi, fj, pi, pj]

                                # Write ray to pipe
                                rt_dat.AddRay(wavl.zos_wavl.WavelengthNumber, fx, fy, px, py, constants.OPDMode_None)

                        # Execute the raytrace
                        tool.RunAndWaitForCompletion()

                        # Initialize the process of reading the results of the raytrace
                        rt_dat.StartReadingResults()

                        # Loop over pupil and read the results of the raytrace
                        for pi in range(num_pupil_x):
                            for pj in range(num_pupil_y):

                                # Read next result from pipe
                                (ret, n, err, vig, x, y, z, l, m, n, l2, m2, n2, opd, I) = rt_dat.ReadNextResult()

                                # Store next result in output arrays
                                V[s, w, fi, fj, pi, pj] = vig
                                X[s, w, fi, fj, pi, pj] = x
                                Y[s, w, fi, fj, pi, pj] = y

        return V, X, Y

    def _find_surface(self, comment: str) -> ZOSAPI.Editors.LDE.ILDERow:
        """
        Find the surface matching the provided comment string

        :param comment: Comment string to search for
        :return: Surface matching comment string
        :rtype: ZOSAPI.Editors.LDE.ILDERow
        """

        # Grab pointer to lens data editor
        lde = self.zos_sys.LDE

        # Save the number of surfaces to local variable
        n_surf = lde.NumberOfSurfaces

        # Loop through every surface and look for a match to the comment string
        for s in range(n_surf):

            # Save the pointer to this surface
            surf = lde.GetSurfaceAt(s)

            # Check if the comment of this surface matches the comment we're looking for
            if surf.Comment == comment:

                return surf

    @property
    def _rows(self):
        """
        :return: A list of all the ILDERows (ZOSAPI Surfaces) in the ZOSAPI System.
        """

        # Grab pointer to lens data editor
        lde = self.zos_sys.LDE

        # Save the number of surfaces to local variable
        n_rows = lde.NumberOfSurfaces

        # Allocate space for return variable
        rows = []

        # Loop through every surface and look for a match to the comment string
        for r in range(n_rows):

            # Grab pointer to this row
            row = lde.GetSurfaceAt(r)

            # Append row to the list of rows
            rows.append(row)

        return rows

    def _syntax_tree_from_comments(self) -> 'OrderedDict[Tuple[str, str], Tuple[Component, OrderedDict[str, ILDERow]]]':
        """
        Reads through a Zemax file and constructs a syntax tree of all the components, surfaces and attributes.
        This function is necessary to parse all the attributes into a single structure, to more easily construct a
        ZmxSurface object.
        :return: A dictionary where the keys are the names of surfaces and the values are a tuple with the zeroth element
        being the component associated with the surface, and the first element being a dictionary of attributes (where
        the key is the attribute name and the value is the Zemax row associated with the attribute).
        """

        # Allocate a dictionary to store the components parsed by this function
        components = {}                 # type: Dict[str, Component]
        surfaces = OrderedDict()     # type: OrderedDict[Tuple[str, str], Tuple[Component, OrderedDict[str, ILDERow]]]

        # Loop through every surface and look for a match to the comment string
        for r, zmx_row in enumerate(self._rows):

            # Parse the comment associated with this surface into tokens
            comp_str, surf_str, attr_str = self._parse_comment(zmx_row.Comment)

            # If there is no component provided, give the surface a default component
            if comp_str is None:

                # Special component for object surface
                if zmx_row.IsObject:
                    comp_str = System.object_str

                # # Another special component for image surfaces
                # elif zmx_row.IsImage:
                #     comp_str = System.image_str

                # Every other surface is part of the main component
                else:
                    comp_str = System.main_str

            # Initialize the component node from the token if we have not seen it already
            if comp_str not in components:
                components[comp_str] = Component(comp_str)

            # Store a pointer to the component node for later
            comp = components[comp_str]

            # Give the surface a default name if none is provided.
            if surf_str == '':
                surf_str = 'Surface' + str(r)

            # Initialize the surface node from the token if we have not seen it already
            full_str = (comp_str, surf_str)
            if full_str not in surfaces:
                surfaces[full_str] = (comp, OrderedDict())

            # Store a pointer to the surface node for later
            surf = surfaces[full_str]

            # If there is no attribute provided, assume this is the main attribute
            if attr_str is None:
                attr_str = 'main'

            # Throw error if this attribute has been seen already
            if attr_str in surf[1]:
                raise ValueError('More than one row with attr', attr_str)

            # Store the attribute string and the LDE row as key value pairs
            surf[1][attr_str] = zmx_row

            zmx_row.Comment = comp_str + '.' + surf_str + '.' + attr_str

        return surfaces

    def _init_system_from_zmx(self) -> None:
        """
        Read the current Zemax model and initialize the list of surfaces
        :return: None
        """

        self._surfaces = []

        # Parse the Zemax file and construct a list of surfaces parameters
        surfaces_dict = self._syntax_tree_from_comments()

        # Loop through the surfaces list, construct a new Zemax surface, and add it to the system
        for surf_name, surf_item in surfaces_dict.items():

            # Extract surface parameters
            component, attrs_dict = surf_item

            # Read the attributes dictionary into a ZmxSurface object
            surf = ZmxSurface.from_attr_dict(surf_name[1], attrs_dict)

            # Attach the surface to the component and to the system
            component.append(surf)

            # Append to the list of surfaces.
            # Use the superclass constructor since we don't want to add a new ILDERow to the ILensDataEditor, and this
            # is an initialization method.
            super().append(surf)

    def _update_system_from_zmx(self) -> None:
        """
        Read the current Zemax model and update the ILDERow pointer for every surface in the system.
        This function is necessary because the ZOS API is designed poorly, when a new surface is added/deleted from
        Zemax, every ILDERow shifts, so the pointer in the ZmxSurface object is out of date.
        This function reads through the Zemax model and uses the comment strings to update the ILDERow pointer in every
        ZmxSurface in the system.
        Someone looking at this function may ask why it is necessary, since you could just reinitialize the system using
        `self._init_system_from_zmx`.
        However, `self._init_system_from_zmx` would make new copies of all the ZmxSurface instances, which would be
        annoying for a user trying to save a single surface from the system.
        :return: None
        """

        # Parse the Zemax file and construct a list of surfaces parameters
        surfaces_dict = self._syntax_tree_from_comments()

        # Check that the surfaces list and the zemax model have the same number of elements
        if len(self) != len(surfaces_dict):
            raise ValueError('Number of surfaces not the same between Zemasx and Python.')

        # Loop through all the surfaces in this object and the zemax model, and update the ILDERow pointers
        for current_surf, surf_name, surf_item in zip(self.__iter__(), surfaces_dict.keys(), surfaces_dict.values()):

            # Extract component and attributes dictionary
            comp, attrs = surf_item

            # # Check that the component names at this index are the same
            # if current_surf.component != comp:
            #     raise ValueError('Two surfaces at same index with different components', current_surf.component, comp)
            #
            # # Check that the surface names at this index are the same
            # if current_surf.name != surf_name:
            #     raise ValueError('Two surfaces at same index with different names')

            # Loop through all the ILDERows in this surface, and update with new value
            for current_attr, attr_name, attr_row in zip(current_surf.__iter__(), attrs.keys(), attrs.values()):

                # Extract the name of the attribute for the persistent object
                current_attr_name, _ = current_attr

                # # Check that the attribute names at this index are the same
                # if current_attr_name != attr_name:
                #     raise ValueError('Two attributes at same index but with different names')

                # Update ILDERow value
                current_surf[current_attr_name] = attr_row

    @staticmethod
    def _parse_comment(comment_str: str) -> Tuple[str, str, str]:
        """
        Split a comment string into Zemax input tokens.
        To read in a Zemax file, the user can give hints as to what component, surface, or surface attribute each Zemax
        `ZOSAPI.Editors.LDE.ILDERow` corresponds to.
        The syntax of these hints is relatively simple: `ComponentStr.SurfaceStr.attr_str`.
        Notice that the components and surfaces use camel case, and the attributes use snake case.
        This distinction allows the user to not have to specify all three strings if they don't need to.
        If there is no attribute string (`ComponentStr.SurfaceStr`, `SurfaceStr`), the associated
        `ZOSAPI.Editors.LDE.ILDERow` is the main row for this surface.
        If there is no component string (`SurfaceStr.attr_str`, `SurfaceStr`), this surface is not part of an overall
        component.
        Note that there has to always be a `SurfaceStr`, however it can be the empty string ''.
        This means that if a surface has an empty comment string, it will end up an empty `SurfaceStr`
        :param comment_str: String to be parsed into tokens
        :return: Three values, corresponding to the component, surface and surface attribute tokens.
        """

        # Initialize return variables
        comp_name = None    # type: str
        attr_name = None    # type: str

        # Split the input string at the delimiters
        s = comment_str.split('.')

        # If there is only one token, it must be the SurfaceStr token.
        if len(s) == 1:

            # Ensure that the token is a surface string by checking that it is camel case, or that it is the empty
            # string.
            if ZmxSystem.is_camel_case(s[0]) or s[0] == '':
                surf_name = s[0]
            else:
                raise ValueError('Surface name should be camel case')

        # Else if there are two tokens, it can be either ComponentStr.SurfaceStr or SurfaceStr.attr_str
        elif len(s) == 2:

            # If the second token is camel case, it is a surface token
            if ZmxSystem.is_camel_case(s[1]):

                # Then the first token should also be camel case, to match with a component string
                if ZmxSystem.is_camel_case(s[0]):
                    comp_name = s[0]
                    surf_name = s[1]
                else:
                    raise ValueError('Component name should be camel case')

            # Else the second token uses snake case and it is an attribute token
            else:

                # Then the first token should be camel case, to match with surface string
                if ZmxSystem.is_camel_case(s[0]):
                    surf_name = s[0]
                    attr_name = s[1]
                else:
                    raise ValueError('Surface name should be camel case')

        # Else if there are three tokens, it must be ComponentStr.SurfaceStr.attr_str
        elif len(s) == 3:

            # Ensure that the component string is camel case
            if ZmxSystem.is_camel_case(s[0]):
                comp_name = s[0]
            else:
                raise ValueError('Component name should be camel case')

            # Ensure that the surface string is camel case
            if ZmxSystem.is_camel_case(s[1]):
                surf_name = s[1]
            else:
                raise ValueError('Surface name should be camel case')

            # Ensure that the attribute string is camel case
            if not ZmxSystem.is_camel_case(s[2]):
                attr_name = s[2]
            else:
                raise ValueError('Attribute name should be snake case')

        # Else there are not 1, 2 or 3 tokens and the string is undefined
        else:
            raise ValueError('Incorrect number of tokens')

        return comp_name, surf_name, attr_name

    @staticmethod
    def is_camel_case(s: str):
        """
        Utility function to determine whether a given string is camel case.
        This is defined to be: A string that has both lowercase and capital letters, and has no underscores.
        This function is based off of this stackoverflow answer: https://stackoverflow.com/a/10182901
        :param s: String to be checked
        :return: True if the string is an camel case string.
        """

        return s != s.lower() and s != s.upper() and "_" not in s

    @property
    def lens_units(self) -> u.Unit:
        """
        Extract the units used by the Lens Data Editor
        :return: Astropy units instance corresponding to the units used by Zemax
        """

        # If the units have been read from the Zemax file previously
        if self._lens_units is None:

            units = self.zos_sys.SystemData.Units.LensUnits

            if units == ZOSAPI.SystemData.ZemaxSystemUnits.Millimeters:
                self._lens_units = u.mm
            elif units == ZOSAPI.SystemData.ZemaxSystemUnits.Centimeters:
                self._lens_units = u.cm
            elif units == ZOSAPI.SystemData.ZemaxSystemUnits.Inches:
                self._lens_units = u.imperial.inch
            elif units == ZOSAPI.SystemData.ZemaxSystemUnits.Meters:
                self._lens_units = u.m
            else:
                raise ValueError('Unrecognized units')

        return self._lens_units

    @lens_units.setter
    def lens_units(self, units: u.Unit) -> None:
        """
        Set the units used by Zemax using the astropy units package
        :param units: New units for Zemax to use
        :return: None
        """

        # Update storage variable
        self._lens_units = units

        # Update Zemax
        if units == u.mm:
            self.zos_sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Millimeters
        elif units == u.cm:
            self.zos_sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Centimeters
        elif units == u.imperial.inch:
            self.zos_sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Inches
        elif units == u.m:
            self.zos_sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Meters
        else:
            raise ValueError('Unrecognized units')

    @property
    def object_dir(self):
        
        return self._app.ObjectsDir

    def __del__(self) -> None:
        """
        Delete this instance of Zemax
        :return: None
        """

        # Close application
        if self._app is not None:
            self._app.CloseApplication()
            self._app = None

        # Close connection
        self._connection = None

    def __iter__(self) -> Iterator[ZmxSurface]:

        return self._surfaces.__iter__()

    def _init_zos_api(self) -> None:

        # Create COM connection to Zemax
        self._connection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")   # type: ZOSAPI.ZOSAPI_Connection
        if self._connection is None:
            raise self.ConnectionException("Unable to initialize COM connection to ZOSAPI")

        # Open Zemax application
        self._app = self._connection.CreateNewApplication()
        if self._app is None:
            raise self.InitializationException("Unable to acquire ZOSAPI application")

        # Check if license is valid
        if self._app.IsValidLicenseForAPI is False:
            raise self.LicenseException("License is not valid for ZOSAPI use")

        # Check that we can open the primary system object
        self.zos_sys = self._app.PrimarySystem
        if self.zos_sys is None:
            raise self.SystemNotPresentException("Unable to acquire Primary system")

    def example_constants(self):
        if self._app.LicenseStatus is constants.LicenseStatusType_PremiumEdition:
            return "Premium"
        elif self._app.LicenseStatus is constants.LicenseStatusType_ProfessionalEdition:
            return "Professional"
        elif self._app.LicenseStatus is constants.LicenseStatusType_StandardEdition:
            return "Standard"
        else:
            return "Invalid"

    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass
