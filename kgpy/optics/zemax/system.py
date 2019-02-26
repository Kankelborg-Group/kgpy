
from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from typing import List, Union, Tuple
from numbers import Real
import numpy as np
import astropy.units as u

from kgpy.optics import System, Component, Surface
from kgpy.optics.zemax import ZmxSurface, ZOSAPI

__all__ = ['ZmxSystem']


class ZmxSystem(System):
    """
    A class used to interface with a particular Zemax file
    """

    def __init__(self, name: str, model_path: str):
        """
        Constructor for the Zemax object

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

        :param name: Human-readable name of this system
        :param components: List of components we want to divide the system into. Note that Zemax does not have the
        concept of components, which is why this list needs to provided, it tells this function how to split up the
        surfaces in the Zemax model.
        :param model_path: Path to the Zemax model that will be opened when this constructor is called.
        """

        # Save input arguments
        self.name = name
        self.model_path = model_path

        # Initialize the connection to the Zemax system
        self._init_sys()

        self.first_surface = None

        # Open Zemax model
        self.open_file(model_path, False)



    @property
    def surfaces(self) -> List[ZmxSurface]:

        # Allocate space for the list to be returned from this function
        surfaces = []

        # Grab pointer to lens data editor
        lde = self._sys.LDE

        # Save the number of surfaces to local variable
        n_surf = lde.NumberOfSurfaces

        # Loop through every surface and look for a match to the comment string
        for s in range(n_surf):

            # Save the pointer to this surface
            surf = lde.GetSurfaceAt(s)

            # Add to the return list
            surfaces.append(surf)

        return surfaces

    def raytrace(self, surface_indices: List[int], wavl_indices: List[int],
                 field_coords_x: Union[List[Real], np.ndarray], field_coords_y: Union[List[Real], np.ndarray],
                 pupil_coords_x: Union[List[Real], np.ndarray], pupil_coords_y: Union[List[Real], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute an arbitrary raytrace of the system
        :param surface_indices: List of surfaces indices where ray positions will be evaluated.
        :param wavl_indices: List of wavelength indices to generate rays
        :param field_coords_x: array of normalized field coordinates in the x direction
        :param field_coords_y: array of normalized field coordinates in the y direction
        :param pupil_coords_x: array of normalized pupil coordinates in the x direction
        :param pupil_coords_y: array of normalized pupil coordinates in the y direction
        :return: three ndarrrays of shape [len(surface_indices), len(wavl_indices), len(field_coords_x),
        len(field_coords_y), len(pupil_coords_x), len(pupil_coords_y)] for the vignetting code, x position and
        y position of every ray in the raytrace.
        """

        # Grab a handle to the zemax system
        sys = self._sys

        # Deal with shorthand for last surface
        for s, surf in enumerate(surface_indices):
            if surf < 0:
                surface_indices[s] = sys.LDE.NumberOfSurfaces + surf

        # Initialize raytrace
        rt = sys.Tools.OpenBatchRayTrace()  # raytrace object
        tool = sys.Tools.CurrentTool  # pointer to active tool

        # Store number of surfaces
        num_surf = len(surface_indices)

        # store number of wavelengths
        num_wavl = len(wavl_indices)

        # Store length of each axis in ray grid
        num_field_x = len(field_coords_x)
        num_field_y = len(field_coords_y)
        num_pupil_x = len(pupil_coords_x)
        num_pupil_y = len(pupil_coords_y)

        # Create grid of rays
        Fx, Fy, Px, Py = np.meshgrid(field_coords_x, field_coords_y, pupil_coords_x, pupil_coords_y, indexing='ij')

        # Store shape of grid
        sh = list(Fx.shape)

        # Shape of grid for each surface and wavelength
        tot_sh = [num_surf, num_wavl] + sh

        # Allocate output arrays
        V = np.empty(tot_sh)      # Vignetted rays
        X = np.empty(tot_sh)
        Y = np.empty(tot_sh)
        C = np.empty(num_surf, dtype=np.str)      # Comment at each surface

        # Loop over each surface and run raytrace to surface
        for s, surf_ind in enumerate(surface_indices):

            print('surface', surf_ind)

            # Save comment at this surface
            # C.append(sys.LDE.GetSurfaceAt(surf_ind).Comment)
            C[s] = sys.LDE.GetSurfaceAt(surf_ind).Comment

            # Run raytrace for each wavelength
            for w, wavl_ind in enumerate(wavl_indices):

                # Run raytrace for each field angle
                for fi in range(num_field_x):
                    for fj in range(num_field_y):

                        # Open instance of batch raytrace
                        rt_dat = rt.CreateNormUnpol(num_pupil_x * num_pupil_y, constants.RaysType_Real, surf_ind)

                        # Loop over pupil to add rays to batch raytrace
                        for pi in range(num_pupil_x):
                            for pj in range(num_pupil_y):

                                # Select next ray
                                fx = Fx[fi, fj, pi, pj]
                                fy = Fy[fi, fj, pi, pj]
                                px = Px[fi, fj, pi, pj]
                                py = Py[fi, fj, pi, pj]

                                # Write ray to pipe
                                rt_dat.AddRay(wavl_ind, fx, fy, px, py, constants.OPDMode_None)

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

    def find_surface(self, comment: str) -> ZOSAPI.Editors.LDE.ILDERow:
        """
        Find the surface matching the provided comment string

        :param comment: Comment string to search for
        :return: Surface matching comment string
        :rtype: ZOSAPI.Editors.LDE.ILDERow
        """

        # Grab pointer to lens data editor
        lde = self._sys.LDE

        # Save the number of surfaces to local variable
        n_surf = lde.NumberOfSurfaces

        # Loop through every surface and look for a match to the comment string
        for s in range(n_surf):

            # Save the pointer to this surface
            surf = lde.GetSurfaceAt(s)

            # Check if the comment of this surface matches the comment we're looking for
            if surf.Comment == comment:

                return surf

    def build_sys_from_comments(self):

        # Allocate a dictionary to store the components parsed by this function
        components = {}

        # Grab pointer to lens data editor
        lde = self._sys.LDE

        # Save the number of surfaces to local variable
        n_surf = lde.NumberOfSurfaces

        # Loop through every surface and look for a match to the comment string
        for s in range(n_surf):

            # Save the pointer to this surface
            zmx_surf = lde.GetSurfaceAt(s)

            # Parse the comment associated with this surface into tokens
            comp_str, surf_str, attr_str = self.parse_comment(zmx_surf.Comment)

            # If there is no component provided, assume the surface belongs to the main component
            if comp_str is None:
                comp_str = 'Main'

            # Initialize the component from the token if we have not seen it already
            if comp_str not in components:
                components[comp_str] = Component(comp_str)

            # Store a pointer to the token for later
            comp = components[comp_str]

            # Check to see if this surface has already been seen
            if any([srf.name == surf_str for srf in comp.surfaces]):

                # Initialize the surface if it has not been already and add it to the component
                # Note that the zmx_surf argument is not defined since we don't know it yet.
                surf = ZmxSurface(surf_str, None, self.lens_units)
                comp.append_surface(surf)

            # If the attribute is None, this is the main ILDERow for this surface
            if attr_str is None:
                surf.zmx_surf = zmx_surf

            # Otherwise the attribute is not None and this IDLERow will be placed into a dictionary, where the key
            # is the attribute that this ILDERow corresponds to.
            else:

                # If we have npt seen this attribute string before, add the IDLERow to the dictionary.
                if attr_str not in surf.attr_surfaces:
                    surf.attr_surfaces[attr_str] = zmx_surf

                # If we've seen this attribute before, that is incorrect syntax.
                else:
                    raise ValueError('Attribute already defined')





    @staticmethod
    def parse_comment(comment_str: str) -> Tuple[str, str, str]:
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
        Note that there has to always be a `SurfaceStr`.
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

            # Ensure that the token is a surface string by checking that it is camel case
            if ZmxSystem.is_camel_case(s[0]):
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

        units = self._sys.SystemData.Units.LensUnits

        if units == constants.ZemaxSystemUnits_Millimeters:
            return u.mm
        elif units == constants.ZemaxSystemUnits_Centimeters:
            return u.cm
        elif units == constants.ZemaxSystemUnits_Inches:
            return u.imperial.inch
        elif units == constants.ZemaxSystemUnits_Meters:
            return u.m
        else:
            raise ValueError('Unrecognized units')

    @lens_units.setter
    def lens_units(self, units: u.Unit) -> None:
        """
        Set the units used by Zemax using the astropy units package
        :param units: New units for Zemax to use
        :return: None
        """

        if units == u.mm:
            self._sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Millimeters
        elif units == u.cm:
            self._sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Centimeters
        elif units == u.imperial.inch:
            self._sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Inches
        elif units == u.m:
            self._sys.SystemData.Units.LensUnits = constants.ZemaxSystemUnits_Meters
        else:
            raise ValueError('Unrecognized units')

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

    def _init_sys(self) -> None:

        # Create COM connection to Zemax
        self._connection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")   # type: ZOSAPI.ZOSAPI_Connection
        if self._connection is None:
            raise self.ConnectionException("Unable to intialize COM connection to ZOSAPI")

        # Open Zemax application
        self._app = self._connection.CreateNewApplication()
        if self._app is None:
            raise self.InitializationException("Unable to acquire ZOSAPI application")

        # Check if license is valid
        if self._app.IsValidLicenseForAPI is False:
            raise self.LicenseException("License is not valid for ZOSAPI use")

        # Check that we can open the primary system object
        self._sys = self._app.PrimarySystem
        if self._sys is None:
            raise self.SystemNotPresentException("Unable to acquire Primary system")

    def open_file(self, filepath, saveIfNeeded):
        if self._sys is None:
            raise self.SystemNotPresentException("Unable to acquire Primary system")
        self._sys.LoadFile(filepath, saveIfNeeded)

    def close_file(self, save):
        if self._sys is None:
            raise self.SystemNotPresentException("Unable to acquire Primary system")
        self._sys.Close(save)

    def samples_dir(self):
        if self._app is None:
            raise self.InitializationException("Unable to acquire ZOSAPI application")

        return self._app.SamplesDir

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
