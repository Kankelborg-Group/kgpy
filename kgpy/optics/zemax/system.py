from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from unittest import TestCase
import numpy as np
import os
import math

from .. import System, Component, Surface
from . import ZmxSurface

__all__ = ['ZmxSystem']

# Notes
#
# The python project and script was tested with the following tools:
#       Python 3.4.3 for Windows (32-bit) (https://www.python.org/downloads/) - Python interpreter
#       Python for Windows Extensions (32-bit, Python 3.4) (http://sourceforge.net/projects/pywin32/) - for COM support
#       Microsoft Visual Studio Express 2013 for Windows Desktop (https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx) - easy-to-use IDE
#       Python Tools for Visual Studio (https://pytools.codeplex.com/) - integration into Visual Studio
#
# Note that Visual Studio and Python Tools make development easier, however this python script should should run without either installed.


class ZmxSystem(System):
    """
    A class used to interface with a particular Zemax file
    """

    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass

    def __init__(self, name, components, model_path):
        """
        Constructor for the Zemax object

        Make sure the Python wrappers are available for the COM client and interfaces
        EnsureModule('ZOSAPI_Interfaces', 0, 1, 0)
        Note - the above can also be accomplished using 'makepy.py' in the
        following directory:
        ::

             {PythonEnv}\Lib\site-packages\wind32com\client\
        Also note that the generate wrappers do not get refreshed when the COM library changes.
        To refresh the wrappers, you can manually delete everything in the cache directory:
        ::

            {PythonEnv}\Lib\site-packages\win32com\gen_py\*.*

        :param name: Human-readable name of this system
        :param components: List of components we want to divide the system into. Note that Zemax does not have the
        concept of components, which is why this list needs to provided, it tells this function how to split up the
        surfaces in the Zemax model.
        :param model_path: Path to the Zemax model that will be opened when this constructor is called.
        :type name: str
        :type components: list[kgpy.optics.Component]
        :type model_path: str
        """

        # Save input arguments
        self.name = name
        self.components = components
        self.model_path = model_path

        # Create COM connection to Zemax
        self._connection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
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

        # Open Zemax model
        self.open_file(model_path, False)

        # Loop through component list to overwrite Surfaces with ZmxSurfaces
        for component in components:

            # Loop through each surface in this component, find the corresponding Zemax surface and overwrite the
            # Surface with a ZmxSurface
            for s, surf in enumerate(component.surfaces):

                # Find Zemax surface using the comment field of the provided surface
                z_surf = self.find_surface(surf.comment)

                # Overwrite original Surface
                component.surfaces[s] = ZmxSurface(surf.name, z_surf)

    def calc_surface_intersections(self, z):
        """
        This function is used to determine which surfaces to split when inserting a baffle.
        The optics model is sequential, so if the baffle is used by rays going in different directions, we need to model
        the baffle as multiple surfaces.
        :return: List of surface indices which intersect a baffle
        :rtype: list[int]
        """

        # Grab pointer to the lens data editor in Zemax
        lde = self.sys.TheSystem.LDE

        # Initialize looping variables
        z = 0  # Test z-coordinate
        z_is_greater = False  # Flag to store what side of the baffle the test coordinate was on in the last iteration
        surfaces = []  # List of surface indices which cross a baffle

        # Loop through every surface and keep track of how often we cross the global z coordinate of the baffle
        for s in range(1, lde.NumberOfSurfaces - 1):

            # Update test z-coordinate
            z += lde.GetSurfaceAt(s).Thickness

            # Check if the updated test coordinate has crossed the baffle coordinate since the last iteration.
            # If so, append surface to list of intersecting surfaces
            if z_is_greater:  # Crossing from larger to smaller
                if z < self.z:
                    surfaces.append(s)
            else:  # Crossing from smaller to larger
                if z > self.z:
                    surfaces.append(s)

        return surfaces

    def raytrace(self, surface_indices, wavl_indices, field_coords_x, field_coords_y, pupil_coords_x, pupil_coords_y):
        """
        Execute an arbitrary raytrace of the system
        :param surface_indices: List of surfaces indices where ray positions will be evaluated.
        :param wavl_indices: List of wavelength indices to generate rays
        :param field_coords_x: array of normalized field coordinates in the x direction
        :param field_coords_y: array of normalized field coordinates in the y direction
        :param pupil_coords_x: array of normalized pupil coordinates in the x direction
        :param pupil_coords_y: array of normalized pupil coordinates in the y direction
        :type surface_indices: list[int]
        :type wavl_indices: list[int]
        :type field_coords_x: numpy.ndarray
        :type field_coords_y: numpy.ndarray
        :type pupil_coords_x: numpy.ndarray
        :type pupil_coords_y: numpy.ndarray
        :return: three ndarrrays of shape [len(surface_indices), len(wavl_indices), len(field_coords_x), len(field_coords_y), len(pupil_coords_x), len(pupil_coords_y)]
        for the vignetting code, x position and y position of every ray in the raytrace.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
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
                                (ret, n, err, vig, x, y, z, l, m, n, l2, m2, n2, opd, intensity) = rt_dat.ReadNextResult()

                                # Store next result in output arrays
                                V[s, w, fi, fj, pi, pj] = vig
                                X[s, w, fi, fj, pi, pj] = x
                                Y[s, w, fi, fj, pi, pj] = y

        return V, X, Y

    def find_surface(self, comment_str):
        """
        Find the surface matching the provided comment string

        :param comment_str: Comment string to search for
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
            if surf.Comment == comment_str:

                return surf

    def __del__(self):
        if self._app is not None:
            self._app.CloseApplication()
            self._app = None

        self._connection = None

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

        return self._app.samples_dir

    def example_constants(self):
        if self._app.LicenseStatus is constants.LicenseStatusType_PremiumEdition:
            return "Premium"
        elif self._app.LicenseStatus is constants.LicenseStatusType_ProfessionalEdition:
            return "Professional"
        elif self._app.LicenseStatus is constants.LicenseStatusType_StandardEdition:
            return "Standard"
        else:
            return "Invalid"


class TestZmxSystem(TestCase):

    test_path = os.path.join(os.path.dirname(__file__), 'test_model.zmx')

    def setUp(self):
        name = 'test surface'
        stop = Component('Stop', [Surface('stop', comment='Stop')])
        lens = Component('Primary', [Surface('primary', comment='Primary')])
        img = Component('Detector', [Surface('detector', comment='Detector')])
        self.components = [stop, lens, img]

        self.sys = ZmxSystem(name=name, components=self.components, model_path=self.test_path)

    def tearDown(self):

        del self.sys
        self.sys = None

    def test__init(self):

        # Check that Zemax instance started correctly
        self.assertTrue(self.sys.example_constants() is not None)

        # Check that the zmx_surf field contains a valid object for every surface in every component
        for orig_comp, zmx_comp in zip(self.components, self.sys.components):
            for orig_surf, zmx_surf in zip(orig_comp.surfaces, zmx_comp.surfaces):

                # Check that all comment strings are equal
                self.assertTrue(orig_surf.comment, zmx_surf.comment)

    def test_raytrace(self):

        surf_indices = [-1]
        wavl_indices = [1]
        fx = [0.0]
        fy = [1.0]
        px = [0.0]
        py = [0.0]

        V, X, Y = self.zmx.raytrace(surf_indices, wavl_indices, fx, fy, px, py)

        self.assertTrue(V[0] == 0.)
        self.assertTrue(X[0] == 0.)
        self.assertTrue(math.isclose(Y[0], 100 * np.sin(np.radians(1.0)), abs_tol=1e-3))

    def test_find_surface(self):

        primary_comment = 'Primary'
        true_primary_ind = 2

        primary_surf = self.zmx.find_surface(primary_comment)

        print(type(primary_surf))

        self.assertTrue(primary_surf.SurfaceNumber == true_primary_ind)

    def test_find_surface_no_match(self):

        unmatching_comment = 'asdfasdf'

        surf = self.zmx.find_surface(unmatching_comment)

        self.assertTrue(surf is None)







