
from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from unittest import TestCase
from typing import List, Union, Tuple
from numbers import Real
import numpy as np
import math

from kgpy.optics import System, Component, Surface
from . import ZmxSurface

__all__ = ['ZmxSystem']


class ZmxSystem(System):
    """
    A class used to interface with a particular Zemax file
    """

    def __init__(self, name: str, components: List[Component], model_path: str):
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
        self.components = components
        self.model_path = model_path

        self.first_surface = None

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
                component.surfaces[s] = ZmxSurface(surf.name, z_surf, self._sys)

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

    def find_surface(self, comment: str) -> 'ZOSAPI.Editors.ILDERow':
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

    class LicenseException(Exception):
        pass

    class ConnectionException(Exception):
        pass

    class InitializationException(Exception):
        pass

    class SystemNotPresentException(Exception):
        pass
