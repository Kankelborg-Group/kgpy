from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from unittest import TestCase
import numpy as np

# Notes
#
# The python project and script was tested with the following tools:
#       Python 3.4.3 for Windows (32-bit) (https://www.python.org/downloads/) - Python interpreter
#       Python for Windows Extensions (32-bit, Python 3.4) (http://sourceforge.net/projects/pywin32/) - for COM support
#       Microsoft Visual Studio Express 2013 for Windows Desktop (https://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx) - easy-to-use IDE
#       Python Tools for Visual Studio (https://pytools.codeplex.com/) - integration into Visual Studio
#
# Note that Visual Studio and Python Tools make development easier, however this python script should should run without either installed.

class Zemax(object):
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

    def __init__(self, model_path):
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

        :param model_path: Path to the Zemax model that will be opened when this constructor is called.


        """


        # Create COM connection to Zemax
        self.TheConnection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
        if self.TheConnection is None:
            raise Zemax.ConnectionException("Unable to intialize COM connection to ZOSAPI")

        # Open Zemax application
        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise Zemax.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise Zemax.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise Zemax.SystemNotPresentException("Unable to acquire Primary system")

        # Open zemax model
        self.OpenFile(model_path, False)

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None

        self.TheConnection = None

    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise Zemax.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise Zemax.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise Zemax.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus is constants.LicenseStatusType_PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_ProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus is constants.LicenseStatusType_StandardEdition:
            return "Standard"
        else:
            return "Invalid"

    def raytrace(self, surface_indices, wavl_indices, field_coords_x, field_coords_y, pupil_coords_x, pupil_coords_y):

        # Grab a handle to the zemax system
        sys = self.TheSystem

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
        # F = np.zeros((num_surf, num_field_x, num_field_y))      # Field coordinates at each surface
        # P = np.zeros((num_surf, num_pupil_x, num_pupil_y))      # Pupil coordinates at each surface
        C = np.empty(num_surf, dtype=np.str)      # Comment at each surface

        # Loop over each surface and run raytrace to surface
        for s, surf_ind in enumerate(surface_indices):

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
                                rt_dat.AddRay(w, fx, fy, px, py, constants.OPDMode_None)

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




class TestZemax(TestCase):

    def test__init(self):
        zosapi = Zemax()
        value = zosapi.ExampleConstants()

        self.assertTrue(value is not None)

        del zosapi
        zosapi = None






