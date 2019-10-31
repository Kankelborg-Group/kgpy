
import typing as tp
import win32com.client
import astropy.units as u

from kgpy.optics import system
from kgpy.optics.zemax import ZOSAPI




def add_aperture_to_zemax_surface(zemax_system: ZOSAPI.IOpticalSystem,
                                  zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                  zemax_units: u.Unit,
                                  configuration_index: int,
                                  surface_index: int,
                                  aperture: 'optics.system.configuration.surface.Aperture'):
    aperture_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APTP,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APDX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APDY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APMN,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.APMX
    ]

    if isinstance(aperture, kgpy.optics.system.surface.aperture.Rectangular):

        if configuration_index == 0:

            for op_type in aperture_op_types:
                op = zemax_system.MCE.AddOperand()
                op.ChangeType(op_type)
                op.Param1 = surface_index

        if aperture.is_obscuration:
            zemax_aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularObscuration
        else:
            zemax_aperture_type = ZOSAPI.Editors.LDE.SurfaceApertureTypes.RectangularAperture

        zemax_aperture_settings = zemax_surface.ApertureData.CreateApertureTypeSettings(
            zemax_aperture_type)._S_RectangularAperture  # type: ZOSAPI.Editors.LDE.ISurfaceApertureRectangular

        zemax_aperture_settings.XHalfWidth = aperture.half_width_x.to(zemax_units).value
        zemax_aperture_settings.YHalfWidth = aperture.half_width_y.to(zemax_units).value
        zemax_aperture_settings.ApertureXDecenter = aperture.decenter_x.to(zemax_units).value
        zemax_aperture_settings.ApertureYDecenter = aperture.decenter_y.to(zemax_units).value

        zemax_surface.ApertureData.ChangeApertureTypeSettings(zemax_aperture_settings)



def add_standard_surface_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                  zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                  zemax_units: u.Unit,
                                  configuration_index: int,
                                  surface_index: int,
                                  surface: 'optics.system.configuration.surface.Standard'):

    standard_surface_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CRVT,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.GLSS,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CONN,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBDY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBTZ,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CBOR,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CADX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CADY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CATX,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CATY,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CATZ,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.CAOR,
    ]

    if configuration_index == 0:

        for op_type in standard_surface_op_types:
            op = zemax_system.MCE.AddOperand()
            op.ChangeType(op_type)
            op.Param1 = surface_index

    zemax_surface.Radius = surface.radius.to(zemax_units).value
    zemax_surface.Material = str(surface.material)
    zemax_surface.Conic = float(surface.conic)

    if surface.pre_tilt_decenter.translation_first:
        zemax_surface.TiltDecenterData.BeforeSurfaceOrder = 0

    else:
        zemax_surface.TiltDecenterData.BeforeSurfaceOrder = 1

    translation = surface.pre_tilt_decenter.translation
    zemax_surface.TiltDecenterData.BeforeSurfaceDecenterX = translation.x.to(zemax_units).value
    zemax_surface.TiltDecenterData.BeforeSurfaceDecenterY = translation.y.to(zemax_units).value

    rotation = surface.pre_tilt_decenter.rotation

    zemax_surface.TiltDecenterData.BeforeSurfaceTiltX = rotation.x.to(u.deg).value
    zemax_surface.TiltDecenterData.BeforeSurfaceTiltY = rotation.y.to(u.deg).value
    zemax_surface.TiltDecenterData.BeforeSurfaceTiltZ = rotation.z.to(u.deg).value

    if surface.post_tilt_decenter.translation_first:
        zemax_surface.TiltDecenterData.AfterSurfaceOrder = 0

    else:
        zemax_surface.TiltDecenterData.AfterSurfaceOrder = 1

    translation = surface.post_tilt_decenter.translation
    zemax_surface.TiltDecenterData.AfterSurfaceDecenterX = translation.x.to(zemax_units).value
    zemax_surface.TiltDecenterData.AfterSurfaceDecenterY = translation.y.to(zemax_units).value



    rotation = surface.pre_tilt_decenter.rotation

    zemax_surface.TiltDecenterData.AfterSurfaceTiltX = rotation.x.to(u.deg).value
    zemax_surface.TiltDecenterData.AfterSurfaceTiltY = rotation.y.to(u.deg).value
    zemax_surface.TiltDecenterData.AfterSurfaceTiltZ = rotation.z.to(u.deg).value

    add_aperture_to_zemax_surface(zemax_system, zemax_surface, zemax_units, configuration_index, surface_index,
                                  surface.aperture)



def add_diffraction_grating_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                  zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                  configuration_index: int,
                                  surface_index: int,
                                  surface: 'optics.system.configuration.surface.DiffractionGrating'):

    n_params = 2

    if configuration_index == 0:

        for p in range(n_params):
            op = zemax_system.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM)
            op.Param1 = surface_index
            op.Param2 = p + 1

    zemax_surface_settings = zemax_surface.GetSurfaceTypeSettings(
        ZOSAPI.Editors.LDE.SurfaceType.DiffractionGrating)

    zemax_surface.ChangeType(zemax_surface_settings)

    zemax_surface_data = win32com.client.CastTo(
        zemax_surface.SurfaceData,
        ZOSAPI.Editors.LDE.ISurfaceDiffractionGrating.__name__
    )  # type: ZOSAPI.Editors.LDE.ISurfaceDiffractionGrating

    zemax_surface_data.DiffractionOrder = surface.diffraction_order
    zemax_surface_data.LinesPerMicroMeter = surface.groove_frequency.to(1 / u.um).value


def add_toroidal_surface_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                         zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                         zemax_units: u.Unit,
                                         configuration_index: int,
                                         surface_index: int,
                                         surface: 'optics.system.configuration.surface.Toroidal'):
    n_params = 14

    if configuration_index == 0:

        for p in range(n_params):
            op = zemax_system.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM)
            op.Param1 = surface_index
            op.Param2 = p + 1

    zemax_surface_settings = zemax_surface.GetSurfaceTypeSettings(
        ZOSAPI.Editors.LDE.SurfaceType.Toroidal)

    zemax_surface.ChangeType(zemax_surface_settings)

    zemax_surface_data = win32com.client.CastTo(
        zemax_surface.SurfaceData,
        ZOSAPI.Editors.LDE.ISurfaceToroidal.__name__
    )  # type: ZOSAPI.Editors.LDE.ISurfaceToroidal

    zemax_surface_data.RadiusOfRotation = surface.radius_of_rotation.to(zemax_units).value


def add_coordinate_break_surface_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                                 zemax_surface: ZOSAPI.Editors.LDE.ILDERow,
                                                 zemax_units: u.Unit,
                                                 configuration_index: int,
                                                 surface_index: int,
                                                 surface: 'optics.system.configuration.surface.CoordinateBreak'):

    n_params = 6

    if configuration_index == 0:
        for p in range(n_params):
            op = zemax_system.MCE.AddOperand()
            op.ChangeType(ZOSAPI.Editors.MCE.MultiConfigOperandType.PRAM)
            op.Param1 = surface_index
            op.Param2 = p + 1

    zemax_surface_settings = zemax_surface.GetSurfaceTypeSettings(
        ZOSAPI.Editors.LDE.SurfaceType.CoordinateBreak)

    zemax_surface.ChangeType(zemax_surface_settings)

    zemax_surface_data = win32com.client.CastTo(
        zemax_surface.SurfaceData,
        ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak.__name__
    )  # type: ZOSAPI.Editors.LDE.ISurfaceCoordinateBreak



    if surface.tilt_decenter.translation_first:
        zemax_surface_data.Order = 0

    else:
        zemax_surface_data.Order = 1

    translation = surface.tilt_decenter.translation
    zemax_surface_data.Decenter_X = translation.x.to(zemax_units).value
    zemax_surface_data.Decenter_Y = translation.y.to(zemax_units).value

    rotation = surface.tilt_decenter.rotation

    zemax_surface_data.TiltAbout_X = rotation.x.to(u.deg).value
    zemax_surface_data.TiltAbout_Y = rotation.y.to(u.deg).value
    zemax_surface_data.TiltAbout_Z = rotation.z.to(u.deg).value


def add_surfaces_to_zemax_system(zemax_system: ZOSAPI.IOpticalSystem,
                                 zemax_units: u.Unit,
                                 configuration_index: int,
                                 surfaces: 'tp.List[optics.system.configuration.Surface]'):
    surface_op_types = [
        ZOSAPI.Editors.MCE.MultiConfigOperandType.MCOM,
        ZOSAPI.Editors.MCE.MultiConfigOperandType.THIC,
    ]

    zemax_stop_surface = None

    for surface_index, surface in enumerate(surfaces):

        if configuration_index == 0:

            if surface_index >= zemax_system.LDE.NumberOfSurfaces - 1 <= surface_index < len(surfaces) - 1:
                zemax_system.LDE.AddSurface()

            for op_type in surface_op_types:
                op = zemax_system.MCE.AddOperand()
                op.ChangeType(op_type)
                op.Param1 = surface_index


        zemax_surface = zemax_system.LDE.GetSurfaceAt(surface_index)

        zemax_surface.Comment = surface.name
        # zemax_surface.IsStop = surface.is_stop
        zemax_surface.Thickness = surface.thickness.to(zemax_units).value

        if surface.is_stop:
            zemax_surface.IsStop = surface.is_stop

        if isinstance(surface, kgpy.optics.system.surface.Standard):

            add_standard_surface_to_zemax_system(zemax_system, zemax_surface, zemax_units, configuration_index,
                                                 surface_index, surface)

            if isinstance(surface, kgpy.optics.system.surface.DiffractionGrating):

                add_diffraction_grating_to_zemax_system(zemax_system, zemax_surface, configuration_index, surface_index,
                                                        surface)

            if isinstance(surface, kgpy.optics.system.surface.Toroidal):

                add_toroidal_surface_to_zemax_system(zemax_system, zemax_surface, zemax_units, configuration_index,
                                                     surface_index, surface)

        if isinstance(surface, kgpy.optics.system.surface.CoordinateBreak):

            add_coordinate_break_surface_to_zemax_system(zemax_system, zemax_surface, zemax_units, configuration_index,
                                                         surface_index, surface)






