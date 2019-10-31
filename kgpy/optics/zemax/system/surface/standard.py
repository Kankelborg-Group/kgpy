from astropy import units as u

from kgpy.optics.zemax import ZOSAPI
from kgpy.optics.zemax.system.surface.aperture import add_aperture_to_zemax_surface


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