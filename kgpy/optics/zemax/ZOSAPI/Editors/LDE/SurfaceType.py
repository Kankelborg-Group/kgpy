
from win32com.client import constants

__all__ = ['SurfaceType']


class SurfaceType:

    @property
    def ABCD(self) -> int:
        return constants.SurfaceType_ABCD

    @property
    def AlternateEven(self) -> int:
        return constants.SurfaceType_AlternateEven

    @property
    def AlternateOdd(self) -> int:
        return constants.SurfaceType_AlternateOdd

    @property
    def AnnularZernikeSag(self) -> int:
        return constants.SurfaceType_AnnularZernikeSag

    @property
    def Atmospheric(self) -> int:
        return constants.SurfaceType_Atmospheric

    @property
    def Biconic(self) -> int:
        return constants.SurfaceType_Biconic

    @property
    def BiconicZernike(self) -> int:
        return constants.SurfaceType_BiconicZernike

    @property
    def Binary1(self) -> int:
        return constants.SurfaceType_Binary1

    @property
    def Binary2(self) -> int:
        return constants.SurfaceType_Binary2

    @property
    def Binary3(self) -> int:
        return constants.SurfaceType_Binary3

    @property
    def Binary4(self) -> int:
        return constants.SurfaceType_Binary4

    @property
    def BirefringentIn(self) -> int:
        return constants.SurfaceType_BirefringentIn

    @property
    def BirefringentOut(self) -> int:
        return constants.SurfaceType_BirefringentOut

    @property
    def BlackBoxLens(self) -> int:
        return constants.SurfaceType_BlackBoxLens

    @property
    def ChebyShv(self) -> int:
        return constants.SurfaceType_ChebyShv

    @property
    def Conjugate(self) -> int:
        return constants.SurfaceType_Conjugate

    @property
    def CoordinateBreak(self) -> int:
        return constants.SurfaceType_CoordinateBreak

    @property
    def CubicSpline(self) -> int:
        return constants.SurfaceType_CubicSpline

    @property
    def CylinderFrensel(self) -> int:
        return constants.SurfaceType_CylinderFrensel

    @property
    def Data(self) -> int:
        return constants.SurfaceType_Data

    @property
    def DiffractionGrating(self) -> int:
        return constants.SurfaceType_DiffractionGrating

    @property
    def EllipticalGrating1(self) -> int:
        return constants.SurfaceType_EllipticalGrating1

    @property
    def EllipticalGrating2(self) -> int:
        return constants.SurfaceType_EllipticalGrating2

    @property
    def EvenAspheric(self) -> int:
        return constants.SurfaceType_EvenAspheric

    @property
    def ExtendedToroidalGrating(self) -> int:
        return constants.SurfaceType_ExtendedToroidalGrating

    @property
    def ExtendedAsphere(self) -> int:
        return constants.SurfaceType_ExtendedAsphere

    @property
    def ExtendedCubicSpline(self) -> int:
        return constants.SurfaceType_ExtendedCubicSpline

    @property
    def ExtendedFresnel(self) -> int:
        return constants.SurfaceType_ExtendedFresnel

    @property
    def ExtendedOddAsphere(self) -> int:
        return constants.SurfaceType_ExtendedOddAsphere

    @property
    def ExtendedPolynomial(self) -> int:
        return constants.SurfaceType_ExtendedPolynomial

    @property
    def Fresnel(self) -> int:
        return constants.SurfaceType_Fresnel

    @property
    def GeneralizedFresnel(self) -> int:
        return constants.SurfaceType_GeneralizedFresnel

    @property
    def Gradient1(self) -> int:
        return constants.SurfaceType_Gradient1

    @property
    def Gradient2(self) -> int:
        return constants.SurfaceType_Gradient2

    @property
    def Gradient3(self) -> int:
        return constants.SurfaceType_Gradient3

    @property
    def Gradient4(self) -> int:
        return constants.SurfaceType_Gradient4

    @property
    def Gradient5(self) -> int:
        return constants.SurfaceType_Gradient5

    @property
    def Gradient6(self) -> int:
        return constants.SurfaceType_Gradient6

    @property
    def Gradient7(self) -> int:
        return constants.SurfaceType_Gradient7

    @property
    def Gradient9(self) -> int:
        return constants.SurfaceType_Gradient9

    @property
    def Gradient10(self) -> int:
        return constants.SurfaceType_Gradient10

    @property
    def Gradient12(self) -> int:
        return constants.SurfaceType_Gradient12

    @property
    def Gradium(self) -> int:
        return constants.SurfaceType_Gradium

    @property
    def GridGradient(self) -> int:
        return constants.SurfaceType_GridGradient

    @property
    def GridPhase(self) -> int:
        return constants.SurfaceType_GridPhase

    @property
    def GridSag(self) -> int:
        return constants.SurfaceType_GridSag

    @property
    def Hologram1(self) -> int:
        return constants.SurfaceType_Hologram1

    @property
    def Hologram2(self) -> int:
        return constants.SurfaceType_Hologram2

    @property
    def Irregular(self) -> int:
        return constants.SurfaceType_Irregular

    @property
    def JonesMatrix(self) -> int:
        return constants.SurfaceType_JonesMatrix

    @property
    def OddAsphere(self) -> int:
        return constants.SurfaceType_OddAsphere

    @property
    def OddCosine(self) -> int:
        return constants.SurfaceType_OddCosine

    @property
    def OpticallyFabricatedHologram(self) -> int:
        return constants.SurfaceType_OpticallyFabricatedHologram

    @property
    def Paraxial(self) -> int:
        return constants.SurfaceType_Paraxial

    @property
    def ParaxialXY(self) -> int:
        return constants.SurfaceType_ParaxialXY

    @property
    def Periodic(self) -> int:
        return constants.SurfaceType_Periodic

    @property
    def Polynomial(self) -> int:
        return constants.SurfaceType_Polynomial

    @property
    def QTypeAsphere(self) -> int:
        return constants.SurfaceType_QTypeAsphere

    @property
    def RadialGrating(self) -> int:
        return constants.SurfaceType_RadialGrating

    @property
    def RadialNurbs(self) -> int:
        return constants.SurfaceType_RadialNurbs

    @property
    def RetroReflect(self) -> int:
        return constants.SurfaceType_RetroReflect

    @property
    def SlideSurface(self) -> int:
        return constants.SurfaceType_SlideSurface

    @property
    def Standard(self) -> int:
        return constants.SurfaceType_Standard

    @property
    def Superconic(self) -> int:
        return constants.SurfaceType_Superconic

    @property
    def Tilted(self) -> int:
        return constants.SurfaceType_Tilted

    @property
    def Toroidal(self) -> int:
        return constants.SurfaceType_Toroidal

    @property
    def ToroidalGrat(self) -> int:
        return constants.SurfaceType_

    @property
    def ToroidalHologram(self) -> int:
        return constants.SurfaceType_ToroidalGrat

    @property
    def ToroidalNurbs(self) -> int:
        return constants.SurfaceType_ToroidalNurbs

    @property
    def UserDefined(self) -> int:
        return constants.SurfaceType_UserDefined

    @property
    def VariableLineSpaceGrating(self) -> int:
        return constants.SurfaceType_VariableLineSpaceGrating

    @property
    def ZernikeAnnularPhase(self) -> int:
        return constants.SurfaceType_ZernikeAnnularPhase

    @property
    def ZernikeFringePhase(self) -> int:
        return constants.SurfaceType_ZernikeFringePhase

    @property
    def ZernikeFringeSag(self) -> int:
        return constants.SurfaceType_ZernikeFringeSag

    @property
    def ZernikeStandardPhase(self) -> int:
        return constants.SurfaceType_ZernikeStandardPhase

    @property
    def ZernikeStandardSag(self) -> int:
        return constants.SurfaceType_ZernikeStandardSag

    @property
    def ZonePlate(self) -> int:
        return constants.SurfaceType_ZonePlate
