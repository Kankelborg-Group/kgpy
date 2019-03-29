
from win32com.client import constants

__all__ = ['SurfaceType']


class SurfaceType:

    @property
    def ABCD(self) -> 'SurfaceType':
        return constants.SurfaceType_ABCD

    @property
    def AlternateEven(self) -> 'SurfaceType':
        return constants.SurfaceType_AlternateEven

    @property
    def AlternateOdd(self) -> 'SurfaceType':
        return constants.SurfaceType_AlternateOdd

    @property
    def AnnularZernikeSag(self) -> 'SurfaceType':
        return constants.SurfaceType_AnnularZernikeSag

    @property
    def Atmospheric(self) -> 'SurfaceType':
        return constants.SurfaceType_Atmospheric

    @property
    def Biconic(self) -> 'SurfaceType':
        return constants.SurfaceType_Biconic

    @property
    def BiconicZernike(self) -> 'SurfaceType':
        return constants.SurfaceType_BiconicZernike

    @property
    def Binary1(self) -> 'SurfaceType':
        return constants.SurfaceType_Binary1

    @property
    def Binary2(self) -> 'SurfaceType':
        return constants.SurfaceType_Binary2

    @property
    def Binary3(self) -> 'SurfaceType':
        return constants.SurfaceType_Binary3

    @property
    def Binary4(self) -> 'SurfaceType':
        return constants.SurfaceType_Binary4

    @property
    def BirefringentIn(self) -> 'SurfaceType':
        return constants.SurfaceType_BirefringentIn

    @property
    def BirefringentOut(self) -> 'SurfaceType':
        return constants.SurfaceType_BirefringentOut

    @property
    def BlackBoxLens(self) -> 'SurfaceType':
        return constants.SurfaceType_BlackBoxLens

    @property
    def ChebyShv(self) -> 'SurfaceType':
        return constants.SurfaceType_ChebyShv

    @property
    def Conjugate(self) -> 'SurfaceType':
        return constants.SurfaceType_Conjugate

    @property
    def CoordinateBreak(self) -> 'SurfaceType':
        return constants.SurfaceType_CoordinateBreak

    @property
    def CubicSpline(self) -> 'SurfaceType':
        return constants.SurfaceType_CubicSpline

    @property
    def CylinderFrensel(self) -> 'SurfaceType':
        return constants.SurfaceType_CylinderFrensel

    @property
    def Data(self) -> 'SurfaceType':
        return constants.SurfaceType_Data

    @property
    def DiffractionGrating(self) -> 'SurfaceType':
        return constants.SurfaceType_DiffractionGrating

    @property
    def EllipticalGrating1(self) -> 'SurfaceType':
        return constants.SurfaceType_EllipticalGrating1

    @property
    def EllipticalGrating2(self) -> 'SurfaceType':
        return constants.SurfaceType_EllipticalGrating2

    @property
    def EvenAspheric(self) -> 'SurfaceType':
        return constants.SurfaceType_EvenAspheric

    @property
    def ExtendedToroidalGrating(self) -> 'SurfaceType':
        return constants.SurfaceType_ExtendedToroidalGrating

    @property
    def ExtendedAsphere(self) -> 'SurfaceType':
        return constants.SurfaceType_ExtendedAsphere

    @property
    def ExtendedCubicSpline(self) -> 'SurfaceType':
        return constants.SurfaceType_ExtendedCubicSpline

    @property
    def ExtendedFresnel(self) -> 'SurfaceType':
        return constants.SurfaceType_ExtendedFresnel

    @property
    def ExtendedOddAsphere(self) -> 'SurfaceType':
        return constants.SurfaceType_ExtendedOddAsphere

    @property
    def ExtendedPolynomial(self) -> 'SurfaceType':
        return constants.SurfaceType_ExtendedPolynomial

    @property
    def Fresnel(self) -> 'SurfaceType':
        return constants.SurfaceType_Fresnel

    @property
    def GeneralizedFresnel(self) -> 'SurfaceType':
        return constants.SurfaceType_GeneralizedFresnel

    @property
    def Gradient1(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient1

    @property
    def Gradient2(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient2

    @property
    def Gradient3(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient3

    @property
    def Gradient4(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient4

    @property
    def Gradient5(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient5

    @property
    def Gradient6(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient6

    @property
    def Gradient7(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient7

    @property
    def Gradient9(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient9

    @property
    def Gradient10(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient10

    @property
    def Gradient12(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradient12

    @property
    def Gradium(self) -> 'SurfaceType':
        return constants.SurfaceType_Gradium

    @property
    def GridGradient(self) -> 'SurfaceType':
        return constants.SurfaceType_GridGradient

    @property
    def GridPhase(self) -> 'SurfaceType':
        return constants.SurfaceType_GridPhase

    @property
    def GridSag(self) -> 'SurfaceType':
        return constants.SurfaceType_GridSag

    @property
    def Hologram1(self) -> 'SurfaceType':
        return constants.SurfaceType_Hologram1

    @property
    def Hologram2(self) -> 'SurfaceType':
        return constants.SurfaceType_Hologram2

    @property
    def Irregular(self) -> 'SurfaceType':
        return constants.SurfaceType_Irregular

    @property
    def JonesMatrix(self) -> 'SurfaceType':
        return constants.SurfaceType_JonesMatrix

    @property
    def OddAsphere(self) -> 'SurfaceType':
        return constants.SurfaceType_OddAsphere

    @property
    def OddCosine(self) -> 'SurfaceType':
        return constants.SurfaceType_OddCosine

    @property
    def OpticallyFabricatedHologram(self) -> 'SurfaceType':
        return constants.SurfaceType_OpticallyFabricatedHologram

    @property
    def Paraxial(self) -> 'SurfaceType':
        return constants.SurfaceType_Paraxial

    @property
    def ParaxialXY(self) -> 'SurfaceType':
        return constants.SurfaceType_ParaxialXY

    @property
    def Periodic(self) -> 'SurfaceType':
        return constants.SurfaceType_Periodic

    @property
    def Polynomial(self) -> 'SurfaceType':
        return constants.SurfaceType_Polynomial

    @property
    def QTypeAsphere(self) -> 'SurfaceType':
        return constants.SurfaceType_QTypeAsphere

    @property
    def RadialGrating(self) -> 'SurfaceType':
        return constants.SurfaceType_RadialGrating

    @property
    def RadialNurbs(self) -> 'SurfaceType':
        return constants.SurfaceType_RadialNurbs

    @property
    def RetroReflect(self) -> 'SurfaceType':
        return constants.SurfaceType_RetroReflect

    @property
    def SlideSurface(self) -> 'SurfaceType':
        return constants.SurfaceType_SlideSurface

    @property
    def Standard(self) -> 'SurfaceType':
        return constants.SurfaceType_Standard

    @property
    def Superconic(self) -> 'SurfaceType':
        return constants.SurfaceType_Superconic

    @property
    def Tilted(self) -> 'SurfaceType':
        return constants.SurfaceType_Tilted

    @property
    def Toroidal(self) -> 'SurfaceType':
        return constants.SurfaceType_Toroidal

    @property
    def ToroidalGrat(self) -> 'SurfaceType':
        return constants.SurfaceType_

    @property
    def ToroidalHologram(self) -> 'SurfaceType':
        return constants.SurfaceType_ToroidalGrat

    @property
    def ToroidalNurbs(self) -> 'SurfaceType':
        return constants.SurfaceType_ToroidalNurbs

    @property
    def UserDefined(self) -> 'SurfaceType':
        return constants.SurfaceType_UserDefined

    @property
    def VariableLineSpaceGrating(self) -> 'SurfaceType':
        return constants.SurfaceType_VariableLineSpaceGrating

    @property
    def ZernikeAnnularPhase(self) -> 'SurfaceType':
        return constants.SurfaceType_ZernikeAnnularPhase

    @property
    def ZernikeFringePhase(self) -> 'SurfaceType':
        return constants.SurfaceType_ZernikeFringePhase

    @property
    def ZernikeFringeSag(self) -> 'SurfaceType':
        return constants.SurfaceType_ZernikeFringeSag

    @property
    def ZernikeStandardPhase(self) -> 'SurfaceType':
        return constants.SurfaceType_ZernikeStandardPhase

    @property
    def ZernikeStandardSag(self) -> 'SurfaceType':
        return constants.SurfaceType_ZernikeStandardSag

    @property
    def ZonePlate(self) -> 'SurfaceType':
        return constants.SurfaceType_ZonePlate


# Make singleton class
SurfaceType = SurfaceType()
