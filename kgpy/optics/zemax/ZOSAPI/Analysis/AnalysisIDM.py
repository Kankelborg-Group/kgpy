
from win32com.client import constants

__all__ = ['AnalysisIDM']


class AnalysisIDMBase:

    @property
    def XXXTemplateXXX(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_XXXTemplateXXX

    @property
    def RayFan(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RayFan

    @property
    def OpticalPathFan(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_OpticalPathFan

    @property
    def PupilAberrationFan(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PupilAberrationFan

    @property
    def FieldCurvatureAndDistortion(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FieldCurvatureAndDistortion

    @property
    def FocalShiftDiagram(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FocalShiftDiagram

    @property
    def GridDistortion(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GridDistortion

    @property
    def LateralColor(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_LateralColor

    @property
    def LongitudinalAberration(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_LongitudinalAberration

    @property
    def RayTrace(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RayTrace

    @property
    def SeidelCoefficients(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SeidelCoefficients

    @property
    def SeidelDiagram(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SeidelDiagram

    @property
    def ZernikeAnnularCoefficients(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ZernikeAnnularCoefficients

    @property
    def ZernikeCoefficientsVsField(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ZernikeCoefficientsVsField

    @property
    def ZernikeFringeCoefficients(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ZernikeFringeCoefficients

    @property
    def ZernikeStandardCoefficients(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ZernikeStandardCoefficients

    @property
    def FftMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FftMtf

    @property
    def FftThroughFocusMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FftThroughFocusMtf

    @property
    def GeometricThroughFocusMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricThroughFocusMtf

    @property
    def GeometricMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricMtf

    @property
    def FftMtfMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FftMtfMap

    @property
    def GeometricMtfMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricMtfMap

    @property
    def FftSurfaceMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FftSurfaceMtf

    @property
    def FftMtfvsField(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FftMtfvsField

    @property
    def GeometricMtfvsField(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricMtfvsField

    @property
    def HuygensMtfvsField(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_HuygensMtfvsField

    @property
    def HuygensMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_HuygensMtf

    @property
    def HuygensSurfaceMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_HuygensSurfaceMtf

    @property
    def HuygensThroughFocusMtf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_HuygensThroughFocusMtf

    @property
    def FftPsf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_

    @property
    def FftPsfCrossSection(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FftPsfCrossSection

    @property
    def FftPsfLineEdgeSpread(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FftPsfLineEdgeSpread

    @property
    def HuygensPsfCrossSection(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_HuygensPsfCrossSection

    @property
    def HuygensPsf(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_HuygensPsf

    @property
    def DiffractionEncircledEnergy(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_DiffractionEncircledEnergy

    @property
    def GeometricEncircledEnergy(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricEncircledEnergy

    @property
    def GeometricLineEdgeSpread(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricLineEdgeSpread

    @property
    def ExtendedSourceEncircledEnergy(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ExtendedSourceEncircledEnergy

    @property
    def SurfaceCurvatureCross(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SurfaceCurvatureCross

    @property
    def SurfacePhaseCross(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SurfacePhaseCross

    @property
    def SurfaceSagCross(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SurfaceSagCross

    @property
    def SurfaceCurvature(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SurfaceCurvature

    @property
    def SurfacePhase(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SurfacePhase

    @property
    def SurfaceSag(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SurfaceSag

    @property
    def StandardSpot(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_StandardSpot

    @property
    def ThroughFocusSpot(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ThroughFocusSpot

    @property
    def FullFieldSpot(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FullFieldSpot

    @property
    def MatrixSpot(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_MatrixSpot

    @property
    def ConfigurationMatrixSpot(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ConfigurationMatrixSpot

    @property
    def RMSField(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RMSField

    @property
    def RMSFieldMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RMSFieldMap

    @property
    def RMSLambdaDiagram(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RMSLambdaDiagram

    @property
    def RMSFocus(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RMSFocus

    @property
    def Foucault(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_Foucault

    @property
    def Interferogram(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_Interferogram

    @property
    def WavefrontMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_WavefrontMap

    @property
    def Draw2D(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_Draw2D

    @property
    def Draw3D(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_Draw3D

    @property
    def ImageSimulation(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ImageSimulation

    @property
    def GeometricImageAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricImageAnalysis

    @property
    def IMABIMFileViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_IMABIMFileViewer

    @property
    def GeometricBitmapImageAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GeometricBitmapImageAnalysis

    @property
    def BitmapFileViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_BitmapFileViewer

    @property
    def LightSourceAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_LightSourceAnalysis

    @property
    def PartiallyCoherentImageAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PartiallyCoherentImageAnalysis

    @property
    def ExtendedDiffractionImageAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ExtendedDiffractionImageAnalysis

    @property
    def BiocularFieldOfViewAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_BiocularFieldOfViewAnalysis

    @property
    def BiocularDipvergenceConvergence(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_BiocularDipvergenceConvergence

    @property
    def RelativeIllumination(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RelativeIllumination

    @property
    def VignettingDiagramSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_VignettingDiagramSettings

    @property
    def FootprintSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FootprintSettings

    @property
    def YYbarDiagram(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_YYbarDiagram

    @property
    def PowerFieldMapSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PowerFieldMapSettings

    @property
    def PowerPupilMapSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PowerPupilMapSettings

    @property
    def IncidentAnglevsImageHeight(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_IncidentAnglevsImageHeight

    @property
    def FiberCouplingSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FiberCouplingSettings

    @property
    def YNIContributions(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_YNIContributions

    @property
    def SagTable(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SagTable

    @property
    def CardinalPoints(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_CardinalPoints

    @property
    def DispersionDiagram(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_DispersionDiagram

    @property
    def GlassMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GlassMap

    @property
    def AthermalGlassMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_AthermalGlassMap

    @property
    def InternalTransmissionvsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_InternalTransmissionvsWavelength

    @property
    def DispersionvsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_DispersionvsWavelength

    @property
    def GrinProfile(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GrinProfile

    @property
    def GradiumProfile(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_GradiumProfile

    @property
    def UniversalPlot1D(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_UniversalPlot1D

    @property
    def UniversalPlot2D(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_UniversalPlot2D

    @property
    def PolarizationRayTrace(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PolarizationRayTrace

    @property
    def PolarizationPupilMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PolarizationPupilMap

    @property
    def Transmission(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_Transmission

    @property
    def PhaseAberration(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PhaseAberration

    @property
    def TransmissionFan(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_TransmissionFan

    @property
    def ParaxialGaussianBeam(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ParaxialGaussianBeam

    @property
    def SkewGaussianBeam(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SkewGaussianBeam

    @property
    def PhysicalOpticsPropagation(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PhysicalOpticsPropagation

    @property
    def BeamFileViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_BeamFileViewer

    @property
    def ReflectionvsAngle(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ReflectionvsAngle

    @property
    def TransmissionvsAngle(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_TransmissionvsAngle

    @property
    def AbsorptionvsAngle(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_AbsorptionvsAngle

    @property
    def DiattenuationvsAngle(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_DiattenuationvsAngle

    @property
    def PhasevsAngle(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PhasevsAngle

    @property
    def RetardancevsAngle(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RetardancevsAngle

    @property
    def ReflectionvsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ReflectionvsWavelength

    @property
    def TransmissionvsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_TransmissionvsWavelength

    @property
    def AbsorptionvsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_AbsorptionvsWavelength

    @property
    def DiattenuationvsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_DiattenuationvsWavelength

    @property
    def PhasevsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PhasevsWavelength

    @property
    def RetardancevsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RetardancevsWavelength

    @property
    def DirectivityPlot(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_DirectivityPlot

    @property
    def SourcePolarViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SourcePolarViewer

    @property
    def PhotoluminscenceViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PhotoluminscenceViewer

    @property
    def SourceSpectrumViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SourceSpectrumViewer

    @property
    def RadiantSourceModelViewerSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RadiantSourceModelViewerSettings

    @property
    def SurfaceDataSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SurfaceDataSettings

    @property
    def PrescriptionDataSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PrescriptionDataSettings

    @property
    def FileComparatorSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FileComparatorSettings

    @property
    def PartViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PartViewer

    @property
    def ReverseRadianceAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ReverseRadianceAnalysis

    @property
    def PathAnalysis(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PathAnalysis

    @property
    def FluxvsWavelength(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_FluxvsWavelength

    @property
    def RoadwayLighting(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RoadwayLighting

    @property
    def SourceIlluminationMap(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SourceIlluminationMap

    @property
    def ScatterFunctionViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ScatterFunctionViewer

    @property
    def ScatterPolarPlotSettings(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ScatterPolarPlotSettings

    @property
    def ZemaxElementDrawing(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ZemaxElementDrawing

    @property
    def ShadedModel(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ShadedModel

    @property
    def NSCShadedModel(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_NSCShadedModel

    @property
    def NSC3DLayout(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_NSC3DLayout

    @property
    def NSCObjectViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_NSCObjectViewer

    @property
    def RayDatabaseViewer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_RayDatabaseViewer

    @property
    def ISOElementDrawing(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_ISOElementDrawing

    @property
    def SystemData(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SystemData

    @property
    def TestPlateList(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_TestPlateList

    @property
    def SourceColorChart1931(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SourceColorChart1931

    @property
    def SourceColorChart1976(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_SourceColorChart1976

    @property
    def PrescriptionGraphic(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_PrescriptionGraphic

    @property
    def CriticalRayTracer(self) -> 'AnalysisIDM':
        return constants.AnalysisIDM_CriticalRayTracer


AnalysisIDM = AnalysisIDMBase()
