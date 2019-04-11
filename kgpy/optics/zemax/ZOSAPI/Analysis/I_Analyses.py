
from typing import Union

from kgpy.optics.zemax import ZOSAPI

__all__ = ['I_Analyses']


class I_Analyses:

    NumberOfAnalyses = None     # type: int

    def CloseAnalysis(self, analysis: Union[int, 'ZOSAPI.Analysis.IA_']) -> bool:
        pass

    def Get_AnalysisAtIndex(self, index: int) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_Analysis(self, AnalysisType: 'ZOSAPI.Analysis.AnalysisIDM') -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_Analysis_SettingsFirst(self, AnalysisType: 'ZOSAPI.Analysis.AnalysisIDM') -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_ConfigurationMatrixSpot(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_CriticalRayTracer(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_DetectorViewer(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_DiffractionEncircledEnergy(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_ExtendedSourceEncircledEnergy(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftMtfMap(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftMtfvsField(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftPsf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftPsfCrossSection(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftPsfLineEdgeSpread(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftSurfaceMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FftThroughFocusMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FieldCurvatureAndDistortion(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FocalShiftDiagram(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_Foucault(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_FullFieldSpot(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_GeometricEncircledEnergy(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_GeometricLineEdgeSpread(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_GeometricMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_GeometricMtfMap(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_GeometricMtfvsField(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_GeometricThroughFocusMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_HuygensMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_HuygensMtfvsField(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_HuygensPsf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_HuygensPsfCrossSection(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_HuygensSurfaceMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_HuygensThroughFocusMtf(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_Interferogram(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_LateralColor(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_LongitudinalAberration(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_MatrixSpot(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_OpticalPathFan(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_PathAnalysis(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_PupilAberrationFan(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_RayFan(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_RayTrace(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_RMSField(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_RMSFieldMap(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_RMSFocus(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_RMSLambdaDiagram(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SeidelCoefficients(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SeidelDiagram(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_StandardSpot(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SurfaceCurvature(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SurfaceCurvatureCross(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SurfacePhase(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SurfacePhaseCross(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SurfaceSag(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_SurfaceSagCross(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_ThroughFocusSpot(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_WavefrontMap(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_XXXTemplateXXX(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_ZernikeAnnularCoefficients(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_ZernikeCoefficientsVsField(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_ZernikeFringeCoefficients(self) -> 'ZOSAPI.Analysis.IA_':
        pass

    def New_ZernikeStandardCoefficients(self) -> 'ZOSAPI.Analysis.IA_':
        pass

