
__all__ = ['Detector', 'TestDetector']

from kso.model import Model, TestModel


class Detector(Model):

    def __init__(self, binning_model, shot_noise_model, read_noise_model):

        self.binning_model = binning_model
        self.shot_noise_model = shot_noise_model
        self.read_noise_model = read_noise_model


    def __call__(self, cube):

        cube = self.binning_model(cube)
        cube = self.shot_noise_model(cube)
        cube = self.read_noise_model(cube)

        return cube

class TestDetector(TestModel):

    def setUp(self):

        super().setUp()

    def tearDown(self):

        super().tearDown()

