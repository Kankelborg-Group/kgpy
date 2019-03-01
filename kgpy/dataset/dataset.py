from abc import ABC, abstractmethod
from unittest import TestCase

__all__ = ['Dataset', 'TestDataset']

class Dataset(ABC):

    def __init__(self):

        pass

    @abstractmethod
    def locate(self):

        pass

    @abstractmethod
    def download(self):

        pass

    @abstractmethod
    def calc_stats(self):

        pass

class TestDataset(TestCase):

    def setUp(self):

        pass

    def tearDown(self):

        pass