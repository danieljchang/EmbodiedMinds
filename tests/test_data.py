import unittest
from src.datasets.dataloader import DataLoader
from src.preprocessing.augmentation import augment_data

class TestDataLoading(unittest.TestCase):

    def setUp(self):
        self.data_loader = DataLoader()
        self.sample_data = self.data_loader.load_data('path/to/sample/data')

    def test_data_loading(self):
        self.assertIsNotNone(self.sample_data)
        self.assertGreater(len(self.sample_data), 0)

class TestDataAugmentation(unittest.TestCase):

    def test_augmentation(self):
        augmented_data = augment_data(self.sample_data)
        self.assertIsNotNone(augmented_data)
        self.assertNotEqual(augmented_data, self.sample_data)

if __name__ == '__main__':
    unittest.main()