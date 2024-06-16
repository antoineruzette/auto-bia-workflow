import unittest
import numpy as np
from auto_bia import bia_workflow as biaw


class TestImageOperators(unittest.TestCase):

    def test_smoothing_operator(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        operator = biaw.SmoothingOperator(kernel_size=5)
        smoothed_image = operator.apply(image)
        self.assertEqual(smoothed_image.shape, image.shape, "SmoothingOperator apply() failed")

    def test_equalization_operator(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        operator = biaw.EqualizationOperator(clip_limit=2.0, tile_grid_size=(15, 15))
        equalized_image = operator.apply(image)
        self.assertEqual(equalized_image.shape, image.shape, "EqualizationOperator apply() failed")

    def test_segmentation_operator(self):
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        operator = biaw.SegmentationOperator(threshold=128)
        segmented_image = operator.apply(image)
        self.assertEqual(segmented_image.shape, image.shape, "SegmentationOperator apply() failed")


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
