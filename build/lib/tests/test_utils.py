import unittest
import numpy as np
from auto_bia import bia_workflow as biaw

# Create a dummy image for testing purposes
dummy_image_path = '/mnt/data/sample_image.jpg'
dummy_image = np.ones((100, 100), dtype=np.uint8) * 255  # A white square image
# Save the dummy image
biaw.save_image(dummy_image, dummy_image_path)


class TestUtils(unittest.TestCase):

    def test_load_image(self):
        image = biaw.load_image('/mnt/data/sample_image.jpg', grayscale=True)
        self.assertIsNotNone(image, "Image loading failed")

    def test_save_image(self):
        image = np.zeros((100, 100), dtype=np.uint8)
        biaw.save_image(image, '/mnt/data/test_save_image.jpg')
        loaded_image = biaw.load_image('/mnt/data/test_save_image.jpg', grayscale=True)
        self.assertTrue(np.array_equal(image, loaded_image), "Image saving failed")

    def test_compute_similarity(self):
        image1 = np.ones((100, 100), dtype=np.uint8)
        image2 = np.ones((100, 100), dtype=np.uint8) * 255
        similarity = biaw.compute_similarity(image1, image2)
        self.assertLess(similarity, 0.1, "Similarity computation failed")


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)
