import unittest
import numpy as np
from auto_bia import bia_workflow as biaw

dummy_image = np.ones((100, 100), dtype=np.uint8) * 255  # A white square image


class TestImageProcessingWorkflow(unittest.TestCase):

    def test_workflow_with_arrays(self):
        image_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask_array = np.zeros((100, 100), dtype=np.uint8)

        operators = [
            biaw.SmoothingOperator(kernel_size=5),
            biaw.EdgeDetectionOperator(threshold1=50, threshold2=150)
        ]
        workflow = biaw.ImageProcessingWorkflow(
            operators=operators, 
            cutoff=0.9, 
            learning_rate=0.1
            )
        result, best_combination, best_score = workflow.run(
            image_input=image_array, 
            mask_input=mask_array, 
            result_save_path=None
        )

        self.assertIsNotNone(result, "ImageProcessingWorkflow run() failed with arrays")
        self.assertIsInstance(best_combination, list, "ImageProcessingWorkflow run() failed with arrays")
        self.assertIsInstance(best_score, float, "ImageProcessingWorkflow run() failed with arrays")

    def test_workflow_with_invalid_image_array(self):
        image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # 3D array
        mask_array = np.zeros((100, 100), dtype=np.uint8)

        operators = [
            biaw.SmoothingOperator(kernel_size=5),
            biaw.EdgeDetectionOperator(threshold1=50, threshold2=150)
        ]
        workflow = biaw.ImageProcessingWorkflow(
            operators=operators, 
            cutoff=0.9, 
            learning_rate=0.1
            )

        with self.assertRaises(ValueError, msg="image_input must be a file path or a 2D numpy array"):
            workflow.run(
                image_input=image_array, 
                mask_input=mask_array, 
                result_save_path=None
                )

    def test_workflow_with_invalid_mask_array(self):
        image_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        mask_array = np.zeros((100, 100, 3), dtype=np.uint8)  # 3D array

        operators = [
            biaw.SmoothingOperator(kernel_size=5),
            biaw.EdgeDetectionOperator(threshold1=50, threshold2=150)
        ]
        workflow = biaw.ImageProcessingWorkflow(
            operators=operators, 
            cutoff=0.9, 
            learning_rate=0.1
            )

        with self.assertRaises(ValueError, msg="mask_input must be a file path or a 2D numpy array"):
            workflow.run(
                image_input=image_array, 
                mask_input=mask_array, 
                result_save_path=None
                )


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)