import pytest
import numpy as np
from auto_bia import bia_workflow as biaw

dummy_image = np.ones((100, 100), dtype=np.uint8) * 255  # A white square image


def test_workflow_with_arrays():
    image_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    mask_array = np.zeros((100, 100), dtype=np.uint8)

    operators = [
        biaw.SmoothingOperator(kernel_size=5),
        biaw.SegmentationOperator(threshold=128)
    ]
    workflow = biaw.ImageAnalysisWorkflow(
        operators=operators, 
        cutoff=0.9, 
        learning_rate=0.1
    )
    result, best_combination, best_score = workflow.run(
        image_input=image_array, 
        mask_input=mask_array, 
        result_save_path=None
    )

    assert result is not None, "ImageProcessingWorkflow run() failed with arrays"
    assert isinstance(best_combination, list), "ImageProcessingWorkflow run() failed with arrays"
    assert isinstance(best_score, float), "ImageProcessingWorkflow run() failed with arrays"


def test_workflow_with_invalid_image_array():
    image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # 3D array
    mask_array = np.zeros((100, 100), dtype=np.uint8)

    operators = [
        biaw.SmoothingOperator(kernel_size=5),
        biaw.SegmentationOperator(threshold=128)
    ]
    workflow = biaw.ImageAnalysisWorkflow(
        operators=operators, 
        cutoff=0.9, 
        learning_rate=0.1
    )

    with pytest.raises(ValueError, match="image_input must be a file path or a 2D numpy array"):
        workflow.run(
            image_input=image_array, 
            mask_input=mask_array, 
            result_save_path=None
        )


def test_workflow_with_invalid_mask_array():
    image_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    mask_array = np.zeros((100, 100, 3), dtype=np.uint8)  # 3D array

    operators = [
        biaw.SmoothingOperator(kernel_size=5),
        biaw.SegmentationOperator(threshold=128)
    ]
    workflow = biaw.ImageAnalysisWorkflow(
        operators=operators, 
        cutoff=0.9, 
        learning_rate=0.1
    )

    with pytest.raises(ValueError, match="mask_input must be a file path or a 2D numpy array"):
        workflow.run(
            image_input=image_array, 
            mask_input=mask_array, 
            result_save_path=None
        )
