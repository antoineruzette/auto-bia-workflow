import pytest
import numpy as np
from auto_bia import bia_workflow as biaw


def test_smoothing_operator():
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    operator = biaw.SmoothingOperator(kernel_size=5)
    smoothed_image = operator.apply(image)
    assert smoothed_image.shape == image.shape, "SmoothingOperator apply() failed"


def test_equalization_operator():
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    operator = biaw.EqualizationOperator(clip_limit=2.0, tile_grid_size=(15, 15))
    equalized_image = operator.apply(image)
    assert equalized_image.shape == image.shape, "EqualizationOperator apply() failed"


def test_segmentation_operator():
    image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    operator = biaw.SegmentationOperator(threshold=128)
    segmented_image = operator.apply(image)
    assert segmented_image.shape == image.shape, "SegmentationOperator apply() failed"
