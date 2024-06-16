import pytest
import numpy as np
from auto_bia import bia_workflow as biaw


def test_compute_similarity():
    image1 = np.ones((100, 100), dtype=np.uint8)
    image2 = np.ones((100, 100), dtype=np.uint8)
    similarity = biaw.compute_similarity(image1, image2)
    assert similarity == 1.0, "Similarity computation failed"
