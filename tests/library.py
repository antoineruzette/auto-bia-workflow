"""import os
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
from auto_bia import bia_workflow as biaw

# Create an image analysis workflow
workflow = biaw.ImageAnalysisWorkflow()
workflow.add_operator(biaw.SmoothingOperator(kernel_size=5))
workflow.add_operator(biaw.SegmentationOperator(threshold=30))
workflow.set_cutoff(0.95)

# Run the workflow
image_path = "/Users/antoine/Harvard/IAC/auto-bia-workflow/tests/images/original.png"
mask_path = "/Users/antoine/Harvard/IAC/auto-bia-workflow/tests/images/mask.png"
result_save_path = "/Users/antoine/Harvard/IAC/auto-bia-workflow/tests/images/result.png"
result_image, best_combination, best_score = workflow.run(image_path, 
                                                          mask_path, 
                                                          result_save_path)
