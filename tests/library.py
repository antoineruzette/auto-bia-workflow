import os
os.chdir("/Users/antoine/Harvard/IAC/auto-bia-workflow/auto_bia")

from auto_bia.bia_operator import ImageLoadingOperator, SmoothingOperator, SegmentationOperator, ImageOperator
from auto_bia.bia_workflow import ImageAnalysisWorkflow

# Create image processing operators
loading_op = ImageLoadingOperator('images/Ch0050.png')
smoothing_op = SmoothingOperator("Gaussian Blur", kernel_size=5)
segmentation_op = SegmentationOperator("Thresholding", threshold=1)

# Create an image analysis workflow
workflow = ImageAnalysisWorkflow()
workflow.add_operator(loading_op)
workflow.add_operator(smoothing_op)
workflow.add_operator(segmentation_op)

# Apply the workflow to the image
result_image = workflow.run_workflow()

'''# Display the result
cv2.imshow("Result Image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
