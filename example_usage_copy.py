from auto_bia import bia_workflow as biaw

workflow = biaw.ImageAnalysisWorkflow()
workflow.add_operator(biaw.SmoothingOperator(kernel_size=5))
workflow.add_operator(biaw.SegmentationOperator(threshold=30))
workflow.set_cutoff(0.95)


# Run it
image_path = "tests/images/original.png"
mask_path = "tests/images/mask.png"
result_save_path = "/tests/images/result.png"
workflow.run(
    image_input=image_path, 
    mask_input=mask_path, 
    result_save_path=result_save_path
    )
