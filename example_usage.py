from auto_bia import bia_workflow as biaw

# Create an image analysis workflow
workflow = biaw.ImageAnalysisWorkflow(
    operators=[
        biaw.SmoothingOperator(kernel_size=5),
        biaw.EqualizationOperator(clip_limit=2.0, tile_grid_size=(8, 8)),
        biaw.SegmentationOperator(threshold=30)
        ], 
    cutoff=0.99, 
    learning_rate=0.1
    )

# Run it
image_path = "tests/images/original.png"
mask_path = "tests/images/mask.png"
result_save_path = "tests/images/result.png"

workflow.run(
    image_input=image_path, 
    mask_input=mask_path, 
    result_save_path=result_save_path
    )
