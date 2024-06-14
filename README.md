[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

# AutoBIA

AutoBIA is a Python library to optimize image analysis workflows. It allows users to create and apply various image processing operators in a sequential manner, and optimize the parameters of these operators to achieve the best results. 

## Installation

1. **Clone the repository**: Clone this repository to your local machine using Git.
The library is not stable at this point and new features are added regularly. For now, the best way to install the library is to clone the repository and locally install the package using pip.

    ```bash
    git clone https://github.com/antoineruzette/auto-bia-workflow.git
    ```
2. **Install the dependencies**: Install the dependencies using pip.

    ```bash
pip install -r requirements.txt
    ```
3. **Install the package**: Install the package using pip.

    ```bash
pip install dist/auto_bia-0.1-py3-none-any.whl
    ```

## Usage
Construct a workflow by adding operators and setting their initial parameters, then optimize workflow using the `optimize` method. The `optimize` method uses a gradient descent algorithm to find the best parameters for the operators in the workflow.

```python
from auto_bia import bia_workflow as biaw

# Create an image analysis workflow
workflow = biaw.ImageAnalysisWorkflow()
workflow.add_operator(biaw.SmoothingOperator(kernel_size=5))
workflow.add_operator(biaw.SegmentationOperator(threshold=30))
workflow.set_cutoff(0.95)

# Run the workflow
image_path = "/tests/images/original.png"
mask_path = "/tests/images/mask.png"
result_save_path = "/tests/images/result.png"
result_image, best_combination, best_score = workflow.run(image_path, 
                                                          mask_path, 
                                                          result_save_path)


```
