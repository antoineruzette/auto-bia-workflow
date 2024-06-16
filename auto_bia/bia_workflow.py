import numpy as np
from auto_bia.utils import load_image, save_image, compute_similarity
from auto_bia.bia_operator import *


class ImageAnalysisWorkflow:
    """A class to represent an image analysis workflow.

    A image analysis workflow is a sequence of operators that are applied to an image to
    achieve a desired result. The workflow can be optimized using gradient descent to
    maximize the similarity score between the result and a given mask.

    Attributes: 
        operators (list): The list of operators to apply to the image.
        cutoff (float): The similarity score cutoff to stop the optimization.
        max_iter (int): The maximum number of iterations to run the optimization.
        learning_rate (float): The learning rate for the gradient descent optimization.
    
    Methods:
        add_operator(operator): Add an operator to the workflow.
        run(image_path, mask_path, result_save_path): Run the workflow on an image.
        optimize(image, mask): Optimize the workflow using gradient descent.
    """
    def __init__(self, operators=None, cutoff=0.9, max_iter=100, learning_rate=0.1):
        self._operators = operators if operators is not None else []
        self._cutoff = cutoff
        self._max_iter = max_iter
        self._learning_rate = learning_rate

    @property
    def cutoff(self):
        return self._cutoff
    
    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value

    @property
    def operators(self):
        return self._operators
    
    @operators.setter
    def operators(self, operator):
        self._operators.append(operator)

    @property
    def max_iter(self):
        return self._max_iter
    
    @max_iter.setter
    def max_iter(self, value):
        self._max_iter = value

    @property
    def learning_rate(self):
        return self._learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        self._learning_rate = value

    def apply_operators(self, image):
        result = image.copy()
        for operator in self._operators:
            result = operator.apply(result)
        return result
    
    def run(self, image_input, mask_input, result_save_path=None):
        if isinstance(image_input, str):
            print(f"Loading image from {image_input}")
            image = load_image(file_path=image_input)
            print(image.shape)
        elif isinstance(image_input, np.ndarray) and image_input.ndim == 2:
            image = image_input
        else:
            raise ValueError("image_input must be a file path or a 2D numpy array")
        
        if isinstance(mask_input, str):
            print(f"Loading mask from {mask_input}")
            mask = load_image(file_path=mask_input, grayscale=True)
            print(image.shape)
        elif isinstance(mask_input, np.ndarray) and mask_input.ndim == 2:
            mask = mask_input
        else:
            raise ValueError("mask_input must be a file path or a 2D numpy array")

        # Perform gradient descent to optimize the operators' parameters
        best_result, best_combination, best_score = self.optimize(image, mask)

        if result_save_path:
            print(f"Saving best result to {result_save_path}")
            save_image(best_result, result_save_path)

        return best_result, best_combination, best_score
    
    def optimize(self, image, mask):
        best_score = -1
        best_combination = None
        best_result = image.copy()

        for i in range(self.max_iter):
            print(f"Iteration {i + 1}/{self.max_iter}")

            for idx, operator in enumerate(self._operators):
                # Calculate gradient
                original_params = operator.get_params()
                gradients = operator.compute_gradients(
                    best_result, 
                    mask, 
                    self.apply_operators, 
                    compute_similarity
                    )

                # Update parameters
                new_params = original_params - self.learning_rate * gradients
                operator.set_params(new_params)

                # Apply operator with new parameters
                result = self.apply_operators(best_result)

                # Compute similarity score
                score = compute_similarity(mask, result)
                print(f"Gradients: {gradients}")
                print(f"New parameters: {new_params}")
                print(f"Similarity score: {score}")

                # Check if this is the best score
                if score > best_score:
                    best_score = score
                    best_combination = [op.get_params() for op in self._operators]
                    best_result = result.copy()

                # If the score exceeds the cutoff, break
                if score >= self.cutoff:
                    print(f"Cutoff reached: {score}")
                    break

                # Revert to original parameters
                operator.set_params(original_params)

            # Break the outer loop if the score exceeds the cutoff
            if best_score >= self.cutoff:
                break

        # Set the operators to the best combination found
        for idx, operator in enumerate(self._operators):
            operator.set_params(best_combination[idx])

        return best_result, best_combination, best_score
