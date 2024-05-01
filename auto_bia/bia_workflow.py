from bia_operator import ImageSavingOperator

class ImageAnalysisWorkflow:
    def __init__(self):
        self.operators = []

    def add_operator(self, operator):
        self.operators.append(operator)

    def run(self, image):
        result = image.copy()
        for operator in self.operators:
            result = operator.apply(result)
            ImageSavingOperator(result, "images/result.png")
        return result
