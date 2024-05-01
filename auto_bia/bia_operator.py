import cv2


class ImageOperator:
    def __init__(self):
        pass
    
    def apply(self, image):
        raise NotImplementedError("Subclasses must implement apply() method")
    
    
class ImageLoadingOperator(ImageOperator):
    def __init__(self, file_path, grayscale=True):
        super().__init__()
        self.file_path = file_path
        self.grayscale = grayscale

    def apply(self, image):
        if self.grayscale:
            return cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imread(self.file_path)
        
        
class ImageSavingOperator(ImageOperator):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
    
    def apply(self, image):
        cv2.imwrite(self.file_path, image)


class SmoothingOperator(ImageOperator):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
    
    def apply(self, image):
        return cv2.blur(image, (self.kernel_size, self.kernel_size))


class SegmentationOperator(ImageOperator):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    
    def apply(self, image):
        ret, segmented_image = cv2.threshold(image, 
                                             self.threshold, 
                                             255, 
                                             cv2.THRESH_BINARY)
        return segmented_image
