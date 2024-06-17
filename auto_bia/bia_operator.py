import cv2
import numpy as np
from auto_bia.utils import load_image


class ImageOperator:
    def __init__(self):
        pass
    
    def apply(self, image):
        raise NotImplementedError("Subclasses must implement apply() method")
    
    def get_params(self):
        raise NotImplementedError("Subclasses must implement get_params() method")
    
    def set_params(self, params):
        raise NotImplementedError("Subclasses must implement set_params() method")
    
    def compute_gradients(self, image, mask, apply_operators, compute_similarity):
        return np.array([])  # No gradients to compute

    def __repr__(self):
        return self.__class__.__name__


class ImageLoadingOperator(ImageOperator):
    def __init__(self, file_path, grayscale=True):
        super().__init__()
        self.file_path = file_path
        self.grayscale = grayscale

    def apply(self, image=None):
        return load_image(self.file_path, self.grayscale)
        
    def get_params(self):
        return np.array([])  # No parameters to optimize

    def set_params(self, params):
        pass

    def compute_gradients(self, image, mask, apply_operators, compute_similarity):
        return np.array([])  # No gradients to compute


class SmoothingOperator(ImageOperator):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
    
    def apply(self, image):
        return cv2.blur(image, (self.kernel_size, self.kernel_size))
    
    def get_params(self):
        return np.array([self.kernel_size])
    
    def set_params(self, params):
        self.kernel_size = int(params[0])
    
    def compute_gradients(self, image, mask, apply_operators, compute_similarity):
        original_image = apply_operators(image)
        score = compute_similarity(mask, original_image)
        
        self.kernel_size += 1
        new_image = apply_operators(image)
        new_score = compute_similarity(mask, new_image)
        gradient = new_score - score
        self.kernel_size -= 1
        
        return np.array([gradient])


class SegmentationOperator(ImageOperator):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
    
    def apply(self, image):
        ret, segmented_image = cv2.threshold(image, 
                                             self.threshold, 
                                             255, 
                                             cv2.THRESH_OTSU)
        return segmented_image
    
    def get_params(self):
        return np.array([self.threshold])
    
    def set_params(self, params):
        self.threshold = params[0]
    
    def compute_gradients(self, image, mask, apply_operators, compute_similarity):
        original_image = apply_operators(image)
        score = compute_similarity(mask, original_image)
        
        self.threshold += 1
        new_image = apply_operators(image)
        new_score = compute_similarity(mask, new_image)
        gradient = new_score - score
        self.threshold -= 1
        
        return np.array([gradient])


class EqualizationOperator:
    def __init__(self, clip_limit, tile_grid_size):
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(map(int, tile_grid_size))  # Ensure tile_grid_size is a tuple of integers

    def apply(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                tileGridSize=self.tile_grid_size)
        return clahe.apply(image)
    
    def get_params(self):
        return np.array([self.clip_limit, *self.tile_grid_size])
    
    def set_params(self, params):
        self.clip_limit = params[0]
        self.tile_grid_size = (int(params[1]), int(params[2]))
    
    def compute_gradients(self, image, mask, apply_operators, compute_similarity):
        original_image = apply_operators(image)
        score = compute_similarity(mask, original_image)
        
        # Compute gradient for clip_limit
        delta = 0.1
        self.clip_limit += delta
        new_image = apply_operators(image)
        new_score = compute_similarity(mask, new_image)
        gradient_clip_limit = (new_score - score) / delta
        self.clip_limit -= delta
        
        # Compute gradient for tile_grid_size
        gradient_tile_grid_size = []
        for i in range(2):
            if i == 0:
                self.tile_grid_size = (int(self.tile_grid_size[0] + delta), self.tile_grid_size[1])
            else:
                self.tile_grid_size = (self.tile_grid_size[0], int(self.tile_grid_size[1] + delta))
            
            new_image = apply_operators(image)
            new_score = compute_similarity(mask, new_image)
            gradient_tile_grid_size.append((new_score - score) / delta)
            
            if i == 0:
                self.tile_grid_size = (int(self.tile_grid_size[0] - delta), self.tile_grid_size[1])
            else:
                self.tile_grid_size = (self.tile_grid_size[0], int(self.tile_grid_size[1] - delta))
        
        return np.array([gradient_clip_limit, *gradient_tile_grid_size])
