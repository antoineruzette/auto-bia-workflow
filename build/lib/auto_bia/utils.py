import cv2
from skimage.metrics import structural_similarity as ssim


def load_image(file_path, grayscale=True):
    """
    Loads an image from the specified file path.
    """
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(file_path)
    if image is None:
        print(f"[WARN] Failed to load image from {file_path}")
    return image


def save_image(image, file_path):
    """
    Saves an image to the specified file path.
    """
    cv2.imwrite(file_path, image)


def compute_similarity(mask, image):
    """
    Computes the structural similarity index between the mask and the image.
    """
    return ssim(mask, image)
