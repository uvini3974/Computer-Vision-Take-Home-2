import numpy as np
import cv2

def generate_image(width, height):
    """Generate a new image with a triangle and an ellipse using 3 pixel values."""
    
    image = np.ones((height, width), dtype=np.uint8) * 255  

    triangle_cnt = np.array([
        [width // 4, height // 4],
        [width // 8, 3 * height // 4],
        [width // 2, 3 * height // 4]
    ])
    cv2.drawContours(image, [triangle_cnt], 0, 128, -1)


    center = (3 * width // 4, height // 2)
    axes = (width // 6, height // 4)
    cv2.ellipse(image, center, axes, angle=0, startAngle=0, endAngle=360, color=0, thickness=-1)

    return image

def add_gaussian_noise(image, mean=0, stddev=50):
    """Add Gaussian noise to a grayscale image."""
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image
