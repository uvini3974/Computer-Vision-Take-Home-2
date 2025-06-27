import os
import cv2
from Q1 import generate_image, add_gaussian_noise


os.makedirs("output", exist_ok=True)

image = generate_image(300, 300)
cv2.imshow("Original Image (3 Pixel Values)", image)
cv2.imwrite("output/original_image.png", image)


noisy_image = add_gaussian_noise(image)
cv2.imshow("Noisy Image", noisy_image)
cv2.imwrite("output/noisy_image.png", noisy_image)

_, otsu_result = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu Thresholded Image", otsu_result)
cv2.imwrite("output/otsu_thresholded_image.png", otsu_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
