import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def show_segmentation(mask):
    cv2.imshow('Segmentation Progress', mask)
    cv2.waitKey(1)

def region_growing(image, seed_points, threshold_range):
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)

    seed_intensity = image[seed_points[0][1], seed_points[0][0]]

    queue = list(seed_points)
    for x, y in seed_points:
        visited[y, x] = True
        mask[y, x] = 255

    while queue:
        x, y = queue.pop(0)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    pixel_value = image[ny, nx]
                    if abs(int(pixel_value) - int(seed_intensity)) <= threshold_range:
                        visited[ny, nx] = True
                        mask[ny, nx] = 255
                        queue.append((nx, ny))

        if len(queue) % 50 == 0:
            show_segmentation(mask)

    return mask

def main():
    # Load grayscale image
    image = cv2.imread('coin.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("❌ Error: Could not load image. Check the file path.")
        return

    # Define seed points (x, y)
    seed_points = [(75, 100), (170, 100), (270, 100)]  # Adjust if needed

    # Set threshold
    threshold = 10

    # Run region growing algorithm
    result = region_growing(image, seed_points, threshold)

    # Create output folder
    os.makedirs("output", exist_ok=True)

    # Save result
    cv2.imwrite("output/segmented_result.png", result)

    # Save original with seed points using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    for x, y in seed_points:
        plt.plot(x, y, 'ro')
    plt.title("Original Image with Seed Points")
    plt.axis('off')
    plt.savefig("output/original_with_seeds.png", bbox_inches='tight')
    plt.close()

    # Show results
    cv2.imshow("Original Image", image)
    cv2.imshow("Region Growing Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
