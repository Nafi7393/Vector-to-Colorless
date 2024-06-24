import cv2
import os
import numpy as np

def img_color_remover(files):
    for file in files:
        # Load the image
        image = cv2.imread(f"input/{file}", cv2.IMREAD_UNCHANGED)

        # Check if the image has an alpha channel
        if image.shape[2] == 4:
            # If so, replace transparent parts with white
            trans_mask = image[:, :, 3] == 0
            image[trans_mask] = [255, 255, 255, 255]
            # Convert from BGRA to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        # Convert to grayscale
        grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Enhance edges with adaptive thresholding
        _, binary_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert the binary image
        binary_img = cv2.bitwise_not(binary_img)

        # Apply morphological operations to enhance edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(binary_img, kernel, iterations=1)

        # Invert the edges to get black lines on white background
        edges = cv2.bitwise_not(edges)

        # Save the result
        cv2.imwrite(f"output/{file}", edges)


def get_files(folder_name="input"):
    filenames = next(os.walk(f"{folder_name}"), (None, None, []))[2]  # [] if no file
    return filenames


# Ensure the output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

# Process images
img_color_remover(get_files())
