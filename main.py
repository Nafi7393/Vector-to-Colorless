import os
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, img_input_path, img_output_path):
        self.img_input_path = img_input_path
        self.img_output_path = img_output_path

    def process_image(self):
        try:
            # Load the image
            image = cv2.imread(self.img_input_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"Failed to read image: {self.img_input_path}")
                return False

            # Handle alpha channel if present
            if image.shape[2] == 4:
                # Create mask of pixels with alpha channel
                trans_mask = image[:, :, 3] == 0
                # Replace alpha channel with white pixels
                image[trans_mask] = [255, 255, 255, 255]
                # Convert image from BGRA to BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # Convert to grayscale
            grey_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhance edges with adaptive thresholding
            _, binary_img = cv2.threshold(grey_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_img = cv2.bitwise_not(binary_img)

            # Apply morphological operations to enhance edges
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(binary_img, kernel, iterations=1)
            edges = cv2.bitwise_not(edges)

            # Save the result
            cv2.imwrite(self.img_output_path, edges)

            print(f"Processed: {self.img_input_path}")
            return True

        except Exception as e:
            print(f"Error processing {self.img_input_path}: {str(e)}")
            return False


def main(input_folder="input", output_folder="output"):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all files in input folder
    files = os.listdir(input_folder)

    for file in files:
        # Construct paths
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")

        # Process each image
        processor = ImageProcessor(input_path, output_path)
        processor.process_image()


if __name__ == "__main__":
    main()
