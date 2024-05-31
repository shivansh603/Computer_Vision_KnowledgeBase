import argparse
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = None
        self.equalized_image = None

    def read_image(self):
        self.image = cv2.imread(self.image_path, 0)

    def equalize_histogram(self):
        if self.image is not None:
            self.equalized_image = cv2.equalizeHist(self.image)
        else:
            print("Please read an image first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram Equalization")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    image_processor = ImageProcessor(args.image_path)
    image_processor.read_image()
    image_processor.equalize_histogram()

    # Display the original and equalized images (you need to have a GUI environment for this)
    cv2.imshow("Original Image", image_processor.image)
    cv2.imshow("Equalized Image", image_processor.equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
