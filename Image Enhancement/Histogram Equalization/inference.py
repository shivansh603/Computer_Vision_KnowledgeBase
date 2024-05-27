import argparse
import cv2
import os


class ImageProcessor:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.image = None
        self.equalized_image = None

    def read_image(self):
        self.image = cv2.imread(self.image_path, 0)

    def equalize_histogram(self):
        if self.image is not None:
            self.equalized_image = cv2.equalizeHist(self.image)
        else:
            print("Please read an image first.")

    def write_image(self):
        if self.equalized_image is not None:
            cv2.imwrite(self.output_path, self.equalized_image)
        else:
            print("No equalized image to write. Please equalize the histogram first.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Histogram Equalization")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("output_path", type=str, help="Path to save the output image")
    args = parser.parse_args()

    image_processor = ImageProcessor(args.image_path, args.output_path)
    image_processor.read_image()
    image_processor.equalize_histogram()
    image_processor.write_image()

    print(f"Equalized image saved to {args.output_path}")