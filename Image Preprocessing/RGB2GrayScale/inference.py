import cv2
import argparse


class GrayscaleConverter:
    """Converts an image to grayscale and saves it."""

    def __init__(self, image_path):
        """
        Args:
          image_path (str): Path to the image file.
        """
        self.image_path = image_path
        self.image = None
        self.grayscale_image = None

    def convert(self):
        """Reads the image, converts to grayscale, and saves it."""
        # Read the image in color
        self.image = cv2.imread(self.image_path)

        # Convert the image to grayscale
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        cv2.imwrite("grayscale.jpg", self.grayscale_image)

        print(f"Grayscale image saved: grayscale.jpg")


if __name__ == "__main__":
    """Parses arguments and performs grayscale conversion."""
    parser = argparse.ArgumentParser(
        description="Convert image to grayscale with OpenCV"
    )
    parser.add_argument(
        "image_path", type=str, help="Path to the image file (Required)"
    )
    args = parser.parse_args()

    converter = GrayscaleConverter(args.image_path)
    converter.convert()