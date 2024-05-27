import argparse
import cv2


class CannyEdgeDetector:
    """
    A class for performing Canny edge detection on an image using OpenCV (cv2).

    Attributes:
        image_path (str): The path to the input image.
        canny_edges (numpy.ndarray, optional): The grayscale image with detected edges
            (None if not yet processed).

    Methods:
        load_image(self): Loads the image from the specified path.
        detect_edges(self, threshold1=30, threshold2=60): Applies Canny edge detection
            with optional thresholds.
        save_edges(self, output_path="edges.png"): Saves the edge image to a file.
    """

    def __init__(self, image_path):
        self.image_path = image_path
        self.canny_edges = None

    def load_image(self):
        """Loads the image from the specified path."""
        self.image = cv2.imread(
            self.image_path, cv2.IMREAD_GRAYSCALE
        )  # Read as grayscale

        if self.image is None:
            raise ValueError(f"Error: Could not read image from '{self.image_path}'.")

    def detect_edges(self, threshold1=30, threshold2=60):
        """
        Applies Canny edge detection with optional thresholds.

        Args:
            threshold1 (int, optional): The lower threshold for hysteresis thresholding.
                Defaults to 30.
            threshold2 (int, optional): The upper threshold for hysteresis thresholding.
                Defaults to 60.
        """

        # Apply Gaussian filtering for noise reduction (optional, adjust kernel size if needed)
        blurred_image = cv2.GaussianBlur(self.image, (5, 5), 0)

        # Perform Canny edge detection
        self.canny_edges = cv2.Canny(blurred_image, threshold1, threshold2)

    def save_edges(self, output_path="edges.png"):
        """
        Saves the edge image to a file.

        Args:
            output_path (str, optional): The path to save the edge image. Defaults to "edges.png".
        """

        if self.canny_edges is None:
            raise ValueError(
                "Error: Edges haven't been detected yet. Call detect_edges() first."
            )

        cv2.imwrite(output_path, self.canny_edges)
        print(f"Edges saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canny Edge Detection with OpenCV")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    edge_detector = CannyEdgeDetector(args.image_path)
    edge_detector.load_image()
    edge_detector.detect_edges()
    edge_detector.save_edges()