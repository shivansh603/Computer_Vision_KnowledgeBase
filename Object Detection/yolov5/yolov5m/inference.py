import argparse
from ultralytics import YOLO

class YOLOv5Inference:
    """
    A class to test YOLOv5 models on single images.

    Attributes:
    - model: YOLOv5 model instance.
    """

    def __init__(self, model_path):
        """
        Initializes the YOLOv5Inference with the specified model.

        Args:
        - model_path (str): Path to the YOLOv5 model file.
        """
        self.model = YOLO(model_path)

    def infer_image(self, image_path):
        """
        Tests the YOLOv5 model on a single image and displays the results.

        Args:
        - image_path (str): Path to the image to be tested.

        Returns:
        None
        """
        # Perform inference on the image
        results = self.model(image_path)

        # Extract and display the results
        for result in results:
            result.show()
            result.save(filename='result.jpg')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test YOLOv5 model on a single image")
    parser.add_argument("--model_path", type=str, required=True, default="yolov5m.pt", help="Path to the YOLOv5 model file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to be tested")
    args = parser.parse_args()

    # Create YOLOv5Tester object
    infer = YOLOv5Inference(args.model_path)

    # Test image using YOLOv5Tester
    infer.infer_image(args.image_path)
