from inference import ObjectDetector
import cv2 





if __name__ == "__main__":
    detector = ObjectDetector()
    image_path = "Sample_images/1.png"
    image = cv2.imread(image_path)
    detector.detect_objects(image)