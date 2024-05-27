from inference import YOLOv8Inference


if __name__ == "__main__":
    yolo_model = YOLOv8Inference()
    image_path = "temp.png"
    yolo_model.infer_image(image_path)

