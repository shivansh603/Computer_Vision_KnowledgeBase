## YOLOv8n: Real-time Object Detection for Resource-Constrained Devices

**Computer Vision Task:** Object Detection

**Model Name:** YOLOv8n

**Model Architecture:**

<div style="text-align: center;">
    <img width="100%" src="https://www.researchgate.net/publication/372207753/figure/fig2/AS:11431281173322965@1688824328575/The-improved-YOLOv8-network-architecture-includes-an-additional-module-for-the-head.ppm" alt="YOLOv8n Architecture">
    <p>YOLOv8 Architecture</p>
</div>

**Description:**
YOLOv8n is a compact and efficient object detection model designed for deployment on devices with limited computational resources. It achieves a balance between speed and accuracy, making it suitable for real-time applications like embedded systems and mobile devices.

**Metrics:**

* Accuracy (mAPval 50-95): 37.3%
* Inference Time:
    * 80.4 ms per image on CPU (ONNX)
    * 0.99 ms per image on A100 TensorRT
* Model Size: 3.2M parameters, 8.7B FLOPs

**Dependencies:**

* Software: PyTorch, OpenCV, Numpy, Pillow
* Hardware: GPU (A100 TensorRT recommended), CPU

**Limitations:**

* Lower accuracy compared to larger YOLOv8 models due to the smaller size.
* May struggle with detecting very small or occluded objects.
* Limited performance on complex scenes with many objects.

**Example Detection Images**
<div style="text-align: center;">
    <img width="100%" src="https://media.hackerearth.com/blog/wp-content/uploads/2018/08/shutterstock_668209624-1.jpg" alt="YOLO Object Detection Example">
</div>

**References / Source:**
* [Ultralytics YOLOv8 Object Detection Documentation](https://docs.ultralytics.com/tasks/detect/)