# 🚗 Real-Time Vehicle & Pedestrian Detection (YOLOv8)

![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end Computer Vision pipeline designed for real-time street object detection (vehicles, pedestrians) using **YOLOv8** pretrained COCO weights.

---

## 🌟 Features

* **Real-time Detection:** Supports live webcam streams, video files, and batch image processing.
* **Pretrained Efficiency:** Leverages COCO pretrained weights—no heavy retraining required.
* **Interactive CLI:** Built-in console menu (`main.py`) for easy workflow navigation.
* **Modular Codebase:** Clean separation between core detection logic, utilities, and configs.

---

## 📁 Repository Structure

```text
yolov8-street-vision/
 ├── input/
 │   ├── images/          # Input images directory
 │   └── videos/          # Input videos directory
 ├── output/
 │   ├── images/          # Processed images output
 │   └── videos/          # Processed videos output
 ├── src/
 │   ├── config.py        # System threshold & class filtering configuration
 │   ├── detect_image.py  # Image batch inference module
 │   ├── detect_video.py  # Video processing module
 │   └── detect_webcam.py # Live webcam stream handler
 ├── main.py              # Interactive CLI entry point
 ├── requirements.txt     # Python dependencies
 └── README.md            # Documentation
