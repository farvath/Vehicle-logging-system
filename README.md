# Real-Time Vehicle Detection, Tracking, and Counting

This repository contains code for real-time vehicle detection, logging, and tracking . The system utilizes YOLOv8 and DeepSORT algorithms implemented using the OpenCV library in Python.

## Overview

The project aims to perform real-time vehicle logging system , making it suitable for various applications. Specifically, it was developed for our internship project at NITK.

## Features

- **Real-time Processing:** The system performs vehicle detection, counting, and tracking in real-time, enabling its use in dynamic environments.
- **Modified Dataset:** The dataset used in the project is derived from Roboflow. Initially consisting of 20 classes, it was narrowed down to 6 classes to focus on specific vehicle types based on categories, ensuring a more focused and accurate model.
- **Tesseract OCR Integration:** The system integrates Tesseract OCR for automated time extraction from captured frames. This feature enables efficient logging of vehicle passage times.

## Object Classes

The following 6 categories are used for vehicle classification:

1. Car/Jeep/Van/MV
2. LCV/LGV/Mini Bus
3. 2 axle
4. 3 axle Commercial
5. 4 to 6 axle
6. Oversized (7 axle)

## Model Architecture

The vehicle detection model is based on YOLOv8, which is trained on the modified dataset. This architecture provides efficient and accurate detection of vehicles in real-time scenarios.

## Acknowledgments

The project was made possible by leveraging the Roboflow platform for dataset management and annotation.

## Dataset Source

The dataset used in this project can be found at: [Roboflow Vehicle Detection Dataset](https://drive.google.com/drive/folders/1uk5yLv8eTlbnHup6azKl9-Ri1DXeVk1U?usp=drive_link)
