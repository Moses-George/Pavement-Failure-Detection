# Machine Learning for Flexible Pavement Defect Detection and Classification

Automated detection of pavement defects using YOLOv8. Classifies 7 types of flexible pavement distresses from images.

## Problem

Manual pavement inspection is slow, subjective, and costly. This project automates defect detection to enable faster, consistent, and scalable road maintenance.

## Defect Classes

- Alligator cracking
- Edge cracking
- Longitudinal cracking
- Patching
- Pothole
- Rutting 
- Transverse cracking

## Dataset

- 8,000+ images (8,214 train, 783 val, 391 test, 60 inference)
- Sources: Edo State Ministry of Works + [Roboflow](https://universe.roboflow.com/baka-1ravj/road-damage-det)
- Annotation: YOLO format (bounding boxes)
- Preprocessing: 640×640 resize, normalization
- Augmentation: flip, rotation (-40° to +40°), brightness, zoom
- [Dataset Source](https://universe.roboflow.com/baka-1ravj/road-damage-det)

## Model

- Architecture: YOLOv8n (Nano)
- Input size: 640×640
- Backbone + PANet neck + anchor-free head
- Transfer learning from COCO weights

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 0.001 (best) |
| Epochs | 200 |
| Optimizer | SGD/Adam |


**Per-class recall:**
- Rutting: 0.88
- Alligator cracking: 0.81
- Pothole: 0.80
- Patching: 0.74
- Transverse cracking: 0.68
- Edge cracking: 0.61
- Longitudinal cracking: 0.58

## Installation

```bash
pip install torch ultralytics opencv-python matplotlib numpy