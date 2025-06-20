import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch

print('CUDA available:', torch.cuda.is_available())
print("GPU DEVICE:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No Device" )

# --- Configuration ---
# Path to your dataset
DATASET_PATH = 'pavement_dataset' # Make sure this path is correct
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, 'images', 'train')
TRAIN_LABEL_DIR = os.path.join(DATASET_PATH, 'labels', 'train')
VAL_IMG_DIR = os.path.join(DATASET_PATH, 'images', 'val')
VAL_LABEL_DIR = os.path.join(DATASET_PATH, 'labels', 'val')

# Define your classes (e.g., crack, pothole, patch)
# The order here must match the class_id in your annotation files
CLASS_NAMES = ['crack', 'pothole'] # Example classes
NUM_CLASSES = len(CLASS_NAMES)

# Model configuration
MODEL_TYPE = 'yolov8n.pt' # 'yolov8n.pt' (nano), 'yolov8s.pt' (small), 'yolov8m.pt' (medium), 'yolov8l.pt' (large)
EPOCHS = 100 # Number of training epochs (adjust as needed)
BATCH_SIZE = 32 # Batch size for training (adjust based on GPU memory)
IMG_SIZE = 640 # Image size for training and inference

# Output directory for trained models and results
OUTPUT_DIR = 'runs/detect'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Create a YAML configuration file for YOLOv8 ---
# This file tells YOLOv8 where your data is and what your classes are.
data_yaml_content = f"""
path: {os.path.abspath(DATASET_PATH)}
train: images/train
val: images/val
nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""

data_yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

print(f"YOLOv8 data.yaml created at: {data_yaml_path}")

# --- 2. Initialize and Train the YOLOv8 Model ---
print(f"\n--- Initializing YOLOv8 model: {MODEL_TYPE} ---")
model = YOLO(MODEL_TYPE) # Load a pre-trained YOLOv8n model

print(f"\n--- Starting model training for {EPOCHS} epochs ---")
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name='pavement_detection_model', # Name for the training run
    # pin_memory = True
    # Optional: Add more arguments for fine-tuning
    # optimizer='Adam',
    # lr0=0.001,
    # save_period=10, # Save model every 10 epochs
    # cache=False # Cache images for faster training
)

print("\n--- Training complete! ---")
print(f"Results saved to: {model.trainer.save_dir}")

# --- 3. Evaluate the Trained Model (Optional but Recommended) ---
print("\n--- Evaluating the trained model ---")
# The best model is usually saved as 'weights/best.pt' in the run directory
best_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
if os.path.exists(best_model_path):
    print(f"Loading best model from: {best_model_path}")
    trained_model = YOLO(best_model_path)
    metrics = trained_model.val(data=data_yaml_path, imgsz=IMG_SIZE)
    print("\nValidation Metrics:")
    print(metrics) # You can access metrics like metrics.box.map
else:
    print(f"Best model not found at {best_model_path}. Evaluation skipped.")

# --- 4. Make Predictions on New Images ---
print("\n--- Making predictions on example images ---")

# Choose a few example images from your validation set or new images
# You can modify this list to include any images you want to test
example_images_to_test = [
    os.path.join(VAL_IMG_DIR, os.listdir(VAL_IMG_DIR)[0]), # First image in val set
    os.path.join(VAL_IMG_DIR, os.listdir(VAL_IMG_DIR)[1]) # Second image in val set
]

# Or provide paths to new images not in your dataset
# example_images_to_test = ['path/to/new_image1.jpg', 'path/to/new_image2.jpg']

if not os.path.exists(best_model_path):
    print("Cannot perform inference as best model was not found after training.")
else:
    for img_path in example_images_to_test:
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}. Skipping.")
            continue

        print(f"\nPredicting on: {img_path}")
        # Run inference with the trained model
        # You can adjust 'conf' (confidence threshold) and 'iou' (NMS IoU threshold)
        inference_results = trained_model(img_path, imgsz=IMG_SIZE, conf=0.25, iou=0.45)

        # Process results
        for r in inference_results:
            img = r.orig_img
            boxes = r.boxes.xyxy.cpu().numpy() # Bounding box coordinates (x1, y1, x2, y2)
            scores = r.boxes.conf.cpu().numpy() # Confidence scores
            class_ids = r.boxes.cls.cpu().numpy().astype(int) # Class IDs

            # Draw bounding boxes and labels
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                score = scores[i]
                class_id = class_ids[i]
                class_name = CLASS_NAMES[class_id]

                color = (0, 255, 0) # Green for bounding boxes
                label = f"{class_name}: {score:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the result
            plt.figure(figsize=(10, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(f"Detection on: {os.path.basename(img_path)}")
            plt.axis('off')
            plt.show()

print("\n--- Script finished ---")