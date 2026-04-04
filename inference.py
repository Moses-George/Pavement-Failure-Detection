import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Configuration ---
# Define your classes (MUST match the order used during training)
CLASS_NAMES = ['crack', 'pothole'] # Example classes
IMG_SIZE = 640 # Must match the image size used during training
CONFIDENCE_THRESHOLD = 0.10 # Adjust as needed
IOU_THRESHOLD = 0.45 # Adjust as needed

# Path to your best trained model weights
# This assumes you copied 'best.pt' from your training run results to the 'models' folder
# TRAINED_MODEL_PATH = 'models/best.pt'
TRAINED_MODEL_PATH = 'runs/detect/pavement_detection_model3/weights/best.pt'

# Paths to new images for inference
# You can add more paths, or specify a directory:
NEW_IMAGE_PATHS = [
    'pavement_dataset/test/test-crack-01.jpg',
    #  'pavement_dataset/test/test-crack-02.jpeg',
    'pavement_dataset/test/test-crack-03.jpg',
        'pavement_dataset/test/test-crack-04.jpg'
       # Example using a validation image
    # 'path/to/another/new_image.jpg',
    # 'path/to/a/directory_of_images' # For batch processing a folder
]

# Ensure the trained model exists
if not os.path.exists(TRAINED_MODEL_PATH):
    print(f"Error: Trained model not found at {TRAINED_MODEL_PATH}.")
    print("Please ensure you have run 'train_model.py' and copied 'best.pt' to the 'models/' folder, or update TRAINED_MODEL_PATH.")
    exit()

# Load the trained YOLOv8 model
print(f"Loading trained model from: {TRAINED_MODEL_PATH}")
model = YOLO(TRAINED_MODEL_PATH)
print("Model loaded successfully.")

print("\n--- Running Inference on New Images ---")
for img_path in NEW_IMAGE_PATHS:
    if not os.path.exists(img_path):
        print(f"Warning: Image not found at {img_path}. Skipping.")
        continue

    print(f"Processing image: {img_path}")
    # Perform inference
    results = model.predict(
        source=img_path,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,  # Saves the annotated image to 'runs/detect/predict' folder
        show=False  # Don't show pop-up window if running in script without GUI
    )

    # --- Process and Display Results for Each Image ---
    for r in results: # 'results' will be a list of Results objects
        img_original = r.orig_img # The original image as a NumPy array (BGR format)
        boxes = r.boxes.xyxy.cpu().numpy() # Bounding box coordinates (x1, y1, x2, y2)
        scores = r.boxes.conf.cpu().numpy() # Confidence scores
        class_ids = r.boxes.cls.cpu().numpy().astype(int) # Predicted class IDs

        img_annotated = img_original.copy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            score = scores[i]
            class_id = class_ids[i]

            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown ID: {class_id}"

            # You can define colors for each class if you wish
            if class_name == 'crack':
                color = (0, 0, 255) # Red for cracks (BGR)
            elif class_name == 'pothole':
                color = (0, 255, 255) # Yellow for potholes
            elif class_name == 'patch':
                color = (255, 0, 0) # Blue for patches
            else:
                color = (0, 255, 0) # Default Green

            label = f"{class_name}: {score:.2f}"

            cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 2)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(img_annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) # White text

        # Display the annotated image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Pavement Failures on: {os.path.basename(img_path)}")
        plt.axis('off')
        plt.show()

print("\n--- Inference process complete ---")