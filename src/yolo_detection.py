# ============================================================
# YOLO OBJECT DETECTION MODULE
# Detects objects relevant to road danger using YOLOv8.
# ============================================================

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import YOLO_MODEL_PATH


# ============================================================
# 1. Load YOLO model
# ============================================================

def load_yolo_model():
    """
    Loads YOLOv8 nano model.

    If models/yolov8n.pt exists, it uses the local file.
    Otherwise, ultralytics will try to download yolov8n.pt automatically.
    """

    if YOLO_MODEL_PATH.exists():
        model_path = str(YOLO_MODEL_PATH)
    else:
        model_path = "yolov8n.pt"

    model = YOLO(model_path)
    return model


# ============================================================
# 2. Run YOLO detection
# ============================================================

def run_yolo_detection(image_path, model, confidence_threshold=0.25):
    """
    Runs YOLO on an image and returns detections + road-relevant features.
    """

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    image_area = height * width

    results = model(str(image_path), conf=confidence_threshold, verbose=False)

    detections = []

    num_people = 0
    num_vehicles = 0
    num_bikes = 0
    largest_vehicle_area_ratio = 0.0

    vehicle_classes = {"car", "bus", "truck", "motorbike", "motorcycle"}
    bike_classes = {"bicycle"}
    person_classes = {"person"}

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = str(model.names[class_id])

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_ratio = box_area / image_area if image_area > 0 else 0.0

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "area_ratio": float(area_ratio),
            })

            class_name_lower = class_name.lower()

            if class_name_lower in person_classes:
                num_people += 1

            if class_name_lower in vehicle_classes:
                num_vehicles += 1
                largest_vehicle_area_ratio = max(
                    largest_vehicle_area_ratio,
                    area_ratio,
                )

            if class_name_lower in bike_classes:
                num_bikes += 1

    features = {
        "num_people": int(num_people),
        "num_vehicles": int(num_vehicles),
        "num_bikes": int(num_bikes),
        "largest_vehicle_area_ratio": float(largest_vehicle_area_ratio),
    }

    return {
        "detections": detections,
        "yolo_features": features,
    }


# ============================================================
# 3. Draw YOLO detections
# ============================================================

def draw_yolo_detections(image, detections):
    """
    Draws YOLO bounding boxes on an image.

    image can be:
    - BGR image from cv2
    - RGB numpy image

    Returns an RGB numpy image.
    """

    if image is None:
        raise ValueError("Input image is None.")

    output = image.copy()

    # If image likely comes from cv2, convert BGR to RGB later if needed.
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Convert BGR to RGB for matplotlib display.
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return output_rgb