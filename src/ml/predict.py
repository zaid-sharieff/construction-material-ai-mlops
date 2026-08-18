# src/ml/predict.py

import os

from models.classification.infer import MaterialClassifier
from models.detection.infer_yolo import MaterialDetector
from quantification.counter import count_objects
from quantification.volume_estimator import estimate_volume_from_bbox, estimate_mass
from quantification.visualize import draw_boxes_with_labels

# --- DB imports ---
from src.database.db import SessionLocal
from src.database.crud import (
    create_image,
    add_detection,
    add_quantification
)

# ---------------------------
# Model initialization
# ---------------------------
classifier = MaterialClassifier("models/classification/resnet101.pth")

DETECTION_MATERIALS = ["brick", "wood", "steel", "concrete"]
CONTINUOUS_MATERIALS = ["sand", "soil"]


# ---------------------------
# Quantification helpers
# ---------------------------
def estimate_sand_soil(detections, material):
    total_volume = 0.0

    for det in detections:
        total_volume += estimate_volume_from_bbox(det["bbox"])

    total_mass = estimate_mass(total_volume, material)

    return {
        "estimated_volume_m3": round(total_volume, 3),
        "estimated_mass_kg": round(total_mass, 3)
    }


# ---------------------------
# MAIN PREDICT FUNCTION
# ---------------------------
def predict(image_path):
    filename = os.path.basename(image_path)
    db = SessionLocal()

    # ---------------------------
    # Step 1: Classification
    # ---------------------------
    material = classifier.predict(image_path)

    image_row = create_image(db, filename=filename)
    image_id = image_row.id

    # ---------------------------
    # Step 2: Detection (ALWAYS)
    # ---------------------------
    detector = MaterialDetector(material)
    detections = detector.detect(image_path)

    for det in detections:
        add_detection(
            db=db,
            image_id=image_id,
            material=material,
            conf=det["confidence"],
            bbox=det["bbox"]
        )

    count = None
    quantity = None

    # ---------------------------
    # Step 3: Quantification
    # ---------------------------
    if material in CONTINUOUS_MATERIALS:
        quantity = estimate_sand_soil(detections, material)

        add_quantification(
            db=db,
            image_id=image_id,
            material=material,
            count=None,
            volume_m3=quantity["estimated_volume_m3"],
            mass_kg=quantity["estimated_mass_kg"]
        )
    else:
        count = count_objects(detections)

        add_quantification(
            db=db,
            image_id=image_id,
            material=material,
            count=count,
            volume_m3=None,
            mass_kg=None
        )

    # ---------------------------
    # Step 4: Visualization (ALWAYS)
    # ---------------------------
    annotated_image = draw_boxes_with_labels(
        image_path=image_path,
        detections=detections,
        material=material,
        count=count,
        quantity=quantity
    )

    db.close()

    return {
        "material": material,
        "count": count,
        "quantity": quantity,
        "detections": detections,
        "annotated_image": annotated_image
    }
