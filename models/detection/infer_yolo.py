# models/detection/infer_yolo.py

from ultralytics import YOLO

MODEL_PATHS = {
    "brick": "models/detection/brick.pt",
    "concrete": "models/detection/concrete.pt",
    "steel": "models/detection/steel.pt",
    "wood": "models/detection/wood.pt",
    "sand": "models/detection/sand.pt",
    "soil": "models/detection/soil.pt",
}

class MaterialDetector:
    def __init__(self, material: str):
        if material not in MODEL_PATHS:
            raise ValueError(f"No detection model available for {material}")

        self.material = material
        self.model = YOLO(MODEL_PATHS[material])

    def detect(self, image_path: str):
        results = self.model(image_path)[0]

        detections = []
        for box in results.boxes:
            detections.append({
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class": self.material
            })

        return detections
