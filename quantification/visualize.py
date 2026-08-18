# quantification/visualize.py

import os
import cv2

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_boxes_with_labels(image_path, detections, material, count, quantity):
    img = cv2.imread(image_path)

    if img is None:
        return None

    # ---------------------------
    # Case 1: Discrete materials
    # ---------------------------
    if detections:
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det.get("confidence", 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{material} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ---------------------------
    # Case 2: Sand / Soil (no boxes)
    # ---------------------------
    else:
        overlay = img.copy()
    
        # Dark background rectangle for contrast
        cv2.rectangle(
            overlay,
            (20, 20),
            (img.shape[1] - 20, 120),
            (0, 0, 0),
            -1
        )
    
        alpha = 0.6
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
        text = material.upper()
    
        if quantity:
            text += (
                f" | Vol: {quantity['estimated_volume_m3']} m3"
                f" | Mass: {quantity['estimated_mass_kg']} kg"
            )
    
        cv2.putText(
            img,
            text,
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),  # Green text
            2
        )

    # ---------------------------
    # Save SINGLE annotated image
    # ---------------------------
    output_path = os.path.join(OUTPUT_DIR, "annotated.jpg")
    cv2.imwrite(output_path, img)

    return output_path
