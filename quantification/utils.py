# quantification/utils.py

import cv2

CONTINUOUS_MATERIALS = {"sand", "soil"}

def draw_bounding_boxes(image_path, detections, output_path="output.jpg"):
    """
    Draw bounding boxes and labels on the image.
    """
    image = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        label = f'{det["class"]} ({det["confidence"]:.2f})'

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)

        # Put label text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    cv2.imwrite(output_path, image)
    return output_path

def is_continuous_material(material: str) -> bool:
    """
    Check if the material is continuous.
    """
    return material.lower() in CONTINUOUS_MATERIALS