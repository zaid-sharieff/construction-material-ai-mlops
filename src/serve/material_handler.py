import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from ts.torch_handler.base_handler import BaseHandler


class MaterialHandler(BaseHandler):
    """
    TorchServe handler for Construction Material Classification
    """

    def __init__(self):
        super().__init__()
        self.initialized = False

    # -------------------------------------------------
    # INITIALIZE
    # -------------------------------------------------
    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_dir = properties.get("model_dir")
        model_path = f"{model_dir}/resnet101.pth"
        class_file = f"{model_dir}/classes.json"

        # ----------------------------
        # Load class names FIRST
        # ----------------------------
        with open(class_file, "r") as f:
            class_map = json.load(f)

        # Ensure correct index ordering
        self.class_names = [
            class_map[str(i)] for i in range(len(class_map))
        ]

        num_classes = len(self.class_names)

        # ----------------------------
        # Load model
        # ----------------------------
        self.model = models.resnet101(weights=None)
        self.model.fc = nn.Linear(
            self.model.fc.in_features, num_classes
        )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )

        self.model.to(self.device)
        self.model.eval()

        # ----------------------------
        # Image preprocessing
        # ----------------------------
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.initialized = True

    # -------------------------------------------------
    # PREPROCESS
    # -------------------------------------------------
    def preprocess(self, data):
        images = []

        for row in data:
            image_bytes = row.get("data") or row.get("body")

            if not isinstance(image_bytes, (bytes, bytearray)):
                raise ValueError("Invalid image input")

            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = self.transform(img)
            images.append(img)

        return torch.stack(images).to(self.device)

    # -------------------------------------------------
    # INFERENCE
    # -------------------------------------------------
    def inference(self, batch):
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
        return probs

    # -------------------------------------------------
    # POSTPROCESS
    # -------------------------------------------------
    def postprocess(self, inference_output):
        results = []

        for probs in inference_output:
            confidence, class_idx = torch.max(probs, dim=0)

            results.append({
                "material": self.class_names[class_idx.item()],
                "confidence": round(confidence.item(), 4)
            })

        return results
