import torch
from torchvision import models, transforms
from PIL import Image

CLASS_NAMES = ["brick", "concrete", "sand", "soil", "steel", "wood"]

class MaterialClassifier:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = models.resnet101(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img).argmax(dim=1).item()

        return CLASS_NAMES[pred]
