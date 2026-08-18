# models/classification/train.py

import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import mlflow
import mlflow.pytorch
import psutil

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "data/raw/Classification"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4

MODEL_PATH = "models/classification/resnet101.pth"
REPORT_DIR = "reports/classification"
JSON_RESULTS_PATH = os.path.join(REPORT_DIR, "tabular_results.json")

MLFLOW_EXPERIMENT = "Material_Classification"
REGISTERED_MODEL_NAME = "Material_Classifier"

os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------


def dump_tabular_results(history, save_path):
    """Dump training and evaluation metrics into a JSON file."""
    with open(save_path, "w") as f:
        json.dump(history, f, indent=4)


def log_torch_gpu_metrics(step=None):
    if torch.cuda.is_available():
        mlflow.log_metric(
            "gpu_memory_allocated_mb",
            torch.cuda.memory_allocated() / 1024**2,
            step=step
        )
        mlflow.log_metric(
            "gpu_memory_reserved_mb",
            torch.cuda.memory_reserved() / 1024**2,
            step=step
        )

# -----------------------------
# TRAINING FUNCTION
# -----------------------------


def train_classifier():

    # -----------------------------
    # MLflow Setup
    # -----------------------------
    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Training on device: {device}")

    # -----------------------------
    # JSON History Store
    # -----------------------------
    history = {
        "training_loss": [],
        "training_accuracy": [],
        "val_accuracy": [],
        "final_metrics": {}
    }

    with mlflow.start_run(run_name="resnet101_training"):

        # -----------------------------
        # Log Parameters
        # -----------------------------
        mlflow.log_params({
            "model": "ResNet101",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "optimizer": "Adam",
            "input_size": "224x224",
            "dataset_path": DATA_DIR
        })

        # -----------------------------
        # Transforms
        # -----------------------------
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        # -----------------------------
        # Dataset
        # -----------------------------
        dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
        class_names = dataset.classes
        print(f"📦 Classes: {class_names}")

        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("class_names", class_names)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # -----------------------------
        # Model
        # -----------------------------
        model = models.resnet101(weights="IMAGENET1K_V1")

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        for param in model.fc.parameters():
            param.requires_grad = True

        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR
        )

        # -----------------------------
        # Training Loop
        # -----------------------------
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct_train, total_train = 0, 0

            train_bar = tqdm(
                train_loader,
                desc=f"Epoch [{epoch+1}/{EPOCHS}]",
                leave=False
            )

            for imgs, labels in train_bar:
                imgs, labels = imgs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

                train_bar.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)

            # -----------------------------
            # Validation
            # -----------------------------
            model.eval()
            correct_val, total_val = 0, 0

            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    correct_val += (preds == labels).sum().item()
                    total_val += labels.size(0)

            val_acc = correct_val / total_val
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            log_torch_gpu_metrics(step=epoch)

            # -----------------------------
            # Store Metrics for JSON
            # -----------------------------
            history["training_loss"].append(round(avg_loss, 4))
            history["training_accuracy"].append(round(train_acc, 4))
            history["val_accuracy"].append(round(val_acc, 4))

            print(
                f"✅ Epoch {epoch+1}/{EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc*100:.2f}% | "
                f"Val Acc: {val_acc*100:.2f}%"
            )

        # -----------------------------
        # Final Evaluation
        # -----------------------------
        print("\n📊 Running final evaluation on validation set...")

        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        history["final_metrics"] = {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

        # -----------------------------
        # Dump JSON Results
        # -----------------------------
        dump_tabular_results(history, JSON_RESULTS_PATH)
        print(f"📄 Tabular results saved to: {JSON_RESULTS_PATH}")

        # -----------------------------
        # Save Model
        # -----------------------------
        torch.save(model.state_dict(), MODEL_PATH)

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print(f"\n💾 Model saved & registered as: {REGISTERED_MODEL_NAME}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    train_classifier()
