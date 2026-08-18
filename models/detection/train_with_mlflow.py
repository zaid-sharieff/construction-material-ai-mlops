import os
import subprocess
import pandas as pd
import mlflow
import mlflow.pytorch

# ---------------- CONFIG ----------------
MLFLOW_EXPERIMENT = "Material_Detection"
BASE_MODEL = "yolov8n.pt"
EPOCHS = 15
IMG_SIZE = 640
BATCH = 4

DATASETS = {
    "brick": "data/raw/Detection/brick/data.yaml",
    "concrete": "data/raw/Detection/concrete/data.yaml",
    "steel": "data/raw/Detection/steel/data.yaml",
    "wood": "data/raw/Detection/wood/data.yaml",
    "sand": "data/raw/Detection/sand/data.yaml",
    "soil": "data/raw/Detection/soil/data.yaml"
}

# ---------------------------------------

def train_detector(material, data_yaml):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"yolov8_{material}"):

        # ---------------- LOG PARAMS ----------------
        mlflow.log_params({
            "material": material,
            "model": BASE_MODEL,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "batch_size": BATCH,
            "dataset": data_yaml
        })

        # ---------------- YOLO TRAIN ----------------
        cmd = [
            "yolo", "task=detect", "mode=train",
            f"model={BASE_MODEL}",
            f"data={data_yaml}",
            f"epochs={EPOCHS}",
            f"imgsz={IMG_SIZE}",
            f"batch={BATCH}",
            "save=True"
        ]

        subprocess.run(cmd, check=True)

        # ---------------- FIND RUN DIR ----------------
        runs_dir = "runs/detect"
        latest_run = sorted(os.listdir(runs_dir))[-1]
        run_path = os.path.join(runs_dir, latest_run)

        # ---------------- LOG METRICS ----------------
        results_csv = os.path.join(run_path, "results.csv")
        df = pd.read_csv(results_csv)

        last_row = df.iloc[-1]

        mlflow.log_metrics({
            "precision": last_row["metrics/precision(B)"],
            "recall": last_row["metrics/recall(B)"],
            "mAP50": last_row["metrics/mAP50(B)"],
            "mAP50-95": last_row["metrics/mAP50-95(B)"],
            "box_loss": last_row["train/box_loss"],
            "cls_loss": last_row["train/cls_loss"],
            "dfl_loss": last_row["train/dfl_loss"]
        })

        # ---------------- LOG ARTIFACTS ----------------
        mlflow.log_artifacts(run_path)

        # ---------------- REGISTER MODEL ----------------
        best_model_path = os.path.join(run_path, "weights", "best.pt")

        mlflow.pytorch.log_model(
            pytorch_model=None,
            artifact_path="model",
            registered_model_name=f"Material_Detector_{material}",
            extra_files=[best_model_path]
        )

        print(f"✅ {material.upper()} detector logged & registered")


if __name__ == "__main__":
    for mat, yaml in DATASETS.items():
        train_detector(mat, yaml)
