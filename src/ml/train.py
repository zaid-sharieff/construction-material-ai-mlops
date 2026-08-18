import mlflow
from models.classification.train import train_classifier
from models.detection.train_yolo import train as train_detector

def train_all():
    mlflow.set_experiment("Construction_Material_AI")

    with mlflow.start_run(run_name="YOLO_Detection"):
        train_detector()

    with mlflow.start_run(run_name="ResNet_Classification"):
        train_classifier()

if __name__ == "__main__":
    train_all()
