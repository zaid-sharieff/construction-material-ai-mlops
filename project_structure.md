CONSTRUCTION_MATERIAL_AI/
│
├── .dvc/                         # DVC metadata
├── .streamlit/                   # Streamlit theming
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   ├── bricks/
│   │   ├── sand/
│   │   ├── soil/
│   │   ├── steel/
│   │   └── wood/
│   │
│   ├── processed/
│   ├── annotations/              # YOLO labels / masks
│   ├── material_data.db          # SQLite DB
│   ├── raw.dvc
│   └── .gitignore
│
├── frontend/                     # Streamlit UI (same role as POLYPS)
│   ├── app.py                    # main UI
│   ├── pages/
│   │   ├── detect_quantify.py
│   │   ├── dashboard.py
│   │   ├── chatbot.py
│   └── assets/
│       ├── lottie/
│       └── icons/
│
├── logs/                         # TorchServe + app logs
│
├── mlruns/                       # MLflow experiments
│
├── model_store/                  # TorchServe .mar files
│   ├── classifier.mar
│   ├── detector.mar
│
├── models/
│   ├── classification/
│   │   ├── resnet101.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── infer.py
│   │
│   ├── detection/
│   │   ├── yolo/
│   │   ├── train_yolo.py
│   │   └── infer_yolo.py
│   │
│   ├── segmentation/             # optional (U-Net / YOLOv8-Seg)
│   │   └── unet.py
│
├── quantification/               # ⭐ CORE NOVELTY
│   ├── counter.py                # brick / rod counting
│   ├── area_calc.py              # pixel area
│   ├── volume_estimator.py       # volume & mass
│   ├── density_constants.py      # IS standard densities
│   └── utils.py
│
├── reports/                      # Evidently drift reports
│   └── drift_report.html
│
├── src/
│   ├── api/                      # FastAPI (same as POLYPS)
│   │   ├── main.py
│   │   └── routes.py
│   │
│   ├── database/
│   │   ├── db.py
│   │   ├── models.py
│   │   └── crud.py
│   │
│   ├── drift/
│   │   └── drift_report.py
│   │
│   ├── ml/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   │
│   ├── serve/
│   │   └── gi_handler.py         # TorchServe handler
│   │
│   └── chatbot/                  # ⭐ NEW (from EL PDF)
│       ├── knowledge_base.json
│       ├── material_info.py
│       └── chatbot_engine.py
│
├── configs/
│   ├── model.yaml
│   ├── quantification.yaml
│   └── chatbot.yaml
│
├── dvc.yaml
├── dvc.lock
├── .dvcignore
├── .gitignore
├── requirements.txt
├── config.properties
├── README.md
└── project_structure.md

yolo task=detect mode=train model=yolov8n.pt data=data/raw/Detection/brick/data.yaml epochs=15 imgsz=640 batch=4

yolo task=detect mode=train model=yolov8n.pt data=data/raw/Detection/concrete/data.yaml epochs=15 imgsz=640 batch=4

yolo task=detect mode=train model=yolov8n.pt data=data/raw/Detection/steel/data.yaml epochs=15 imgsz=640 batch=4 workers=2 mosaic=0 hsv_h=0 hsv_s=0 hsv_v=0

yolo task=detect mode=train model=yolov8n.pt data=data/raw/Detection/wood/data.yaml epochs=15 imgsz=640 batch=4 workers=2 mosaic=0 hsv_h=0 hsv_s=0 hsv_v=0

yolo task=detect mode=train model=yolov8n.pt data=data/raw/Detection/sand/data.yaml epochs=15 imgsz=640 batch=4

yolo task=detect mode=train model=yolov8n.pt data=data/raw/Detection/soil/data.yaml epochs=15 imgsz=640 batch=4

README.md
requirements.txt dvc.yaml configs/ frontend/ quantification/ src/ project_structure.md quantification_test.py