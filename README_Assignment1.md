# Broken Tile Detection - Assignment 1

## Overview

This project implements a computer vision solution to detect broken roof tiles using a YOLOv8 object detection model. The solution is part of a quality control system to help construction teams identify potential issues early using images captured on-site.

---

## 📁 Project Structure

```
tile-detector-api/
├── main.py               # FastAPI app for tile detection
├── Dockerfile            # For containerizing the app
├── requirements.txt      # Dependencies
├── best.pt               # Trained YOLOv8 model weights
└── tile_dataset/         # Training dataset (images + labels + data.yaml)
```

---

## 🚀 Features

- Trained YOLOv8 model to detect `broken_tile`
- Inference API built with FastAPI
- Deployed on Google Cloud Run
- Dataset annotated via CVAT and exported in YOLO format
- mAP@0.5 ≈ `0.215` | Recall ≈ `0.29`

---

## 🧠 Model Training

- Framework: Ultralytics YOLOv8
- Environment: Google Vertex AI Workbench (Jupyter + GPU)
- Data: 700+ labeled images (`broken`, `not broken`)
- Preprocessing: Balanced dataset, split into train/val

Training Command:
```bash
model.train(data="tile_dataset/data.yaml", epochs=50, imgsz=640)
```

---

## ☁️ Deployment

- Containerized with Docker
- Hosted on Github
- Accepts image uploads and returns detection results (JSON with bounding boxes)

Deployment steps:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/tile-detector
gcloud run deploy tile-detector --image gcr.io/PROJECT_ID/tile-detector --platform managed
```

---

## 🔧 API Usage

**POST** `/predict`

Request (multipart/form-data):
```bash
curl -X POST -F image=@example.jpg http://<your-url>/predict
```

Response:
```json
{
  "results": [
    {
      "class": "broken_tile",
      "confidence": 0.91,
      "bbox": [x_min, y_min, x_max, y_max]
    }
  ]
}
```
