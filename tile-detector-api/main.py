from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

model = YOLO("best.pt")  # Path to your trained model

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model.predict(image, conf=0.25)
    labels = results[0].names
    detections = results[0].boxes.cls.tolist()
    
    is_broken = any([int(cls_id) == 0 for cls_id in detections])
    return {
        "broken_detected": is_broken,
        "detections": len(detections),
    }