from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import io
import uvicorn
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS settings (allow all origins for now — adjust for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model
try:
    model = YOLO("best.pt")  # Make sure this path is correct and file is bundled
    logger.info("YOLO model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load YOLO model.")
    raise RuntimeError("Model loading failed.") from e

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received file: {file.filename}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        logger.exception("Uploaded file is not a valid image.")
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        results = model.predict(image, conf=0.25)
        detections = results[0].boxes.cls.tolist() if results[0].boxes else []

        # Class 0 is assumed to be 'broken' class
        is_broken = any(int(cls_id) == 0 for cls_id in detections)

        logger.info(f"Detections: {detections}")
        return {
            "broken_detected": is_broken,
            "detections": len(detections)
        }

    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Error during prediction.")

# Optional: run with uvicorn if using `python main.py` directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)