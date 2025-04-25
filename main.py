from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

model = YOLO("best.pt")  # Make sure this model file is present in the root folder

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        results = model.predict(image, conf=0.1)

        if not results or not results[0].boxes:
            return JSONResponse(content={
                "broken_detected": False,
                "detections": 0
            })

        detections = results[0].boxes.cls.tolist()
        is_broken = any(int(cls_id) == 0 for cls_id in detections)

        return {
            "broken_detected": is_broken,
            "detections": len(detections)
        }

    except UnidentifiedImageError:
        return JSONResponse(
            content={"error": "Could not process image."},
            status_code=400
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Unexpected server error: {str(e)}"},
            status_code=500
        )