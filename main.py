from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import logging
#(чтобы потом разбираться, что пошло не так) потому что иначе не поймешь ничего
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Конфигурация
MODEL_PATH = "animal_classifier.keras"
CLASS_NAMES = ['cat', 'dog', 'panda']
MAX_FILE_SIZE = 2 * 1024 * 1024  # было сделано, чтобы сервак не рухнул, но он все равно рухнул
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Модель не найдена по пути {MODEL_PATH}")
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded. TF version: {tf.__version__}")
    return model
# Проверка картинки перед обработкой (чтобы не сломать сервер) (спойлер: он все равно сломался)
def validate_image(file: UploadFile, content: bytes):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Invalid file type. Only JPEG, PNG supported.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Maximum is {MAX_FILE_SIZE // (1024*1024)}MB.")
#ничего инетерного
def preprocess_image(image: Image.Image):
    try:
        image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)
        processed = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        return processed
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(400, "Invalid image format or corrupt image.")
# Загружаем модель при старте сервера (чтобы потом не тормозить, но сервак рендер не выдерживает такого нахальства, а иначе сервак засыпает и потом отваливается фронтенд)
@app.on_event("startup")
async def startup_event():
    load_model()
# Проверка здоровья сервера (чтобы знать, что он живой)(спойлер: он был долго не живой)
@app.get("/", tags=["health"])
async def health_check():
    return {
        "status": "OK",
        "endpoints": {
            "predict": {
                "method": "POST",
                "path": "/predict/",
                "input": "image/jpeg or image/png",
                "max_size": "2MB"
            }
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    validate_image(file, contents)
    mdl = load_model()
    image = Image.open(io.BytesIO(contents))
    processed_image = preprocess_image(image)
    try:
        predictions = mdl.predict(processed_image)
        out_probs = predictions[0].astype(float)
        result_class_index = int(np.argmax(out_probs))
        return {
            "class": CLASS_NAMES[result_class_index],
            "probabilities": {
                cls: float(prob)
                for cls, prob in zip(CLASS_NAMES, out_probs)
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # 0.0.0.0 - значит слушаем все интерфейсы (шпиён)