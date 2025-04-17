from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import logging

def load_model():
    global model
    if model is None:
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"Model loaded. TF version: {tf.__version__}")

            # Тестовый прогон для проверки
            test_input = np.random.rand(1, 224, 224, 3)
            model.predict(test_input)
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise
    return model
def validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, "Invalid file type")

    if file.size > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE} bytes")

model = load_model()
# Настройка логгера
logger = logging.getLogger("uvicorn")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Конфигурация
MODEL_PATH = "model2.keras"
CLASS_NAMES = ['cat', 'dog', 'panda']
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png"]

#model = None

@app.api_route("/", methods=["GET", "HEAD"])
async def health_check():
    return {
        "status": "OK",
        "endpoints": {
            "predict": {
                "method": "POST",
                "path": "/predict",
                "input": "image/jpeg or image/png",
                "max_size": "2MB"
            }
        }
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Валидация файла
        validate_image(file)

        # Загрузка модели
        model = load_model()

        # Чтение и обработка изображения
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        processed_image = preprocess_image(image)

        # Предсказание
        predictions = model.predict(processed_image)

        return {
            "class": CLASS_NAMES[np.argmax(predictions)],
            "probabilities": {
                cls: float(prob)
                for cls, prob in zip(CLASS_NAMES, predictions[0])
            }
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(500, f"Processing error: {str(e)}")


def preprocess_image(image: Image.Image):
    try:
        image = image.resize((224, 224))
        image_array = np.array(image)
        return tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(400, "Invalid image format")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)