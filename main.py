from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
class_names = ['cat', 'dog', 'panda']


def load_model():
    global model
    if model is None:
        model_path = "model.h5"

        # Отладочная информация
        print("Текущая рабочая директория:", os.getcwd())
        print("Содержимое директории:", os.listdir())

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        model = tf.keras.models.load_model(model_path)
        print("Модель успешно загружена!")
    return model


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)


@app.get("/")
async def root():
    return {
        "message": "Сервер работает",
        "endpoints": {
            "predict": "/predict (POST)"
        }
    }


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    model = load_model()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    return {
        "class": class_names[np.argmax(predictions)],
        "probabilities": {
            class_names[i]: float(predictions[0][i])
            for i in range(3)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))