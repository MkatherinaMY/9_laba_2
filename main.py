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
'''
# Загрузка модели
model_path = os.path.join(os.path.dirname(__file__), "model.h5")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели не найден: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
    print("Модель успешно загружена!")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    raise
class_names = ['cat', 'dog', 'panda']

'''
# Предобработка изображения
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

'''
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Чтение и декодирование изображения
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Предобработка
    processed_image = preprocess_image(image)

    # Предсказание
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    return {
        "class": predicted_class,
        "probabilities": {class_names[i]: float(predictions[0][i]) for i in range(3)}
    }
'''

model = None
class_names = ['cat', 'dog', 'panda']


def load_model():
    global model
    if model is None:
        model_path = "model.keras"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        # Для отладки
        print("Содержимое папки модели:", os.listdir(model_path))

        model = tf.keras.models.load_model(model_path)
        print("Модель успешно загружена!")
    return model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global class_names
    # Загрузка модели, если она еще не загружена
    model = load_model()

    # Чтение и декодирование изображения
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Предобработка
    processed_image = preprocess_image(image)

    # Предсказание
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    return {
        "class": predicted_class,
        "probabilities": {class_names[i]: float(predictions[0][i]) for i in range(3)}
    }

PORT = int(os.getenv("PORT", 8000))  # Берет порт из переменной окружения PORT или использует 8000 по умолчанию
print(f"Using PORT: {PORT}")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)