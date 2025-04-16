import streamlit as st
import requests
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Настройки
API_URL = "https://nine-laba.onrender.com/predict/"
CLASS_NAMES = ['cat', 'dog', 'panda']

# Интерфейс
st.title("Классификация изображений 🐱🐶🐼")
uploaded_file = st.file_uploader("Загрузите изображение", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Превью изображения
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', width=300)

    # Отправка на API
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        result = response.json()

        # Отображение результатов
        st.success(f"**Результат:** {result['class']}")

        # Визуализация вероятностей
        fig, ax = plt.subplots()
        ax.bar(result['probabilities'].keys(), result['probabilities'].values())
        ax.set_title("Вероятности классов")
        st.pyplot(fig)
    else:
        st.error("Ошибка API")