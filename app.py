import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

#API_URL = "https://nine-laba.onrender.com/predict/"
API_URL = "http://127.0.0.1:8000/predict/"# Локальный сервер, потому что НЕ локальный УМЕР БЛИН
CLASS_NAMES = ['cat', 'dog', 'panda']
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2 MB (больше - уже тяжело было для онлайн сервера, комп в принципе все равно)
# красевенька
st.title("Классификация изображений 🐱🐶🐼")
st.markdown("Загрузите изображение кошки, собаки или панды")
# Информационка для пользователя (чтоб не спрашивали лишнего,надоели уже)
st.info(f"""
**Поддерживаемые форматы:** JPEG, PNG  
**Максимальный размер:** {MAX_FILE_SIZE // 1024 // 1024}MB  
**Определяемые классы:** {', '.join(CLASS_NAMES)}
""")


def preprocess_image_client(image: Image.Image) -> bytes:
    """Подготовка изображения для отправки на API"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def api_health_check():
    try:
        health_url = API_URL.replace("/predict/", "/")
        response = requests.get(health_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main():
    if not api_health_check():
        st.warning("⚠️ API сервер недоступен или отвечает с ошибкой.")

    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False
    )
    # Если файл загрузили наконец-то!
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"Файл слишком большой (максимум {MAX_FILE_SIZE // 1024 // 1024}MB)")
            return

        try:
            img = Image.open(uploaded_file)
            st.image(img, caption='Загруженное изображение', use_column_width=True)
            processed = preprocess_image_client(img)
            files = {"file": ("image.png", processed, "image/png")}
            with st.spinner("Анализируем изображение..."):
                response = requests.post(API_URL, files=files, timeout=60)
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Результат:** {result['class'].capitalize()}")
                # Рисуем график вероятностей (просто чтобы красиво и по-умному было)
                fig, ax = plt.subplots()
                bars = ax.bar(result['probabilities'].keys(), result['probabilities'].values())
                ax.bar_label(bars, fmt="%.2f")
                ax.set_ylim(0, 1)
                ax.set_ylabel("Вероятность")
                ax.set_title("Распределение вероятностей по классам")
                st.pyplot(fig)
            elif 400 <= response.status_code < 500:
                st.error(f"Ошибка клиента: {response.text}")
            else:
                st.error(f"Ошибка сервера: {response.status_code}")
        except Exception as e: # Если вообще все плохо
            st.error(f"Ошибка при обработке изображения или соединении: {str(e)}")


if __name__ == "__main__":
    main()