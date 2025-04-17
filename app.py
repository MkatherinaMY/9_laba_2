import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Конфигурация
API_URL = "https://nine-laba.onrender.com/predict/"
CLASS_NAMES = ['cat', 'dog', 'panda']
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

st.title("Классификация изображений 🐱🐶🐼")
st.markdown("Загрузите изображение кошки, собаки или панды")


def main():
    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            # Проверка размера файла
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"Файл слишком большой (максимум {MAX_FILE_SIZE // 1024 // 1024}MB)")
                return

            # Превью изображения
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', use_column_width=True)

            # Отправка на API
            with st.spinner("Анализируем изображение..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=300  # Таймаут 10 секунд
                )

            # Обработка ответа
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Результат:** {result['class']}")

                # Визуализация
                fig, ax = plt.subplots()
                ax.bar(result['probabilities'].keys(), result['probabilities'].values())
                ax.set_ylabel("Вероятность")
                ax.set_title("Распределение вероятностей")
                st.pyplot(fig)

            else:
                st.error(f"Ошибка API: {response.status_code} - {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка соединения: {str(e)}")
        except Exception as e:
            st.error(f"Ошибка обработки: {str(e)}")


if __name__ == "__main__":
    main()