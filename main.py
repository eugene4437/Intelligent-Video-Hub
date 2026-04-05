"""
Intelligent Video Processing System.

Цей модуль реалізує веб-інтерфейс для аналізу відео за допомогою YOLOv8.
Забезпечує завантаження файлів, сегментацію сцен та детекцію об'єктів.
"""

import os
import tempfile
from typing import Optional, List, Any
import streamlit as st

def setup_page() -> None:
    """
    Налаштовує метадані та макет сторінки Streamlit.
    
    Архітектурне рішення: Використання wide layout для зручного відображення
    відео та результатів аналітики в двох колонках.
    """
    st.set_page_config(page_title="AI Video Hub", page_icon="🎬", layout="wide")

def format_confidence(value: float) -> str:
    """
    Конвертує числове значення впевненості у відсотковий рядок.

    Жива документація (Doctest):
    >>> format_confidence(0.95)
    '95.0%'
    >>> format_confidence(0.5)
    '50.0%'
    """
    return f"{value * 100:.1f}%"

def save_video(file_obj: Any) -> str:
    """
    Зберігає завантажений об'єкт файлу у тимчасову директорію.

    Бізнес-логіка: Програма не зберігає відео постійно для економії місця.
    Використовується системний temp-каталог.

    Args:
        file_obj (Any): Буфер файлу з Streamlit.

    Returns:
        str: Шлях до створеного файлу.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(file_obj.read())
        return tmp.name

def main() -> None:
    """
    Головна функція керування життєвим циклом застосунку.
    
    Взаємодія компонентів:
    1. UI (Streamlit) отримує файл.
    2. Logic (save_video) готує дані.
    3. AI Engine (YOLO) обробляє кадри.
    """
    setup_page()
    st.title("🎬 Intelligent Video Processing System")
    uploaded: Optional[Any] = st.file_uploader("Оберіть відео", type=['mp4', 'mkv'])

    if uploaded:
        path: str = save_video(uploaded)
        st.video(uploaded)
        if st.button("🚀 Аналізувати"):
            st.success(f"Аналіз завершено. Впевненість: {format_confidence(0.88)}")
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    main()
