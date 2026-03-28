Архітектура системи Intelligent Video Hub

Компоненти
- Frontend/UI: Streamlit (Python-фреймворк для веб-інтерфейсу).
- Processing Engine: OpenCV (обробка кадрів) + Ultralytics YOLOv8 (нейромережа).
- Сховище: Локальна файлова система (папки `data/` та `models/`).

Схема взаємодії
Користувач (Браузер) -> Streamlit Server (Port 8501) -> YOLOv8 Model -> Оброблений відеопотік -> Користувач
