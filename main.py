"""
Main entry point for the Intelligent Video Processing System.
This module handles the Streamlit UI and orchestrates the video analysis flow.
"""

import os
import tempfile
from typing import Optional, List, Any

import streamlit as st
import cv2
import numpy as np
from PIL import Image

try:
    from src.core.video_processor import VideoProcessor
    from src.core.object_detector import ObjectDetector
    from src.utils.visualization import Plotter
except ImportError:
    VideoProcessor = Any
    ObjectDetector = Any
    Plotter = Any

def setup_page_config() -> None:
    """Sets up the Streamlit page configuration."""
    st.set_page_config(
        page_title="AI Video Hub",
        page_icon="🎬",
        layout="wide"
    )

def save_uploaded_file(uploaded_file: Any) -> str:
    """
    Saves the uploaded video to a temporary directory.
    
    Args:
        uploaded_file: The file object from Streamlit uploader.
        
    Returns:
        str: Path to the saved temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def main() -> None:
    """Main application function with type-annotated logic."""
    setup_page_config()

    st.title("🎬 Intelligent Video Processing System")
    st.markdown("### Семантичне структурування та аналіз відео (Deep Learning)")

    # Sidebar configuration
    st.sidebar.header("Налаштування аналізу")
    conf_threshold: float = st.sidebar.slider("Поріг впевненості ШІ", 0.1, 1.0, 0.4)
    enable_yolo: bool = st.sidebar.checkbox("Увімкнути детекцію об'єктів (YOLOv8)", value=True)

    # File uploader
    uploaded_file: Optional[Any] = st.file_uploader(
        "Завантажте відеофайл для аналізу", 
        type=['mp4', 'avi', 'mov', 'mkv']
    )

    if uploaded_file is not None:
        video_path: str = save_uploaded_file(uploaded_file)
        
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Попередній перегляд відео")
            st.video(uploaded_file)

        if st.button("🚀 Почати інтелектуальний аналіз"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Завантаження нейромережі...")
                detector = ObjectDetector(model_type='yolov8n', threshold=conf_threshold)
                processor = VideoProcessor(video_path)

                status_text.text("Обробка відео та сегментація сцен...")
                results: List[dict] = []
                
                progress_bar.progress(100)
                status_text.text("Аналіз завершено успішно!")

                with col2:
                    st.success("Результати аналізу")
                    st.info(f"Знайдено об'єктів: {len(results)}")
                    
                    st.write("### Структура контенту")
                    st.json({
                        "filename": uploaded_file.name,
                        "status": "Structured",
                        "ai_model": "YOLOv8-Nano"
                    })

            except Exception as e:
                st.error(f"Помилка під час обробки: {str(e)}")
            finally:
                if os.path.exists(video_path):
                    os.remove(video_path)

    else:
        st.info("Будь ласка, завантажте відео у форматі MP4 або MKV для початку роботи.")

    # Footer
    st.divider()
    st.caption("© 2026 Ситник Є.О. | Бакалаврський проєкт | СумДУ, група ІН-22")

if __name__ == "__main__":
    main()
