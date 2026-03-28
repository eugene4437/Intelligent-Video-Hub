"""
File utility helpers for video upload and temporary file management.
"""

import tempfile
import os
import io
from pathlib import Path


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
MAX_FILE_SIZE_MB = 500


def save_uploaded_video(uploaded_file) -> str:
    """
    Save a Streamlit UploadedFile to a temporary file on disk.
    Returns the path to the temporary file.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def cleanup_temp_file(path: str):
    """Safely remove a temporary file."""
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ss"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
