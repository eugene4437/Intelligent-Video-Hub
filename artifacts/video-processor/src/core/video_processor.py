"""
Core video processing module.
Handles video loading, frame extraction, and scene segmentation using OpenCV.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator
import tempfile
import os


@dataclass
class Scene:
    """Represents a detected scene segment in the video."""
    scene_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    representative_frame: np.ndarray = field(default=None, repr=False)
    thumbnail: np.ndarray = field(default=None, repr=False)

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "duration": round(self.duration, 2),
            "frame_count": self.frame_count,
        }


@dataclass
class VideoMetadata:
    """Stores metadata about a video file."""
    filename: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    file_size_mb: float
    codec: str

    def to_dict(self) -> dict:
        return {
            "Filename": self.filename,
            "Resolution": f"{self.width}x{self.height}",
            "FPS": round(self.fps, 2),
            "Total Frames": self.total_frames,
            "Duration (s)": round(self.duration, 2),
            "File Size (MB)": round(self.file_size_mb, 2),
            "Codec": self.codec,
        }


class VideoProcessor:
    """
    Core video processing engine using OpenCV.
    Handles frame extraction, scene detection, and video analysis.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.metadata: VideoMetadata | None = None
        self._open()

    def _open(self):
        """Open the video file and read metadata."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        file_size = os.path.getsize(self.video_path) / (1024 * 1024)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()

        self.metadata = VideoMetadata(
            filename=Path(self.video_path).name,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration=duration,
            file_size_mb=file_size,
            codec=codec or "Unknown",
        )

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def extract_frame(self, frame_index: int) -> np.ndarray | None:
        """Extract a single frame by index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        return frame if ret else None

    def extract_frames_sampled(
        self, max_frames: int = 50
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Extract evenly sampled frames from the video."""
        total = self.metadata.total_frames
        if total == 0:
            return
        step = max(1, total // max_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while frame_idx < total:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += step

    def detect_scenes(
        self,
        threshold: float = 30.0,
        min_scene_length: int = 15,
        progress_callback=None,
    ) -> list[Scene]:
        """
        Detect scene boundaries using frame difference analysis.

        Args:
            threshold: Mean absolute pixel difference to trigger a scene cut.
            min_scene_length: Minimum frames a scene must have.
            progress_callback: Optional callable(progress: float, status: str).
        """
        fps = self.metadata.fps
        total_frames = self.metadata.total_frames

        if total_frames == 0:
            return []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = self.cap.read()
        if not ret:
            return []

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        scene_boundaries = [0]
        frame_idx = 1

        while frame_idx < total_frames:
            ret, curr_frame = self.cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)
            mean_diff = np.mean(diff)

            if mean_diff > threshold:
                if frame_idx - scene_boundaries[-1] >= min_scene_length:
                    scene_boundaries.append(frame_idx)

            prev_gray = curr_gray
            frame_idx += 1

            if progress_callback and frame_idx % 30 == 0:
                progress = frame_idx / total_frames
                progress_callback(progress, f"Analyzing frame {frame_idx}/{total_frames}")

        scene_boundaries.append(total_frames)

        scenes: list[Scene] = []
        for i, (start, end) in enumerate(
            zip(scene_boundaries[:-1], scene_boundaries[1:])
        ):
            mid_frame_idx = (start + end) // 2
            rep_frame = self.extract_frame(mid_frame_idx)
            thumbnail = self._make_thumbnail(rep_frame) if rep_frame is not None else None

            scenes.append(
                Scene(
                    scene_id=i + 1,
                    start_frame=start,
                    end_frame=end,
                    start_time=start / fps,
                    end_time=end / fps,
                    duration=(end - start) / fps,
                    representative_frame=rep_frame,
                    thumbnail=thumbnail,
                )
            )

        return scenes

    def _make_thumbnail(self, frame: np.ndarray, size: tuple = (320, 180)) -> np.ndarray:
        """Create a small thumbnail from a frame (RGB)."""
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def compute_frame_difference_curve(self, sample_rate: int = 5) -> tuple[list[int], list[float]]:
        """Compute the mean frame difference at regular intervals for visualization."""
        fps = self.metadata.fps
        total_frames = self.metadata.total_frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_indices = []
        differences = []
        ret, prev = self.cap.read()
        if not ret:
            return [], []
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

        for i in range(1, total_frames):
            ret, curr = self.cap.read()
            if not ret:
                break
            if i % sample_rate == 0:
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                diff = np.mean(cv2.absdiff(prev_gray, curr_gray))
                frame_indices.append(i)
                differences.append(float(diff))
                prev_gray = curr_gray

        return frame_indices, differences
