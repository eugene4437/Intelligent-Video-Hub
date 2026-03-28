"""
Object detection module using OpenCV DNN with YOLOv3.
Downloads model weights automatically on first use.
"""

import cv2
import numpy as np
import requests
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import tempfile


MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "yolo"

YOLOV3_TINY_CONFIG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
YOLOV3_TINY_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3-tiny.weights"
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Fallback: use a lightweight set of COCO class names if download fails
COCO_CLASS_NAMES_FALLBACK = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
    "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """A single detected object in a frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x, y, w, h) in pixel coordinates
    center: tuple  # (cx, cy) normalized 0-1

    def to_dict(self) -> dict:
        x, y, w, h = self.bbox
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        }


class ModelDownloader:
    """Handles downloading YOLO model files."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _download_file(self, url: str, dest: Path, progress_callback=None) -> bool:
        """Download a file with optional progress reporting."""
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total > 0:
                        progress_callback(downloaded / total)
            return True
        except Exception as e:
            if dest.exists():
                dest.unlink()
            return False

    def ensure_model_files(self, progress_callback=None) -> tuple[Optional[Path], Optional[Path], list[str]]:
        """
        Ensure all model files exist. Download if missing.
        Returns (cfg_path, weights_path, class_names).
        """
        cfg_path = self.models_dir / "yolov3-tiny.cfg"
        weights_path = self.models_dir / "yolov3-tiny.weights"
        names_path = self.models_dir / "coco.names"

        class_names = COCO_CLASS_NAMES_FALLBACK

        if progress_callback:
            progress_callback(0.0, "Checking model files...")

        # Download config
        if not cfg_path.exists():
            if progress_callback:
                progress_callback(0.1, "Downloading YOLOv3-tiny config...")
            ok = self._download_file(YOLOV3_TINY_CONFIG_URL, cfg_path)
            if not ok:
                return None, None, class_names

        if progress_callback:
            progress_callback(0.2, "Config ready.")

        # Download class names
        if not names_path.exists():
            if progress_callback:
                progress_callback(0.25, "Downloading class names...")
            ok = self._download_file(COCO_NAMES_URL, names_path)
            if ok and names_path.exists():
                class_names = names_path.read_text().strip().splitlines()

        elif names_path.exists():
            class_names = names_path.read_text().strip().splitlines()

        # Download weights (largest file)
        if not weights_path.exists():
            if progress_callback:
                progress_callback(0.3, "Downloading YOLOv3-tiny weights (~35 MB)...")

            def weights_progress(p):
                overall = 0.3 + p * 0.7
                if progress_callback:
                    progress_callback(overall, f"Downloading weights: {p*100:.0f}%")

            ok = self._download_file(
                YOLOV3_TINY_WEIGHTS_URL, weights_path, weights_progress
            )
            if not ok:
                return cfg_path, None, class_names

        if progress_callback:
            progress_callback(1.0, "Model files ready.")

        return cfg_path, weights_path, class_names


class ObjectDetector:
    """
    Object detector using OpenCV DNN with YOLOv3-tiny.
    Detects objects in video frames with bounding boxes and class labels.
    """

    def __init__(
        self,
        cfg_path: Path,
        weights_path: Path,
        class_names: list[str],
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.4,
    ):
        self.class_names = class_names
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.net = cv2.dnn.readNet(str(weights_path), str(cfg_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._output_layer_names = self._get_output_layers()

    def _get_output_layers(self) -> list[str]:
        layer_names = self.net.getLayerNames()
        try:
            out_indices = self.net.getUnconnectedOutLayers()
            if isinstance(out_indices[0], (list, np.ndarray)):
                return [layer_names[i[0] - 1] for i in out_indices]
            return [layer_names[i - 1] for i in out_indices]
        except Exception:
            return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLO detection on a single BGR frame."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self._output_layer_names)

        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence >= self.conf_threshold:
                    cx, cy, bw, bh = (
                        int(detection[0] * w),
                        int(detection[1] * h),
                        int(detection[2] * w),
                        int(detection[3] * h),
                    )
                    x = cx - bw // 2
                    y = cy - bh // 2
                    boxes.append([x, y, bw, bh])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.conf_threshold, self.nms_threshold
        )

        detections: list[Detection] = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, "flatten") else indices
            for i in flat_indices:
                x, y, bw, bh = boxes[i]
                class_id = class_ids[i]
                class_name = (
                    self.class_names[class_id]
                    if class_id < len(self.class_names)
                    else f"class_{class_id}"
                )
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidences[i],
                        bbox=(x, y, bw, bh),
                        center=(
                            (x + bw / 2) / w,
                            (y + bh / 2) / h,
                        ),
                    )
                )

        return detections

    def draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame (BGR → RGB output)."""
        result = frame.copy()
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

        for det in detections:
            x, y, bw, bh = det.bbox
            color = [int(c) for c in colors[det.class_id % len(colors)]]
            cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(result, (x, y - text_h - 4), (x + text_w, y), color, -1)
            cv2.putText(
                result, label, (x, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
            )

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
