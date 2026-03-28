# Intelligent Video Processing and Semantic Structuring System

A professional Python application for automated video analysis, scene segmentation, and deep-learning-based object detection using OpenCV and YOLOv3.

---

## Overview

This system allows users to upload any video file and automatically:

- **Segment** the video into semantically meaningful scenes using frame-difference analysis
- **Detect** and label objects in each scene using the YOLOv3-tiny neural network (via OpenCV DNN)
- **Visualise** results with interactive Plotly charts (scene timelines, object frequencies, confidence distributions)
- **Export** structured scene and detection data as CSV files

The entire pipeline runs in a clean, interactive [Streamlit](https://streamlit.io) web interface — no configuration required beyond uploading a video.

---

## Project Structure

```
.
├── app.py                      # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
│
├── src/                        # Source code
│   ├── core/
│   │   ├── video_processor.py  # OpenCV frame extraction & scene detection
│   │   └── object_detector.py  # YOLOv3 DNN object detection
│   ├── ui/
│   │   └── main_page.py        # Streamlit page layout & orchestration
│   └── utils/
│       ├── file_utils.py       # File I/O and formatting helpers
│       └── visualization.py    # Plotly chart builders
│
├── data/
│   └── samples/                # Place sample video files here
│
└── models/
    └── yolo/                   # YOLOv3-tiny weights downloaded automatically on first run
```

---

## Features

### Scene Segmentation
- Frame-to-frame pixel difference analysis detects hard cuts and gradual transitions
- Configurable sensitivity threshold and minimum scene length
- Each scene is represented with a timestamp, thumbnail, and frame count

### Object Detection
- Uses **YOLOv3-tiny** loaded via **OpenCV DNN** — no GPU required
- Detects 80 COCO object classes (person, car, dog, laptop, etc.)
- Model weights (~34 MB) are downloaded automatically on first run
- Annotated frames show bounding boxes with class labels and confidence scores

### Analysis & Visualisation
- Interactive **frame-difference timeline** with scene boundary markers
- **Scene duration** bar chart
- **Object class frequency** ranking
- **Confidence score** histogram
- **Detections per scene** breakdown

### Export
- Download scene data (CSV): scene ID, timestamps, duration, frame count
- Download detection data (CSV): class name, confidence, bounding box coordinates, scene ID

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd video-processing-system

# (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Usage

1. **Upload** a video file (MP4, AVI, MOV, MKV, WebM, FLV supported)
2. **Configure** detection settings in the left sidebar:
   - Adjust the scene change threshold (higher = fewer scenes)
   - Set minimum scene length in frames
   - Toggle object detection on/off
   - Set the confidence threshold
3. Click **Analyze Video**
4. Explore results across four tabs:
   - **Scene Gallery** — thumbnails with timestamps
   - **Analysis Charts** — interactive Plotly visualisations
   - **Object Detection** — class rankings and annotated detections
   - **Export** — download CSV reports

---

## Technical Architecture

| Component | Technology |
|-----------|-----------|
| Web Interface | [Streamlit](https://streamlit.io) |
| Video I/O | [OpenCV](https://opencv.org) |
| Scene Detection | OpenCV frame differencing |
| Object Detection | YOLOv3-tiny via OpenCV DNN |
| Visualisation | [Plotly](https://plotly.com) |
| Data Handling | [Pandas](https://pandas.pydata.org) |
| Image Processing | [Pillow](https://pillow.readthedocs.io), [scikit-image](https://scikit-image.org) |

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| Scene Threshold | 30.0 | Mean pixel difference to trigger a scene cut |
| Min Scene Length | 15 frames | Minimum number of frames per scene |
| Confidence | 0.40 | Minimum YOLO detection confidence |
| Frames per Scene | 3 | Frames sampled per scene for detection |

---

## Notes on Model Weights

On first run, the app automatically downloads the YOLOv3-tiny model files (~34 MB total) from the official Darknet repository into `models/yolo/`. This requires an internet connection on first launch only. The weights are excluded from version control via `.gitignore` — they will be re-downloaded as needed.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [YOLO / Darknet](https://github.com/pjreddie/darknet) by Joseph Redmon — object detection framework
- [OpenCV](https://opencv.org) — Open Source Computer Vision Library
- [Streamlit](https://streamlit.io) — Python web framework for data apps
- [COCO Dataset](https://cocodataset.org) — Common Objects in Context (80-class labels)
