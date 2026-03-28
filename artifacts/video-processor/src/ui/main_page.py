"""
Main Streamlit page layout and orchestration.
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.video_processor import VideoProcessor
from src.core.object_detector import ModelDownloader, ObjectDetector, COCO_CLASS_NAMES_FALLBACK
from src.utils.file_utils import save_uploaded_video, cleanup_temp_file, format_time
from src.utils.visualization import (
    plot_frame_difference_curve,
    plot_scene_durations,
    plot_object_frequency,
    plot_confidence_distribution,
    plot_objects_per_scene,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "yolo"


def render_sidebar() -> dict:
    """Render the sidebar and return user configuration."""
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Scene Detection")
    threshold = st.sidebar.slider(
        "Scene Change Threshold",
        min_value=5.0,
        max_value=80.0,
        value=30.0,
        step=1.0,
        help="Higher = fewer scenes detected. Lower = more sensitive to cuts.",
    )
    min_scene_frames = st.sidebar.slider(
        "Minimum Scene Length (frames)",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
        help="Minimum number of frames a scene must have.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Object Detection")
    run_object_detection = st.sidebar.checkbox("Enable Object Detection", value=True)
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        help="Minimum confidence to accept a detection.",
    )
    frames_to_analyze = st.sidebar.slider(
        "Frames to Analyze per Scene",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of frames sampled per scene for object detection.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.markdown(
        """
        **Intelligent Video Processing**  
        & Semantic Structuring System  
        
        Uses **OpenCV** for scene segmentation  
        and **YOLOv3** for object detection.  
        
        Built with ❤️ using Python & Streamlit.
        """
    )

    return {
        "threshold": threshold,
        "min_scene_frames": min_scene_frames,
        "run_object_detection": run_object_detection,
        "conf_threshold": conf_threshold,
        "frames_to_analyze": frames_to_analyze,
    }


def render_video_metadata(metadata):
    """Display video metadata as a horizontal metric row."""
    st.subheader("📋 Video Information")
    cols = st.columns(4)
    d = metadata.to_dict()
    items = list(d.items())

    for i, col in enumerate(cols):
        if i < len(items):
            col.metric(items[i][0], items[i][1])
    cols2 = st.columns(4)
    for i, col in enumerate(cols2):
        idx = i + 4
        if idx < len(items):
            col.metric(items[idx][0], items[idx][1])


def render_scene_gallery(scenes, detections_by_scene=None):
    """Display scenes as a visual gallery with thumbnails."""
    st.subheader(f"🎬 Detected Scenes ({len(scenes)} total)")

    cols_per_row = 3
    for row_start in range(0, len(scenes), cols_per_row):
        row_scenes = scenes[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, scene in zip(cols, row_scenes):
            with col:
                if scene.thumbnail is not None:
                    st.image(
                        scene.thumbnail,
                        caption=f"Scene {scene.scene_id}",
                        use_container_width=True,
                    )
                else:
                    st.markdown(f"**Scene {scene.scene_id}** *(no thumbnail)*")

                st.markdown(
                    f"⏱ `{format_time(scene.start_time)}` → `{format_time(scene.end_time)}`  \n"
                    f"📊 {scene.frame_count} frames · {scene.duration:.2f}s"
                )
                if detections_by_scene and scene.scene_id in detections_by_scene:
                    det_list = detections_by_scene[scene.scene_id]
                    from collections import Counter
                    ctr = Counter(d["class"] for d in det_list)
                    if ctr:
                        top = ", ".join(f"{cls}×{cnt}" for cls, cnt in ctr.most_common(3))
                        st.caption(f"🔍 {top}")


def render_scene_table(scenes):
    """Show scenes as a sortable dataframe."""
    if not scenes:
        return
    data = [s.to_dict() for s in scenes]
    df = pd.DataFrame(data)
    df["start_time"] = df["start_time"].apply(format_time)
    df["end_time"] = df["end_time"].apply(format_time)
    df.columns = [c.replace("_", " ").title() for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)


def run_scene_detection(video_path: str, config: dict) -> tuple:
    """Run scene detection with a Streamlit progress bar."""
    st.subheader("🔍 Scene Detection")
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(progress: float, status: str):
        progress_bar.progress(min(progress, 1.0))
        status_text.text(status)

    with VideoProcessor(video_path) as vp:
        metadata = vp.metadata

        # Compute frame difference curve
        with st.spinner("Computing frame differences..."):
            frame_indices, differences = vp.compute_frame_difference_curve(sample_rate=5)

        # Detect scenes
        scenes = vp.detect_scenes(
            threshold=config["threshold"],
            min_scene_length=config["min_scene_frames"],
            progress_callback=update_progress,
        )

    progress_bar.progress(1.0)
    status_text.text(f"✅ Found {len(scenes)} scene(s).")
    return metadata, scenes, frame_indices, differences


def run_object_detection(
    video_path: str, scenes, config: dict
) -> tuple[list[dict], dict[int, list[dict]]]:
    """Download model if needed, then run detection on sampled frames."""
    st.subheader("🤖 Object Detection (YOLOv3-tiny)")
    prog = st.progress(0.0)
    status_txt = st.empty()

    downloader = ModelDownloader(MODELS_DIR)

    def dl_progress(p, msg=""):
        prog.progress(min(p * 0.5, 0.5))
        status_txt.text(msg)

    cfg_path, weights_path, class_names = downloader.ensure_model_files(dl_progress)

    if cfg_path is None or weights_path is None:
        status_txt.warning(
            "⚠️ Could not download YOLO weights. Object detection skipped. "
            "Check your network connection and try again."
        )
        return [], {}

    status_txt.text("Loading neural network...")
    detector = ObjectDetector(
        cfg_path=cfg_path,
        weights_path=weights_path,
        class_names=class_names,
        confidence_threshold=config["conf_threshold"],
    )

    all_detections: list[dict] = []
    detections_by_scene: dict[int, list[dict]] = {}
    import cv2

    total_scenes = len(scenes)
    for scene_idx, scene in enumerate(scenes):
        scene_detections: list[dict] = []
        if scene.representative_frame is None:
            continue

        frame_bgr = cv2.cvtColor(scene.representative_frame, cv2.COLOR_RGB2BGR) \
            if scene.representative_frame.shape[2] == 3 else scene.representative_frame

        dets = detector.detect(frame_bgr)
        for d in dets:
            dd = d.to_dict()
            dd["scene_id"] = scene.scene_id
            scene_detections.append(dd)
            all_detections.append(dd)

        # Annotate representative frame
        if dets:
            annotated = detector.draw_detections(frame_bgr, dets)
            scene.thumbnail = cv2.resize(annotated, (320, 180))

        detections_by_scene[scene.scene_id] = scene_detections
        overall = 0.5 + (scene_idx + 1) / total_scenes * 0.5
        prog.progress(min(overall, 1.0))
        status_txt.text(f"Scene {scene.scene_id}/{total_scenes}: {len(dets)} objects found")

    prog.progress(1.0)
    status_txt.text(f"✅ Object detection complete. {len(all_detections)} total detections.")
    return all_detections, detections_by_scene


def render_main_page():
    """Main application rendering function."""

    # --- Header ---
    st.markdown(
        """
        <h1 style="text-align:center; padding-bottom:0;">
            🎬 Intelligent Video Processing System
        </h1>
        <p style="text-align:center; color:#888; margin-top:0; font-size:1.05em;">
            Semantic scene segmentation & deep learning object detection
        </p>
        <hr/>
        """,
        unsafe_allow_html=True,
    )

    config = render_sidebar()

    # --- Upload Section ---
    st.subheader("📁 Upload Video")
    uploaded = st.file_uploader(
        "Drag and drop a video file or click to browse",
        type=["mp4", "avi", "mov", "mkv", "webm", "flv"],
        help="Supported formats: MP4, AVI, MOV, MKV, WebM, FLV",
    )

    if uploaded is None:
        st.info("👆 Upload a video to get started. Supports MP4, AVI, MOV, MKV, WebM, and more.")
        _render_feature_overview()
        return

    # --- Save to temp file ---
    if "temp_path" not in st.session_state or st.session_state.get("last_filename") != uploaded.name:
        if "temp_path" in st.session_state:
            cleanup_temp_file(st.session_state["temp_path"])
        with st.spinner("Saving video..."):
            temp_path = save_uploaded_video(uploaded)
        st.session_state["temp_path"] = temp_path
        st.session_state["last_filename"] = uploaded.name
        # Clear previous results
        for key in ["metadata", "scenes", "frame_indices", "differences", "all_detections", "detections_by_scene"]:
            st.session_state.pop(key, None)

    temp_path = st.session_state["temp_path"]

    # Show video preview
    st.video(uploaded)

    # --- Analyze Button ---
    col_btn, col_tip = st.columns([1, 3])
    with col_btn:
        run = st.button("🚀 Analyze Video", use_container_width=True, type="primary")

    if run:
        # Scene detection
        with st.container():
            metadata, scenes, frame_indices, differences = run_scene_detection(
                temp_path, config
            )
        st.session_state["metadata"] = metadata
        st.session_state["scenes"] = scenes
        st.session_state["frame_indices"] = frame_indices
        st.session_state["differences"] = differences

        # Object detection
        if config["run_object_detection"] and scenes:
            with st.container():
                all_detections, detections_by_scene = run_object_detection(
                    temp_path, scenes, config
                )
            st.session_state["all_detections"] = all_detections
            st.session_state["detections_by_scene"] = detections_by_scene
        else:
            st.session_state["all_detections"] = []
            st.session_state["detections_by_scene"] = {}

    # --- Render results if available ---
    if "metadata" in st.session_state:
        metadata = st.session_state["metadata"]
        scenes = st.session_state["scenes"]
        frame_indices = st.session_state["frame_indices"]
        differences = st.session_state["differences"]
        all_detections = st.session_state.get("all_detections", [])
        detections_by_scene = st.session_state.get("detections_by_scene", {})

        st.markdown("---")
        render_video_metadata(metadata)

        # Tabs for different views
        tab_scenes, tab_analysis, tab_objects, tab_export = st.tabs([
            "🎬 Scene Gallery", "📊 Analysis Charts", "🤖 Object Detection", "💾 Export"
        ])

        with tab_scenes:
            if not scenes:
                st.warning("No scenes detected. Try lowering the threshold in the sidebar.")
            else:
                render_scene_gallery(scenes, detections_by_scene)
                st.markdown("---")
                with st.expander("📄 Scene Details Table", expanded=False):
                    render_scene_table(scenes)

        with tab_analysis:
            st.subheader("📊 Frame Difference Curve")
            if frame_indices and differences:
                boundary_frames = [s.start_frame for s in scenes] + [scenes[-1].end_frame] if scenes else []
                fig_diff = plot_frame_difference_curve(
                    frame_indices, differences, boundary_frames,
                    config["threshold"], metadata.fps
                )
                st.plotly_chart(fig_diff, use_container_width=True)

            if scenes:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_scene_durations(scenes), use_container_width=True)
                with col2:
                    if detections_by_scene:
                        counts = {k: len(v) for k, v in detections_by_scene.items()}
                        st.plotly_chart(plot_objects_per_scene(counts), use_container_width=True)

        with tab_objects:
            if not all_detections:
                if not config["run_object_detection"]:
                    st.info("Enable Object Detection in the sidebar to see results here.")
                else:
                    st.info("No objects detected. Try lowering the confidence threshold.")
            else:
                st.metric("Total Detections", len(all_detections))
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(
                        plot_object_frequency(all_detections), use_container_width=True
                    )
                with col2:
                    st.plotly_chart(
                        plot_confidence_distribution(all_detections), use_container_width=True
                    )

                with st.expander("🔍 Detection Details Table", expanded=False):
                    df = pd.DataFrame(all_detections)
                    st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_export:
            st.subheader("💾 Export Results")
            if scenes:
                scene_data = [s.to_dict() for s in scenes]
                df_scenes = pd.DataFrame(scene_data)
                st.download_button(
                    "⬇️ Download Scene Data (CSV)",
                    df_scenes.to_csv(index=False),
                    file_name=f"{metadata.filename}_scenes.csv",
                    mime="text/csv",
                )

            if all_detections:
                df_dets = pd.DataFrame(all_detections)
                st.download_button(
                    "⬇️ Download Detection Data (CSV)",
                    df_dets.to_csv(index=False),
                    file_name=f"{metadata.filename}_detections.csv",
                    mime="text/csv",
                )

            if not scenes and not all_detections:
                st.info("Run analysis first to export results.")


def _render_feature_overview():
    """Show feature cards on the landing state."""
    st.markdown("---")
    st.subheader("✨ What This System Does")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            #### 🎬 Scene Segmentation
            Automatically detects cuts and transitions using frame-level difference analysis.
            Each scene is timestamped and represented with a thumbnail.
            """
        )
    with c2:
        st.markdown(
            """
            #### 🤖 Object Detection
            Uses **YOLOv3-tiny** (OpenCV DNN) to identify objects in each scene.
            Draws annotated bounding boxes with class labels and confidence scores.
            """
        )
    with c3:
        st.markdown(
            """
            #### 📊 Semantic Analysis
            Interactive Plotly charts show scene lengths, object frequencies,
            confidence distributions, and frame-difference timelines.
            """
        )

    st.markdown("---")
    st.subheader("🛠️ How to Use")
    st.markdown(
        """
        1. **Upload** your video file using the uploader above.
        2. **Configure** detection parameters in the left sidebar.
        3. Click **Analyze Video** to start processing.
        4. Browse results in the **Scene Gallery**, **Analysis Charts**, and **Object Detection** tabs.
        5. **Export** your structured data as CSV files.
        """
    )
