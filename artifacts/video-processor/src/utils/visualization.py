"""
Visualization utilities using Plotly for interactive charts.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter


def plot_frame_difference_curve(
    frame_indices: list[int],
    differences: list[float],
    scene_boundaries: list[int],
    threshold: float,
    fps: float,
) -> go.Figure:
    """Plot the frame-to-frame difference curve with scene boundary markers."""
    times = [f / fps for f in frame_indices]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=times,
            y=differences,
            mode="lines",
            name="Frame Difference",
            line=dict(color="#4f86f7", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(79,134,247,0.15)",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold:.0f})",
        annotation_position="top right",
    )

    for boundary_frame in scene_boundaries[1:-1]:
        t = boundary_frame / fps
        fig.add_vline(
            x=t,
            line_dash="dot",
            line_color="rgba(255, 165, 0, 0.8)",
            line_width=2,
        )

    fig.update_layout(
        title="Frame-to-Frame Difference Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Mean Pixel Difference",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
        template="plotly_dark",
    )
    return fig


def plot_scene_durations(scenes) -> go.Figure:
    """Bar chart showing duration of each scene."""
    if not scenes:
        return go.Figure()

    scene_ids = [f"Scene {s.scene_id}" for s in scenes]
    durations = [round(s.duration, 2) for s in scenes]

    fig = px.bar(
        x=scene_ids,
        y=durations,
        labels={"x": "Scene", "y": "Duration (s)"},
        title="Scene Duration Breakdown",
        color=durations,
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=50, b=60),
        template="plotly_dark",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def plot_object_frequency(all_detections: list[dict]) -> go.Figure:
    """Horizontal bar chart of detected object class frequencies."""
    if not all_detections:
        return go.Figure()

    counter = Counter(d["class"] for d in all_detections)
    classes = list(counter.keys())
    counts = list(counter.values())

    sorted_pairs = sorted(zip(counts, classes), reverse=True)
    counts, classes = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig = go.Figure(
        go.Bar(
            x=list(counts),
            y=list(classes),
            orientation="h",
            marker=dict(
                color=list(counts),
                colorscale="Viridis",
            ),
        )
    )
    fig.update_layout(
        title="Detected Object Classes",
        xaxis_title="Count",
        yaxis_title="Object Class",
        height=max(250, 30 * len(classes) + 80),
        margin=dict(l=120, r=20, t=50, b=40),
        template="plotly_dark",
    )
    return fig


def plot_confidence_distribution(all_detections: list[dict]) -> go.Figure:
    """Histogram of detection confidence scores."""
    if not all_detections:
        return go.Figure()

    confidences = [d["confidence"] for d in all_detections]
    fig = px.histogram(
        x=confidences,
        nbins=20,
        labels={"x": "Confidence Score", "y": "Count"},
        title="Detection Confidence Distribution",
        color_discrete_sequence=["#4f86f7"],
    )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        template="plotly_dark",
    )
    return fig


def plot_objects_per_scene(scene_detection_counts: dict[int, int]) -> go.Figure:
    """Bar chart of object detections per scene."""
    if not scene_detection_counts:
        return go.Figure()

    scene_ids = [f"Scene {k}" for k in sorted(scene_detection_counts.keys())]
    counts = [scene_detection_counts[k] for k in sorted(scene_detection_counts.keys())]

    fig = px.bar(
        x=scene_ids,
        y=counts,
        labels={"x": "Scene", "y": "Detections"},
        title="Object Detections per Scene",
        color=counts,
        color_continuous_scale="Oranges",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=50, b=60),
        template="plotly_dark",
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig
