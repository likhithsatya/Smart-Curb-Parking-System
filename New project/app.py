from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.config import DB_PATH, DEFAULT_MODEL, FRAME_DIR, ensure_directories
from src.detector import VehicleDetector
from src.preprocessing import apply_preprocessing, blend_edges
from src.storage import ViolationStore
from src.tracking import CentroidTracker
from src.violation import RuleBasedViolationDetector
from src.visualization import bgr_to_rgb, draw_detections, draw_polygon


IMAGE_TYPES = {"jpg", "jpeg", "png"}
VIDEO_TYPES = {"mp4", "mov", "avi", "mkv"}


def main() -> None:
    st.set_page_config(page_title="Smart Curb Parking Enforcement", layout="wide")
    ensure_directories()
    store = ViolationStore(DB_PATH)

    st.title("Smart Curb Parking Enforcement")

    with st.sidebar:
        st.header("Detection Settings")
        model_path = st.text_input("YOLO model path", value=DEFAULT_MODEL)
        confidence_threshold = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
        dwell_threshold = st.slider("Dwell threshold seconds", 0.0, 120.0, 10.0, 1.0)
        overlap_threshold = st.slider("Zone overlap threshold", 0.01, 1.0, 0.15, 0.01)
        frame_stride = st.slider("Video frame stride", 1, 30, 5, 1)

        st.divider()
        if st.button("Clear violation history"):
            store.clear()
            st.success("Violation history cleared.")

    uploaded_file = st.file_uploader(
        "Upload image or video",
        type=sorted(IMAGE_TYPES | VIDEO_TYPES),
    )

    if uploaded_file is None:
        render_history(store)
        return

    suffix = Path(uploaded_file.name).suffix.lower()
    media_kind = "image" if suffix.lstrip(".") in IMAGE_TYPES else "video"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        media_path = Path(tmp.name)

    preview_frame = load_preview_frame(media_path, media_kind)
    if preview_frame is None:
        st.error("Could not read the uploaded file.")
        render_history(store)
        return

    st.subheader("Define No-Parking Zone")
    st.caption("Use the polygon tool in the toolbar, then click points around the restricted curb area.")
    polygon = draw_zone_selector(preview_frame)

    preview_with_zone = draw_polygon(preview_frame, polygon)
    blurred, edges = apply_preprocessing(preview_frame)
    edge_preview = blend_edges(blurred, edges)

    left, right = st.columns(2)
    with left:
        st.image(bgr_to_rgb(preview_with_zone), caption="Zone preview", use_container_width=True)
    with right:
        st.image(bgr_to_rgb(edge_preview), caption="Gaussian + edge preprocessing preview", use_container_width=True)

    run_disabled = len(polygon) < 3
    if run_disabled:
        st.info("Draw a no-parking polygon to enable detection.")

    if st.button("Run Detection", type="primary", disabled=run_disabled):
        run_detection(
            media_path=media_path,
            media_kind=media_kind,
            source_name=uploaded_file.name,
            polygon=polygon,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            dwell_threshold=dwell_threshold,
            overlap_threshold=overlap_threshold,
            frame_stride=frame_stride,
            store=store,
        )

    render_history(store)


def load_preview_frame(media_path: Path, media_kind: str) -> np.ndarray | None:
    if media_kind == "image":
        image = Image.open(media_path).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    capture = cv2.VideoCapture(str(media_path))
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def draw_zone_selector(frame: np.ndarray) -> list[tuple[int, int]]:
    rgb_frame = bgr_to_rgb(frame)
    height, width = rgb_frame.shape[:2]

    max_width = 900
    scale = min(1.0, max_width / width)
    canvas_width = int(width * scale)
    canvas_height = int(height * scale)
    display_frame = cv2.resize(rgb_frame, (canvas_width, canvas_height))

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=3,
        stroke_color="#ff0000",
        background_image=Image.fromarray(display_frame),
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="polygon",
        key="zone_canvas",
    )

    if not canvas_result.json_data:
        return []

    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return []

    polygon_object = objects[-1]
    raw_points = polygon_object.get("path") or []
    points: list[tuple[int, int]] = []

    for point in raw_points:
        if len(point) >= 3 and point[0] in {"M", "L"}:
            x = int(float(point[1]) / scale)
            y = int(float(point[2]) / scale)
            points.append((x, y))

    return points


def run_detection(
    media_path: Path,
    media_kind: str,
    source_name: str,
    polygon: list[tuple[int, int]],
    model_path: str,
    confidence_threshold: float,
    dwell_threshold: float,
    overlap_threshold: float,
    frame_stride: int,
    store: ViolationStore,
) -> None:
    detector = VehicleDetector(model_path, confidence_threshold)
    violation_detector = RuleBasedViolationDetector(dwell_threshold, overlap_threshold)
    tracker = CentroidTracker()

    status = st.empty()
    preview = st.empty()
    progress = st.progress(0)
    violation_count = 0

    if media_kind == "image":
        frame = load_preview_frame(media_path, media_kind)
        if frame is None:
            st.error("Could not read image.")
            return
        detections = detector.detect(frame)
        tracks = tracker.update(detections, 0.0)
        image_violation_detector = RuleBasedViolationDetector(0.0, overlap_threshold)
        events = image_violation_detector.evaluate(tracks, polygon, frame.shape, 0.0)
        violation_count += save_events(store, source_name, frame, events, 0.0)
        annotated = draw_detections(draw_polygon(frame, polygon), tracks, {event.track_id for event in events})
        preview.image(bgr_to_rgb(annotated), caption="Detection result", use_container_width=True)
        progress.progress(1.0)
        status.success(f"Detection complete. Stored {violation_count} violation record(s).")
        return

    capture = cv2.VideoCapture(str(media_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        timestamp_seconds = frame_index / fps
        _, edges = apply_preprocessing(frame)
        processed_frame = blend_edges(frame, edges)
        detections = detector.detect(processed_frame)
        tracks = tracker.update(detections, timestamp_seconds)
        events = violation_detector.evaluate(tracks, polygon, frame.shape, timestamp_seconds)
        violation_count += save_events(store, source_name, frame, events, timestamp_seconds)

        annotated = draw_detections(
            draw_polygon(frame, polygon),
            tracks,
            {event.track_id for event in events},
        )
        preview.image(bgr_to_rgb(annotated), caption=f"Frame {frame_index}", use_container_width=True)

        if total_frames:
            progress.progress(min(frame_index / total_frames, 1.0))
        status.info(f"Processed frame {frame_index}. Stored {violation_count} violation(s).")
        frame_index += 1

    capture.release()
    progress.progress(1.0)
    status.success(f"Detection complete. Stored {violation_count} violation record(s).")


def save_events(
    store: ViolationStore,
    source_name: str,
    frame: np.ndarray,
    events,
    timestamp_seconds: float,
) -> int:
    count = 0
    for event in events:
        frame_name = (
            f"{Path(source_name).stem}_track{event.track_id}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        )
        frame_path = FRAME_DIR / frame_name
        annotated_frame = draw_detections(frame, [], {event.track_id})
        x1, y1, x2, y2 = event.bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(frame_path), annotated_frame)

        store.add_violation(
            source_name=source_name,
            frame_time_seconds=timestamp_seconds,
            frame_image_path=str(frame_path),
            confidence=event.confidence,
            vehicle_type=event.vehicle_type,
            track_id=event.track_id,
            dwell_time_seconds=event.dwell_time_seconds,
            overlap_ratio=event.overlap_ratio,
            bbox=event.bbox,
        )
        count += 1
    return count


def render_history(store: ViolationStore) -> None:
    st.subheader("Violation History")
    df = store.list_violations()
    if df.empty:
        st.info("No violations recorded yet.")
        return

    display_df = df.copy()
    display_df["confidence"] = display_df["confidence"].map(lambda value: f"{value:.2f}")
    display_df["dwell_time_seconds"] = display_df["dwell_time_seconds"].map(lambda value: f"{value:.1f}")
    display_df["overlap_ratio"] = display_df["overlap_ratio"].map(lambda value: f"{value:.2f}")
    st.dataframe(display_df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export results as CSV",
        data=csv_bytes,
        file_name="parking_violations.csv",
        mime="text/csv",
    )

    latest = df.iloc[0]
    frame_path = Path(str(latest["frame_image_path"]))
    if frame_path.exists():
        st.image(str(frame_path), caption="Latest violation frame", use_container_width=True)


if __name__ == "__main__":
    main()
