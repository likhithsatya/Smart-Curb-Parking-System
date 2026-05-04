# Smart Curb Parking Enforcement System

A modular Python + Streamlit prototype for detecting curb-parking violations from uploaded images or videos.

## Features

- OpenCV image/video ingestion.
- YOLOv8 vehicle detection through Ultralytics.
- Gaussian blur and Canny edge preprocessing preview.
- User-defined no-parking polygon.
- Rule-based violation detection:
  - vehicle overlaps the restricted polygon
  - vehicle remains in the zone longer than a configurable threshold
- SQLite violation history with timestamp, frame image path, confidence score, vehicle type, and dwell time.
- CSV export from the dashboard.

## Folder Structure

```text
smart-curb-parking/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py
│   ├── preprocessing.py
│   ├── storage.py
│   ├── tracking.py
│   ├── violation.py
│   └── visualization.py
└── data/
    ├── frames/
    └── smart_curb.db
```

The `data/` directory is created automatically at runtime.

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the dashboard.

```bash
streamlit run app.py
```

4. Open the URL Streamlit prints, usually `http://localhost:8501`.

On first run, Ultralytics downloads `yolov8n.pt` unless you provide another model path in the sidebar.

## Usage

1. Upload an image or video.
2. Draw a no-parking polygon over the preview frame.
3. Configure overlap and dwell-time thresholds.
4. Run detection.
5. Review stored violations and export them as CSV.

## Extension Ideas

- Replace centroid tracking with DeepSORT, ByteTrack, or a camera-specific tracker.
- Store zone definitions per camera.
- Add license plate recognition.
- Add alert routing by email, Slack, or webhooks.
- Calibrate frame coordinates to real-world curb geometry.
- Move inference to a background worker for long videos.

