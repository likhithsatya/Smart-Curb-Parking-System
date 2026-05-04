import cv2
import numpy as np

from src.detector import Detection
from src.tracking import Track


def draw_polygon(frame: np.ndarray, polygon: list[tuple[int, int]]) -> np.ndarray:
    output = frame.copy()
    if len(polygon) >= 3:
        points = np.array(polygon, dtype=np.int32)
        overlay = output.copy()
        cv2.fillPoly(overlay, [points], (0, 0, 255))
        output = cv2.addWeighted(overlay, 0.25, output, 0.75, 0)
        cv2.polylines(output, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    return output


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection] | list[Track],
    violating_track_ids: set[int] | None = None,
) -> np.ndarray:
    output = frame.copy()
    violating_track_ids = violating_track_ids or set()

    for item in detections:
        x1, y1, x2, y2 = item.bbox
        track_id = getattr(item, "track_id", None)
        is_violation = track_id in violating_track_ids
        color = (0, 0, 255) if is_violation else (0, 180, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label_parts = [item.vehicle_type, f"{item.confidence:.2f}"]
        if track_id is not None:
            label_parts.insert(0, f"ID {track_id}")
        if is_violation:
            label_parts.append("VIOLATION")

        label = " | ".join(label_parts)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
