from dataclasses import dataclass

import cv2
import numpy as np

from src.tracking import Track


@dataclass(frozen=True)
class ViolationEvent:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    vehicle_type: str
    dwell_time_seconds: float
    overlap_ratio: float


class RuleBasedViolationDetector:
    def __init__(self, dwell_threshold_seconds: float, overlap_threshold: float) -> None:
        self.dwell_threshold_seconds = dwell_threshold_seconds
        self.overlap_threshold = overlap_threshold

    def evaluate(
        self,
        tracks: list[Track],
        polygon: list[tuple[int, int]],
        frame_shape: tuple[int, ...],
        timestamp_seconds: float,
    ) -> list[ViolationEvent]:
        events: list[ViolationEvent] = []
        if len(polygon) < 3:
            return events

        zone_mask = polygon_mask(frame_shape, polygon)

        for track in tracks:
            overlap_ratio = bbox_polygon_overlap_ratio(track.bbox, zone_mask)
            in_zone = overlap_ratio >= self.overlap_threshold

            if in_zone and track.zone_entry_time is None:
                track.zone_entry_time = timestamp_seconds
            elif not in_zone:
                track.zone_entry_time = None

            dwell_time = (
                timestamp_seconds - track.zone_entry_time
                if track.zone_entry_time is not None
                else 0.0
            )

            if in_zone and dwell_time >= self.dwell_threshold_seconds and not track.violation_recorded:
                track.violation_recorded = True
                events.append(
                    ViolationEvent(
                        track_id=track.track_id,
                        bbox=track.bbox,
                        confidence=track.confidence,
                        vehicle_type=track.vehicle_type,
                        dwell_time_seconds=dwell_time,
                        overlap_ratio=overlap_ratio,
                    )
                )

        return events


def polygon_mask(frame_shape: tuple[int, ...], polygon: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask


def bbox_polygon_overlap_ratio(bbox: tuple[int, int, int, int], zone_mask: np.ndarray) -> float:
    x1, y1, x2, y2 = bbox
    height, width = zone_mask.shape
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    vehicle_area = float((x2 - x1) * (y2 - y1))
    overlap_area = float(cv2.countNonZero(zone_mask[y1:y2, x1:x2]))
    return overlap_area / vehicle_area
