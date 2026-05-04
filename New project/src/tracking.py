from dataclasses import dataclass, field
from math import hypot

from src.detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    vehicle_type: str
    first_seen: float
    last_seen: float
    missed_frames: int = 0
    violation_recorded: bool = False
    zone_entry_time: float | None = None

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class CentroidTracker:
    max_distance: float = 80.0
    max_missed_frames: int = 20
    next_id: int = 1
    tracks: dict[int, Track] = field(default_factory=dict)

    def update(self, detections: list[Detection], timestamp_seconds: float) -> list[Track]:
        unmatched_track_ids = set(self.tracks.keys())
        updated_tracks: list[Track] = []

        for detection in detections:
            det_centroid = _centroid(detection.bbox)
            best_track_id = None
            best_distance = self.max_distance

            for track_id in list(unmatched_track_ids):
                distance = hypot(
                    det_centroid[0] - self.tracks[track_id].centroid[0],
                    det_centroid[1] - self.tracks[track_id].centroid[1],
                )
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                track = Track(
                    track_id=self.next_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    vehicle_type=detection.vehicle_type,
                    first_seen=timestamp_seconds,
                    last_seen=timestamp_seconds,
                )
                self.tracks[self.next_id] = track
                self.next_id += 1
            else:
                track = self.tracks[best_track_id]
                track.bbox = detection.bbox
                track.confidence = detection.confidence
                track.vehicle_type = detection.vehicle_type
                track.last_seen = timestamp_seconds
                track.missed_frames = 0
                unmatched_track_ids.remove(best_track_id)

            updated_tracks.append(track)

        for track_id in unmatched_track_ids:
            self.tracks[track_id].missed_frames += 1

        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]
        for track_id in stale_ids:
            del self.tracks[track_id]

        return updated_tracks


def _centroid(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)
