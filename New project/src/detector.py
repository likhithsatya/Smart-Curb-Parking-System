from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.config import VEHICLE_CLASSES


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    vehicle_type: str


class VehicleDetector:
    """YOLO vehicle detector wrapper.

    The detector accepts OpenCV BGR frames. Ultralytics handles conversion
    internally, so callers can pass frames directly from cv2.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.35) -> None:
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = str(names[class_id])
                if label not in VEHICLE_CLASSES:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        vehicle_type=label,
                    )
                )

        return detections
