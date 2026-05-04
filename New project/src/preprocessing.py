import cv2
import numpy as np


def apply_preprocessing(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a Gaussian-filtered frame and Canny edge map."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    return blurred, edges


def blend_edges(frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 0.82, edge_bgr, 0.18, 0)
