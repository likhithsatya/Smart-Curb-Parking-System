import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


class ViolationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    frame_time_seconds REAL NOT NULL,
                    frame_image_path TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    dwell_time_seconds REAL NOT NULL,
                    overlap_ratio REAL NOT NULL,
                    bbox TEXT NOT NULL
                )
                """
            )

    def add_violation(
        self,
        source_name: str,
        frame_time_seconds: float,
        frame_image_path: str,
        confidence: float,
        vehicle_type: str,
        track_id: int,
        dwell_time_seconds: float,
        overlap_ratio: float,
        bbox: tuple[int, int, int, int],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO violations (
                    created_at, source_name, frame_time_seconds, frame_image_path,
                    confidence, vehicle_type, track_id, dwell_time_seconds,
                    overlap_ratio, bbox
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    source_name,
                    frame_time_seconds,
                    frame_image_path,
                    confidence,
                    vehicle_type,
                    track_id,
                    dwell_time_seconds,
                    overlap_ratio,
                    ",".join(str(value) for value in bbox),
                ),
            )

    def list_violations(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM violations ORDER BY id DESC",
                conn,
            )

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM violations")
