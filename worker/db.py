import json
import sqlite3
from datetime import datetime


class EdgeStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                violation_name TEXT,
                severity TEXT,
                detected_at DATETIME,
                bbox TEXT,
                image_url TEXT,
                recording_event_id TEXT,
                sync_status TEXT DEFAULT 'PENDING'
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS recording_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                activity_type TEXT,
                event_start DATETIME,
                event_end DATETIME,
                duration_minutes REAL,
                recording_url TEXT,
                sync_status TEXT DEFAULT 'PENDING'
            )
            """
        )
        self._ensure_recording_event_columns(cursor)
        conn.commit()
        conn.close()

    def _ensure_recording_event_columns(self, cursor):
        cursor.execute("PRAGMA table_info(recording_events)")
        columns = {row[1] for row in cursor.fetchall()}
        if "activity_uid" not in columns:
            cursor.execute(
                "ALTER TABLE recording_events ADD COLUMN activity_uid TEXT"
            )

    def save_alert(
        self,
        camera_id: str,
        violation_name: str,
        severity: str,
        bbox: list,
        image_url: str,
        detected_at: str,
        recording_event_id: str | None = None,
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO alerts (
                camera_id, violation_name, severity, detected_at,
                bbox, image_url, recording_event_id, sync_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            (
                camera_id,
                violation_name,
                severity,
                detected_at,
                json.dumps(bbox),
                image_url,
                recording_event_id,
            ),
        )
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return alert_id

    def fetch_pending_alerts(self, batch_size: int) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, camera_id, violation_name, severity, detected_at, bbox, image_url, recording_event_id
            FROM alerts
            WHERE sync_status = 'PENDING'
            ORDER BY id ASC
            LIMIT ?
            """,
            (batch_size,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "edge_id": row[0],
                "camera_uuid": row[1],
                "violation_name": row[2],
                "severity": row[3],
                "detected_at": row[4],
                "bbox": json.loads(row[5]),
                "image_url": row[6],
                "recording_event_id": row[7],
            }
            for row in rows
        ]

    def mark_alerts_sent(self, alert_ids: list[int]):
        if not alert_ids:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE alerts SET sync_status = 'SENT' WHERE id = ?",
            [(alert_id,) for alert_id in alert_ids],
        )
        conn.commit()
        conn.close()

    def save_recording_event(
        self,
        camera_id: str,
        activity_uid: str,
        activity_type: str,
        event_start: str,
        event_end: str,
        duration_minutes: float,
        recording_url: str,
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recording_events (
                camera_id, activity_uid, activity_type, event_start, event_end,
                duration_minutes, recording_url, sync_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            (
                camera_id,
                activity_uid,
                activity_type,
                event_start,
                event_end,
                duration_minutes,
                recording_url,
            ),
        )
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return event_id

    def fetch_pending_recording_events(self, batch_size: int) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, camera_id, activity_uid, activity_type, event_start, event_end, duration_minutes, recording_url
            FROM recording_events
            WHERE sync_status = 'PENDING'
            ORDER BY id ASC
            LIMIT ?
            """,
            (batch_size,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "edge_id": row[0],
                "camera_uuid": row[1],
                "activity_uid": row[2],
                "activity_type": row[3],
                "event_start": row[4],
                "event_end": row[5],
                "duration_minutes": row[6],
                "recording_url": row[7],
            }
            for row in rows
        ]

    def mark_recording_events_sent(self, event_ids: list[int]):
        if not event_ids:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany(
            "UPDATE recording_events SET sync_status = 'SENT' WHERE id = ?",
            [(event_id,) for event_id in event_ids],
        )
        conn.commit()
        conn.close()
