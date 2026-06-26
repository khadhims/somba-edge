import json
import sqlite3
from datetime import datetime, timezone


def round_duration_minutes(duration_minutes: float) -> float:
    return round(float(duration_minutes), 1)


class EdgeStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS master_cameras (
                camera_uuid TEXT PRIMARY KEY,
                name TEXT,
                rtsp_url TEXT,
                stream_url TEXT,
                activity TEXT NOT NULL,
                alert INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            """
        )
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
        if "activity_uid" in columns and "activity_type" not in columns:
            cursor.execute(
                "ALTER TABLE recording_events RENAME COLUMN activity_uid TO activity_type"
            )
        elif "activity_type" not in columns:
            cursor.execute(
                "ALTER TABLE recording_events ADD COLUMN activity_type TEXT"
            )

    def upsert_master_cameras(self, rows: list[dict]):
        if not rows:
            return
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for row in rows:
            cursor.execute(
                """
                INSERT INTO master_cameras (
                    camera_uuid, name, rtsp_url, stream_url, activity, alert, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_uuid) DO UPDATE SET
                    name = excluded.name,
                    rtsp_url = excluded.rtsp_url,
                    stream_url = excluded.stream_url,
                    activity = excluded.activity,
                    alert = excluded.alert,
                    updated_at = excluded.updated_at
                """,
                (
                    row["camera_uuid"],
                    row.get("name"),
                    row.get("rtsp_url"),
                    row.get("stream_url"),
                    row["activity"],
                    1 if row.get("alert") else 0,
                    now,
                ),
            )
        conn.commit()
        conn.close()

    def delete_stale_master_cameras(self, active_uuids: list[str]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if not active_uuids:
            cursor.execute("DELETE FROM master_cameras")
        else:
            placeholders = ",".join("?" for _ in active_uuids)
            cursor.execute(
                f"DELETE FROM master_cameras WHERE camera_uuid NOT IN ({placeholders})",
                active_uuids,
            )
        conn.commit()
        conn.close()

    def list_master_cameras(self) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT camera_uuid, name, rtsp_url, stream_url, activity, alert
            FROM master_cameras
            ORDER BY name ASC
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "camera_uuid": row[0],
                "name": row[1],
                "rtsp_url": row[2],
                "stream_url": row[3],
                "activity": row[4],
                "alert": bool(row[5]),
            }
            for row in rows
        ]

    def save_alert(
        self,
        camera_id: str,
        violation_name: str,
        severity: str,
        bbox: list,
        image_url: str,
        detected_at: str,
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO alerts (
                camera_id, violation_name, severity, detected_at,
                bbox, image_url, sync_status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            (
                camera_id,
                violation_name,
                severity,
                detected_at,
                json.dumps(bbox),
                image_url,
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
            SELECT id, camera_id, violation_name, severity, detected_at, bbox, image_url
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
        activity_type: str,
        event_start: str,
        event_end: str,
        duration_minutes: float,
        recording_url: str,
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        normalized_duration = round_duration_minutes(duration_minutes)
        cursor.execute(
            """
            INSERT INTO recording_events (
                camera_id, activity_type, event_start, event_end,
                duration_minutes, recording_url, sync_status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            (
                camera_id,
                activity_type,
                event_start,
                event_end,
                normalized_duration,
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
            SELECT id, camera_id, activity_type, event_start, event_end, duration_minutes, recording_url
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
                "activity_type": row[2],
                "event_start": row[3],
                "event_end": row[4],
                "duration_minutes": round_duration_minutes(row[5]),
                "recording_url": row[6],
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
