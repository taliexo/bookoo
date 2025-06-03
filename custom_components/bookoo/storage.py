"""Handles SQLite storage for Bookoo shot history."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DB_FILE_NAME = "bookoo_shots.db"
TABLE_SHOTS = "shots"


def _get_db_path(hass: HomeAssistant) -> Path:
    """Get the path to the SQLite database file."""
    return Path(hass.config.path(DB_FILE_NAME))


def _init_db_sync(db_path: Path) -> None:
    """Initialize the database and create tables if they don't exist (synchronous)."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {TABLE_SHOTS} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    duration_seconds REAL,
                    final_weight_grams REAL,
                    flow_profile_json TEXT,
                    scale_timer_profile_json TEXT,
                    input_parameters_json TEXT,
                    start_trigger TEXT,
                    stop_reason TEXT,
                    channeling_status TEXT,
                    pre_infusion_detected INTEGER, /* Boolean: 0 or 1 */
                    pre_infusion_duration_seconds REAL,
                    extraction_uniformity_metric REAL
                )
                """
            )
            conn.commit()
            _LOGGER.info("Bookoo shot database initialized at %s", db_path)
    except sqlite3.Error as e:
        _LOGGER.error("Error initializing Bookoo shot database: %s", e)
        raise


async def async_init_db(hass: HomeAssistant) -> None:
    """Initialize the database (asynchronous)."""
    db_path = _get_db_path(hass)
    await hass.async_add_executor_job(_init_db_sync, db_path)


def _add_shot_record_sync(db_path: Path, shot_data: dict[str, Any]) -> None:
    """Add a shot record to the database (synchronous)."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Prepare data for insertion, ensuring all keys from the schema are present
            # or have defaults (None for optional fields).
            record = {
                "timestamp_utc": shot_data.get("timestamp_utc"),
                "duration_seconds": shot_data.get("duration_seconds"),
                "final_weight_grams": shot_data.get("final_weight_grams"),
                "flow_profile_json": json.dumps(shot_data.get("flow_profile"))
                if shot_data.get("flow_profile") is not None
                else None,
                "scale_timer_profile_json": json.dumps(
                    shot_data.get("scale_timer_profile")
                )
                if shot_data.get("scale_timer_profile") is not None
                else None,
                "input_parameters_json": json.dumps(shot_data.get("input_parameters"))
                if shot_data.get("input_parameters") is not None
                else None,
                "start_trigger": shot_data.get("start_trigger"),
                "stop_reason": shot_data.get("stop_reason"),
                "channeling_status": shot_data.get("channeling_status"),  # Part 2
                "pre_infusion_detected": shot_data.get(
                    "pre_infusion_detected"
                ),  # Part 2
                "pre_infusion_duration_seconds": shot_data.get(
                    "pre_infusion_duration_seconds"
                ),  # Part 2
                "extraction_uniformity_metric": shot_data.get(
                    "extraction_uniformity_metric"
                ),  # Part 2
            }

            columns = ", ".join(record.keys())
            placeholders = ":" + ", :".join(record.keys())
            sql = f"INSERT INTO {TABLE_SHOTS} ({columns}) VALUES ({placeholders})"

            cursor.execute(sql, record)
            conn.commit()
            _LOGGER.debug(
                "Bookoo shot record added: %s", shot_data.get("timestamp_utc")
            )
    except sqlite3.Error as e:
        _LOGGER.error("Error adding Bookoo shot record: %s", e)
        # Not raising here to avoid crashing the coordinator if DB write fails
    except json.JSONDecodeError as e:
        _LOGGER.error("Error serializing shot data to JSON: %s", e)


async def async_add_shot_record(hass: HomeAssistant, shot_data: dict[str, Any]) -> None:
    """Add a shot record to the database (asynchronous)."""
    db_path = _get_db_path(hass)
    await hass.async_add_executor_job(_add_shot_record_sync, db_path, shot_data)
