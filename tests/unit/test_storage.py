# tests/unit/test_storage.py
import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from custom_components.bookoo.storage import (
    async_add_shot_record,
    async_get_shot_history,
    async_delete_shot_history,
)
from custom_components.bookoo.types import BookooShotCompletedEventDataModel

# Sample data for BookooShotCompletedEventDataModel
SAMPLE_SHOT_DATA_MODEL = BookooShotCompletedEventDataModel(
    device_id="test_device",
    entry_id="test_entry",
    start_time_utc=datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
    end_time_utc=datetime(2023, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
    duration_seconds=30.0,
    final_weight_grams=36.0,
    flow_profile=[(1.0, 0.5), (2.0, 1.5)],  # Using tuples as per NamedTuple in plan
    scale_timer_profile=[(1.0, 1), (2.0, 2)],
    input_parameters={"grind": "fine"},
    start_trigger="manual",
    stop_reason="manual",
    status="completed",
    channeling_status="None",
    pre_infusion_detected=False,
    pre_infusion_duration_seconds=None,
    extraction_uniformity_metric=0.9,
    average_flow_rate_gps=1.2,
    peak_flow_rate_gps=1.5,
    time_to_first_flow_seconds=5.0,
    time_to_peak_flow_seconds=15.0,
    shot_quality_score=90.0,
)

SAMPLE_SHOT_DATA_DICT = SAMPLE_SHOT_DATA_MODEL.model_dump(mode="json")


@pytest.fixture
def mock_hass() -> MagicMock:
    """Fixture for a mock HomeAssistant instance."""
    return MagicMock(spec=HomeAssistant)


@pytest.fixture
def mock_store_instance() -> MagicMock:
    """Fixture for a mock Store instance."""
    store = MagicMock(spec=Store)
    store.async_load = AsyncMock(return_value=None)  # Default to empty store
    store.async_save = AsyncMock()
    store.async_remove = AsyncMock()
    return store


@pytest.fixture(autouse=True)
def mock_get_store(mock_store_instance: MagicMock):
    """Autouse fixture to patch _get_store to return our mock_store_instance."""
    with patch(
        "custom_components.bookoo.storage._get_store", return_value=mock_store_instance
    ) as mock:
        yield mock


# --- Tests for async_add_shot_record ---
async def test_async_add_shot_record_empty_history(
    mock_hass: MagicMock, mock_store_instance: MagicMock, mock_get_store: MagicMock
):
    """Test adding a shot record when history is initially empty."""
    mock_store_instance.async_load.return_value = []

    await async_add_shot_record(mock_hass, SAMPLE_SHOT_DATA_MODEL)

    mock_get_store.assert_called_once_with(mock_hass)
    mock_store_instance.async_load.assert_called_once()
    mock_store_instance.async_save.assert_called_once_with([SAMPLE_SHOT_DATA_DICT])


async def test_async_add_shot_record_existing_history(
    mock_hass: MagicMock, mock_store_instance: MagicMock
):
    """Test adding a shot record when history has existing data."""
    existing_record = {
        "device_id": "old_device",
        "duration_seconds": 25.0,
    }  # Simplified
    mock_store_instance.async_load.return_value = [existing_record]

    await async_add_shot_record(mock_hass, SAMPLE_SHOT_DATA_MODEL)

    mock_store_instance.async_save.assert_called_once_with(
        [existing_record, SAMPLE_SHOT_DATA_DICT]
    )


async def test_async_add_shot_record_load_failure(
    mock_hass: MagicMock,
    mock_store_instance: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test adding a shot record when store.async_load fails."""
    mock_store_instance.async_load.side_effect = Exception("Failed to load")
    caplog.set_level(logging.ERROR)

    await async_add_shot_record(mock_hass, SAMPLE_SHOT_DATA_MODEL)

    assert "Error loading shot history" in caplog.text
    # Should still attempt to save the new record in a new history list
    mock_store_instance.async_save.assert_called_once_with([SAMPLE_SHOT_DATA_DICT])


async def test_async_add_shot_record_save_failure(
    mock_hass: MagicMock,
    mock_store_instance: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test adding a shot record when store.async_save fails."""
    mock_store_instance.async_load.return_value = []
    mock_store_instance.async_save.side_effect = Exception("Failed to save")
    caplog.set_level(logging.ERROR)

    await async_add_shot_record(mock_hass, SAMPLE_SHOT_DATA_MODEL)

    assert "Error saving shot record" in caplog.text


# --- Tests for async_get_shot_history ---
async def test_async_get_shot_history_empty(
    mock_hass: MagicMock, mock_store_instance: MagicMock, mock_get_store: MagicMock
):
    """Test getting shot history when it's empty."""
    mock_store_instance.async_load.return_value = (
        None  # Store returns None if file doesn't exist
    )

    result = await async_get_shot_history(mock_hass)

    mock_get_store.assert_called_once_with(mock_hass)
    mock_store_instance.async_load.assert_called_once()
    assert result == []


async def test_async_get_shot_history_with_valid_data(
    mock_hass: MagicMock, mock_store_instance: MagicMock
):
    """Test getting shot history with valid data."""
    mock_store_instance.async_load.return_value = [
        SAMPLE_SHOT_DATA_DICT,
        SAMPLE_SHOT_DATA_DICT,
    ]

    result = await async_get_shot_history(mock_hass)

    assert len(result) == 2
    assert result[0] == SAMPLE_SHOT_DATA_MODEL
    assert result[1] == SAMPLE_SHOT_DATA_MODEL


async def test_async_get_shot_history_with_invalid_data(
    mock_hass: MagicMock,
    mock_store_instance: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test getting shot history with some invalid data."""
    invalid_record = {"device_id": "invalid", "some_unexpected_field": True}
    mock_store_instance.async_load.return_value = [
        SAMPLE_SHOT_DATA_DICT,
        invalid_record,
    ]
    caplog.set_level(logging.WARNING)

    result = await async_get_shot_history(mock_hass)

    assert len(result) == 1
    assert result[0] == SAMPLE_SHOT_DATA_MODEL
    assert "Skipping invalid shot data" in caplog.text


async def test_async_get_shot_history_with_limit(
    mock_hass: MagicMock, mock_store_instance: MagicMock
):
    """Test getting shot history with a limit."""
    records = [SAMPLE_SHOT_DATA_DICT.copy() for _ in range(5)]
    for i, record in enumerate(records):
        # Make them slightly different for distinction if needed, e.g., by start_time_utc
        # For this test, just ensuring the list slicing works is enough.
        record["duration_seconds"] = 30 + i

    mock_store_instance.async_load.return_value = records

    result = await async_get_shot_history(mock_hass, limit=3)

    assert len(result) == 3
    # Pydantic models will be created, check one field from the last 3
    assert result[0].duration_seconds == 30 + 2  # records[-3]['duration_seconds']
    assert result[1].duration_seconds == 30 + 3  # records[-2]['duration_seconds']
    assert result[2].duration_seconds == 30 + 4  # records[-1]['duration_seconds']


async def test_async_get_shot_history_load_failure(
    mock_hass: MagicMock,
    mock_store_instance: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test getting shot history when store.async_load fails."""
    mock_store_instance.async_load.side_effect = Exception("Failed to load")
    caplog.set_level(logging.ERROR)

    result = await async_get_shot_history(mock_hass)

    assert "Error loading shot history" in caplog.text
    assert result == []


# --- Tests for async_delete_shot_history ---
async def test_async_delete_shot_history_success(
    mock_hass: MagicMock, mock_store_instance: MagicMock, mock_get_store: MagicMock
):
    """Test deleting shot history successfully."""
    await async_delete_shot_history(mock_hass)

    mock_get_store.assert_called_once_with(mock_hass)
    mock_store_instance.async_remove.assert_called_once()


async def test_async_delete_shot_history_failure(
    mock_hass: MagicMock,
    mock_store_instance: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test deleting shot history when store.async_remove fails."""
    mock_store_instance.async_remove.side_effect = Exception("Failed to remove")
    caplog.set_level(logging.ERROR)

    await async_delete_shot_history(mock_hass)

    assert "Error deleting Bookoo shot history" in caplog.text
