# tests/e2e/test_bluetooth_reconnection.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from custom_components.bookoo.const import DOMAIN, EVENT_BOOKOO_SHOT_COMPLETED
from aiobookoov2.const import UPDATE_SOURCE_WEIGHT_CHAR
from custom_components.bookoo.coordinator import BookooCoordinator
from aiobookoov2.exceptions import BookooDeviceNotFound

# Assuming conftest.py provides init_integration and mock_bookoo_scale


async def trigger_coordinator_update(
    hass: HomeAssistant, coordinator: DataUpdateCoordinator
):
    """Helper to manually trigger a coordinator update and wait for it to complete."""
    await coordinator.async_refresh()
    await hass.async_block_till_done()


@pytest.mark.asyncio
async def test_reconnection_during_idle_state(
    hass: HomeAssistant,
    init_integration: ConfigEntry,  # This fixture sets up the integration and returns the config entry
    mock_bookoo_scale: MagicMock,  # The MagicMock for the BookooScale instance from conftest
):
    """Test that the coordinator can reconnect if the connection drops while idle."""
    coordinator = hass.data[DOMAIN][init_integration.entry_id]
    assert isinstance(coordinator, BookooCoordinator)

    # 1. Initial state: Ensure scale is connected
    # The init_integration fixture should ensure this, but let's verify
    assert mock_bookoo_scale.connected is True
    assert coordinator.last_update_success is True

    # 2. Simulate a disconnection
    mock_bookoo_scale.connected = False
    # Make the connect method fail a couple of times before succeeding
    # to simulate retry logic if the coordinator implements it directly or via _async_update_data exceptions
    side_effects = [
        BookooDeviceNotFound("Simulated disconnect - attempt 1"),
        BookooDeviceNotFound("Simulated disconnect - attempt 2"),
        AsyncMock(return_value=None),  # Successful connection on the third attempt
    ]

    original_connect = mock_bookoo_scale.connect
    mock_bookoo_scale.connect = AsyncMock(side_effect=side_effects)

    # This function will be called by connect to simulate success
    async def mock_successful_connect_effect(*args, **kwargs):
        mock_bookoo_scale.connected = True
        # If the original connect was more complex, replicate its essential success actions here
        # For this mock, just setting connected = True is enough for the coordinator's check.
        # If the real connect method returns something, mock that too.
        return None

    # Update side_effects: the successful connect should call our mock_successful_connect_effect
    side_effects[2] = AsyncMock(side_effect=mock_successful_connect_effect)
    mock_bookoo_scale.connect.side_effect = (
        side_effects  # Re-assign with the updated successful mock
    )

    # 3. Trigger coordinator updates to make it attempt reconnection
    # The coordinator's _async_update_data will try to connect.
    # We expect it to fail twice then succeed.

    # First attempt (should fail and raise UpdateFailed, caught by coordinator)
    await trigger_coordinator_update(hass, coordinator)
    assert mock_bookoo_scale.connected is False
    assert coordinator.last_update_success is False  # Expected as connect fails
    assert mock_bookoo_scale.connect.call_count == 1

    # Second attempt (should also fail)
    await trigger_coordinator_update(hass, coordinator)
    assert mock_bookoo_scale.connected is False
    assert coordinator.last_update_success is False
    assert mock_bookoo_scale.connect.call_count == 2

    # Third attempt (should succeed)
    await trigger_coordinator_update(hass, coordinator)
    assert mock_bookoo_scale.connected is True
    assert coordinator.last_update_success is True
    assert mock_bookoo_scale.connect.call_count == 3

    # Restore original connect if necessary for other tests, though pytest fixtures usually isolate
    mock_bookoo_scale.connect = original_connect


@pytest.mark.asyncio
async def test_reconnection_during_active_shot(
    hass: HomeAssistant,
    init_integration: ConfigEntry,
    mock_bookoo_scale: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test reconnection handling when a shot is active and connection drops."""
    coordinator = hass.data[DOMAIN][init_integration.entry_id]
    assert isinstance(coordinator, BookooCoordinator)

    # Setup to capture EVENT_BOOKOO_SHOT_COMPLETED
    completed_shot_events = []

    @callback
    def capture_event(event):
        completed_shot_events.append(event)

    hass.bus.async_listen(EVENT_BOOKOO_SHOT_COMPLETED, capture_event)

    # 1. Start a shot and send some initial data
    await hass.services.async_call(DOMAIN, "start_shot", blocking=True)
    await hass.async_block_till_done()
    assert coordinator.session_manager.is_shot_active is True

    for i in range(5):  # Simulate 5 seconds of data
        mock_bookoo_scale.weight = i * 1.0
        mock_bookoo_scale.flow_rate = 0.5 + (i * 0.1)
        mock_bookoo_scale.timer = float(i)
        coordinator._handle_characteristic_update(UPDATE_SOURCE_WEIGHT_CHAR, None)
        await hass.async_block_till_done()
        await asyncio.sleep(0.01)

    # 2. Simulate a disconnection during the shot
    mock_bookoo_scale.connected = False
    connect_attempts = 0
    max_fail_attempts = 2

    async def mock_connect_during_shot(*args, **kwargs):
        nonlocal connect_attempts
        connect_attempts += 1
        if connect_attempts <= max_fail_attempts:
            raise BookooDeviceNotFound(
                f"Simulated disconnect during shot - attempt {connect_attempts}"
            )
        # Successful connection after failing twice
        mock_bookoo_scale.connected = True
        return None

    original_connect = mock_bookoo_scale.connect
    mock_bookoo_scale.connect = AsyncMock(side_effect=mock_connect_during_shot)

    # 3. Trigger coordinator update - this should detect disconnection and try to reconnect
    # The _async_update_data method will call _ensure_scale_connected_and_processing,
    # which will then call _handle_active_shot_disconnection if an error occurs.
    with patch.object(
        coordinator.session_manager,
        "stop_session",
        wraps=coordinator.session_manager.stop_session,
    ) as mock_stop_session:
        await trigger_coordinator_update(hass, coordinator)  # First failed attempt
        assert coordinator.last_update_success is False
        # Session should be stopped due to connection loss during active shot
        mock_stop_session.assert_called_once_with(
            stop_reason="connection_lost_during_shot"
        )
        assert coordinator.session_manager.is_shot_active is False

    # Verify shot completed event was fired with an error status
    assert len(completed_shot_events) == 1
    event_data = completed_shot_events[0].data
    # The status might depend on exact implementation: "error_disconnected", "aborted", etc.
    # Let's assume it's "error_disconnected" for now, or check for a non-"completed" status.
    assert event_data["status"] != "completed"
    assert "connection_lost_during_shot" in event_data["stop_reason"]

    # 4. Further attempts to update should still fail until connect succeeds
    await trigger_coordinator_update(hass, coordinator)  # Second failed attempt
    assert mock_bookoo_scale.connected is False
    assert coordinator.last_update_success is False

    # 5. Successful reconnection attempt
    await trigger_coordinator_update(hass, coordinator)  # Third attempt, should succeed
    assert mock_bookoo_scale.connected is True
    assert coordinator.last_update_success is True
    assert connect_attempts == max_fail_attempts + 1

    # 6. Verify a new shot can be started
    completed_shot_events.clear()  # Clear previous events
    await hass.services.async_call(DOMAIN, "start_shot", blocking=True)
    await hass.async_block_till_done()
    assert coordinator.session_manager.is_shot_active is True
    # ... (optionally, send more data and complete this new shot to be thorough)

    # Restore original connect
    mock_bookoo_scale.connect = original_connect


@pytest.mark.asyncio
async def test_reconnection_failure_permanent(
    hass: HomeAssistant,
    init_integration: ConfigEntry,
    mock_bookoo_scale: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test coordinator behavior with permanent reconnection failure."""
    coordinator = hass.data[DOMAIN][init_integration.entry_id]
    assert isinstance(coordinator, BookooCoordinator)

    # 1. Initial state: Ensure scale is connected
    assert mock_bookoo_scale.connected is True
    assert coordinator.last_update_success is True

    # 2. Simulate a disconnection and make connect always fail
    mock_bookoo_scale.connected = False
    original_connect = mock_bookoo_scale.connect
    mock_bookoo_scale.connect = AsyncMock(
        side_effect=BookooDeviceNotFound("Simulated permanent failure")
    )

    # 3. Trigger coordinator updates multiple times
    num_attempts = 3
    for attempt in range(1, num_attempts + 1):
        caplog.clear()  # Clear logs for each attempt to check specific logs for that attempt
        await trigger_coordinator_update(hass, coordinator)
        assert coordinator.last_update_success is False, (
            f"Update should fail on attempt {attempt}"
        )
        assert mock_bookoo_scale.connected is False, (
            f"Scale should remain disconnected on attempt {attempt}"
        )
        # Check that the specific error from _async_update_data's exception handling is logged
        assert (
            "Bookoo scale device not found" in caplog.text
            or "Failed to connect to Bookoo scale after multiple retries" in caplog.text
            or "UpdateFailed: Error communicating with Bookoo scale" in caplog.text
        )

    assert mock_bookoo_scale.connect.call_count == num_attempts

    # 4. Verify coordinator and scale state after multiple failures
    assert coordinator.last_update_success is False
    assert mock_bookoo_scale.connected is False
    # Entities should ideally reflect unavailability (actual check depends on entity implementation)

    # Restore original connect
    mock_bookoo_scale.connect = original_connect
