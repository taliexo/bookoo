# tests/integration/test_coordinator.py
import pytest
from unittest.mock import patch, MagicMock, ANY

from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from custom_components.bookoo.coordinator import BookooCoordinator
from custom_components.bookoo.const import DOMAIN, BookooConfig
from homeassistant.const import CONF_ADDRESS
from custom_components.bookoo.session_manager import SessionManager
from custom_components.bookoo.analytics import ShotAnalyzer
from homeassistant.exceptions import HomeAssistantError
import asyncio
from homeassistant.helpers.update_coordinator import UpdateFailed
from aiobookoov2.exceptions import BookooDeviceNotFound, BookooError
from aiobookoov2.exceptions import (
    BookooError as AIOBookooError,
)  # For parametrizing with library exceptions if needed, or ensure coordinator wraps them.
from aiobookoov2.const import UPDATE_SOURCE_COMMAND_CHAR, UPDATE_SOURCE_WEIGHT_CHAR
from datetime import datetime, timezone, timedelta

# From conftest.py, we get init_integration which provides hass, mock_config_entry, mock_bookoo_scale
# MOCK_DEVICE_INFO and MOCK_SERVICE_INFO are also available from conftest.py if needed directly


@pytest.fixture
def mock_session_manager_instance():
    """Provides a mock instance of SessionManager."""
    instance = MagicMock(spec=SessionManager)
    instance.is_shot_active = False  # Default state
    return instance


@pytest.fixture
def mock_shot_analyzer_instance():
    """Provides a mock instance of ShotAnalyzer."""
    return MagicMock(spec=ShotAnalyzer)


@pytest.fixture(autouse=True)
def mock_session_manager_class(mock_session_manager_instance):
    """Patches the SessionManager class to return a mock instance."""
    with patch(
        "custom_components.bookoo.coordinator.SessionManager",
        return_value=mock_session_manager_instance,
    ) as mock_class:
        yield mock_class


@pytest.fixture(autouse=True)
def mock_shot_analyzer_class(mock_shot_analyzer_instance):
    """Patches the ShotAnalyzer class to return a mock instance."""
    with patch(
        "custom_components.bookoo.coordinator.ShotAnalyzer",
        return_value=mock_shot_analyzer_instance,
    ) as mock_class:
        yield mock_class


@pytest.fixture
def coordinator(
    hass: HomeAssistant, init_integration: ConfigEntry
) -> BookooCoordinator:
    """Fixture to get the coordinator instance after init_integration has run."""
    return hass.data[DOMAIN][init_integration.entry_id]


async def test_coordinator_initialization(
    hass: HomeAssistant,
    init_integration: ConfigEntry,  # Runs conftest.py's init_integration
    coordinator: BookooCoordinator,  # Our coordinator fixture
    mock_bookoo_scale: MagicMock,  # From conftest.py
    mock_session_manager_class: MagicMock,  # Patched class
    mock_shot_analyzer_class: MagicMock,  # Patched class
    mock_session_manager_instance: MagicMock,  # Instance returned by patched class
    mock_shot_analyzer_instance: MagicMock,  # Instance returned by patched class
):
    """Test the initialization of the BookooCoordinator."""
    entry = init_integration  # alias for clarity

    # 1. Verify BookooScale is initialized correctly within the coordinator
    # The actual BookooScale mock is mock_bookoo_scale, but the coordinator creates its own instance.
    # We need to assert that the BookooScale class (which is patched in conftest.py to return mock_bookoo_scale)
    # was called correctly by the coordinator during its __init__.
    # The patch is at 'custom_components.bookoo.coordinator.BookooScale'
    with patch(
        "custom_components.bookoo.coordinator.BookooScale",
        return_value=mock_bookoo_scale,
    ) as patched_bookoo_scale_class:
        # Re-trigger coordinator creation by re-running parts of init_integration or a new setup
        # For simplicity, we'll assume init_integration already did its job and the coordinator exists.
        # We'll check the call that *would* have happened if we could intercept it *during* init.
        # This is a bit indirect. A better way might be to check properties of coordinator.scale if they reflect init params.

        # Let's re-initialize a coordinator instance to check the call to BookooScale constructor
        # This is for assertion purposes only and won't affect the 'coordinator' fixture instance directly for this test's scope
        # unless we re-fetch it from hass.data after this.
        temp_coordinator = BookooCoordinator(hass, entry)

        patched_bookoo_scale_class.assert_called_once_with(
            address_or_ble_device=entry.data[CONF_ADDRESS],
            name=entry.title,
            is_valid_scale=entry.data.get(
                "is_valid_scale", False
            ),  # from conftest's mock_config_entry data
            notify_callback=temp_coordinator.async_update_listeners,  # or ANY if checking instance method is tricky
            characteristic_update_callback=temp_coordinator._handle_characteristic_update,  # or ANY
        )
        assert (
            coordinator.scale == mock_bookoo_scale
        )  # Check if the instance used is the one from conftest

    # 2. Verify SessionManager and ShotAnalyzer were instantiated
    # mock_session_manager_class is the patched class
    # mock_session_manager_instance is the instance it returned
    mock_session_manager_class.assert_called_once_with(hass, coordinator)
    assert coordinator.session_manager == mock_session_manager_instance

    mock_shot_analyzer_class.assert_called_once_with()  # No args for ShotAnalyzer constructor
    assert coordinator.shot_analyzer == mock_shot_analyzer_instance

    # 3. Verify BookooConfig is loaded correctly
    assert isinstance(coordinator.bookoo_config, BookooConfig)
    # Check a sample option to ensure it's loaded from the entry's options
    # (conftest.py's mock_config_entry should have some default options)
    expected_config = BookooConfig.from_config_entry(entry)
    assert (
        coordinator.bookoo_config.min_shot_duration == expected_config.min_shot_duration
    )
    assert coordinator.bookoo_config.connect_timeout == expected_config.connect_timeout

    # 4. Verify options update listener is added
    # The mock_config_entry is from conftest.py
    entry.add_update_listener.assert_called_once_with(
        coordinator._options_update_callback
    )
    assert coordinator._options_update_listener is not None

    # Verify logger name and update interval (optional, but good for completeness)
    assert coordinator.name == f"Bookoo {mock_bookoo_scale.mac or entry.title}"
    # SCAN_INTERVAL is used, check if coordinator.update_interval matches
    # from custom_components.bookoo.coordinator import SCAN_INTERVAL
    # assert coordinator.update_interval == SCAN_INTERVAL # This is set in DataUpdateCoordinator's super().__init__


# --- Tests for _async_update_data ---


@pytest.mark.asyncio
async def test_async_update_data_success(
    coordinator: BookooCoordinator,
    mock_bookoo_scale: MagicMock,  # Used by _ensure_scale_connected_and_processing
):
    """Test _async_update_data successfully calls _ensure_scale_connected_and_processing."""
    # Mock the underlying methods that _ensure_scale_connected_and_processing calls
    mock_bookoo_scale.connected = False  # Start with scale not connected
    mock_bookoo_scale.connect = AsyncMock(return_value=None)
    mock_bookoo_scale.process_queue_task = None  # Ensure it tries to start it

    # Make connect set connected to True
    async def _mock_connect(*args, **kwargs):
        mock_bookoo_scale.connected = True

    mock_bookoo_scale.connect.side_effect = _mock_connect

    # Mock _ensure_queue_processor_running to avoid deeper complexities for this specific test
    with patch.object(
        coordinator, "_ensure_queue_processor_running"
    ) as mock_ensure_queue_processing:
        await coordinator._async_update_data()
        mock_bookoo_scale.connect.assert_called_once()
        mock_ensure_queue_processing.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised_exception, expected_translation_key, expect_stop_session",
    [
        (BookooDeviceNotFound("Device not found"), "device_not_found", True),
        (BookooError("Scale communication error"), "cannot_connect", True),
        (asyncio.TimeoutError("Connection timeout"), "connection_timeout", True),
        (Exception("Some generic error"), "unexpected_update_error", True),
    ],
)
async def test_async_update_data_failure_scenarios(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    raised_exception: Exception,
    expected_translation_key: str,
    expect_stop_session: bool,
    caplog: pytest.LogCaptureFixture,
):
    """Test _async_update_data failure scenarios and that UpdateFailed is raised."""
    mock_session_manager_instance.is_shot_active = True  # Simulate active shot

    with patch.object(
        coordinator,
        "_ensure_scale_connected_and_processing",
        side_effect=raised_exception,
    ) as mock_ensure_scale_connected_and_processing:
        with pytest.raises(UpdateFailed) as exc_info:
            await coordinator._async_update_data()

    mock_ensure_scale_connected_and_processing.assert_called_once()
    assert expected_translation_key in str(
        exc_info.value.translation_key
    )  # Check if the key is part of the raised error
    if expect_stop_session:
        mock_session_manager_instance.stop_session.assert_called_once()
    else:
        mock_session_manager_instance.stop_session.assert_not_called()

    # Check logs for specific error messages if needed
    if isinstance(raised_exception, BookooDeviceNotFound):
        assert "Bookoo scale device not found" in caplog.text
    elif isinstance(raised_exception, BookooError):
        assert "Error communicating with Bookoo scale" in caplog.text
    elif isinstance(raised_exception, asyncio.TimeoutError):
        assert "Timeout connecting to Bookoo scale" in caplog.text
    elif isinstance(raised_exception, Exception):  # Generic exception
        assert "Unexpected error updating Bookoo scale data" in caplog.text


@pytest.mark.asyncio
async def test_async_update_data_failure_no_active_shot(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
):
    """Test _async_update_data failure when no shot is active."""
    mock_session_manager_instance.is_shot_active = False  # No active shot
    exception_to_raise = BookooError("Test Error")

    with patch.object(
        coordinator,
        "_ensure_scale_connected_and_processing",
        side_effect=exception_to_raise,
    ) as mock_ensure_scale_connected_and_processing:
        with pytest.raises(UpdateFailed):
            await coordinator._async_update_data()

    mock_ensure_scale_connected_and_processing.assert_called_once()
    mock_session_manager_instance.stop_session.assert_not_called()


# --- Tests for Service Call Handlers ---


@pytest.mark.asyncio
async def test_async_start_shot_service_success(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    mock_bookoo_scale: MagicMock,
    hass: HomeAssistant,  # For async_create_task
):
    """Test successful call to async_start_shot_service."""
    mock_session_manager_instance.is_shot_active = False
    mock_bookoo_scale.tare_and_start_timer = AsyncMock()
    # Mock _reset_realtime_analytics to check it's called
    with patch.object(coordinator, "_reset_realtime_analytics") as mock_reset_analytics:
        await coordinator.async_start_shot_service()

        # Allow tasks created by async_create_task to run
        await hass.async_block_till_done()

        mock_reset_analytics.assert_called_once()
        mock_session_manager_instance.start_session.assert_called_once_with(
            trigger="ha_service"
        )
        mock_bookoo_scale.tare_and_start_timer.assert_called_once()


@pytest.mark.asyncio
async def test_async_start_shot_service_already_active(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    mock_bookoo_scale: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test async_start_shot_service when a shot is already active."""
    mock_session_manager_instance.is_shot_active = True
    mock_bookoo_scale.tare_and_start_timer = AsyncMock()

    await coordinator.async_start_shot_service()

    assert "Start shot service called, but a shot is already active." in caplog.text
    mock_session_manager_instance.start_session.assert_not_called()
    mock_bookoo_scale.tare_and_start_timer.assert_not_called()


@pytest.mark.asyncio
async def test_async_start_shot_service_scale_command_failure(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    mock_bookoo_scale: MagicMock,
    hass: HomeAssistant,  # For async_create_task
):
    """Test async_start_shot_service when the scale command fails."""
    mock_session_manager_instance.is_shot_active = False
    # Using AIOBookooError as that's what the scale lib would raise
    mock_bookoo_scale.tare_and_start_timer = AsyncMock(
        side_effect=AIOBookooError("Scale command failed")
    )

    with patch.object(coordinator, "_reset_realtime_analytics") as mock_reset_analytics:
        with pytest.raises(HomeAssistantError) as exc_info:
            await coordinator.async_start_shot_service()

    # Allow tasks created by async_create_task to run
    await hass.async_block_till_done()

    mock_reset_analytics.assert_called_once()
    # Session manager start_session is called via async_create_task before scale command
    mock_session_manager_instance.start_session.assert_called_once_with(
        trigger="ha_service"
    )
    mock_bookoo_scale.tare_and_start_timer.assert_called_once()
    assert "service_call_failed" in str(exc_info.value.translation_key)
    assert (
        "start_shot (tare_and_start_timer)"
        in exc_info.value.translation_placeholders.get("service_name", "")
    )


@pytest.mark.asyncio
async def test_async_stop_shot_service_success(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    mock_bookoo_scale: MagicMock,
    hass: HomeAssistant,  # For async_create_task
):
    """Test successful call to async_stop_shot_service."""
    mock_session_manager_instance.is_shot_active = True
    mock_bookoo_scale.stop_timer = AsyncMock()

    await coordinator.async_stop_shot_service()
    await hass.async_block_till_done()  # Allow tasks to run

    mock_bookoo_scale.stop_timer.assert_called_once()
    mock_session_manager_instance.stop_session.assert_called_once_with(
        stop_reason="ha_service"
    )


@pytest.mark.asyncio
async def test_async_stop_shot_service_no_active_shot(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    mock_bookoo_scale: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test async_stop_shot_service when no shot is active."""
    mock_session_manager_instance.is_shot_active = False
    mock_bookoo_scale.stop_timer = AsyncMock()

    await coordinator.async_stop_shot_service()

    assert "Stop shot service called, but no shot is active." in caplog.text
    mock_bookoo_scale.stop_timer.assert_not_called()
    mock_session_manager_instance.stop_session.assert_not_called()


@pytest.mark.asyncio
async def test_async_stop_shot_service_scale_command_failure(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    mock_bookoo_scale: MagicMock,
    hass: HomeAssistant,  # For async_create_task
    caplog: pytest.LogCaptureFixture,
):
    """Test async_stop_shot_service when the scale command fails."""
    mock_session_manager_instance.is_shot_active = True
    mock_bookoo_scale.stop_timer = AsyncMock(
        side_effect=AIOBookooError("Scale stop command failed")
    )

    with pytest.raises(HomeAssistantError) as exc_info:
        await coordinator.async_stop_shot_service()

    await hass.async_block_till_done()  # Allow tasks to run

    mock_bookoo_scale.stop_timer.assert_called_once()
    # Session manager should still be called to stop, but with a different reason
    mock_session_manager_instance.stop_session.assert_called_once_with(
        stop_reason="ha_service_scale_cmd_fail"
    )
    assert "service_call_failed" in str(exc_info.value.translation_key)
    assert "stop_shot (stop_timer)" in exc_info.value.translation_placeholders.get(
        "service_name", ""
    )
    assert "Error sending Stop Timer command to scale" in caplog.text


@pytest.mark.asyncio
async def test_async_connect_scale_service_success(
    coordinator: BookooCoordinator,
    mock_bookoo_scale: MagicMock,
    hass: HomeAssistant,  # For async_request_refresh
):
    """Test successful call to async_connect_scale_service."""
    mock_bookoo_scale.connected = False
    coordinator._user_initiated_disconnect = True  # Set to ensure it's cleared

    async def _mock_connect(*args, **kwargs):
        mock_bookoo_scale.connected = True  # Simulate successful connection

    mock_bookoo_scale.connect = AsyncMock(side_effect=_mock_connect)

    with patch.object(
        coordinator, "_ensure_queue_processor_running"
    ) as mock_ensure_queue:
        with patch.object(
            coordinator, "async_request_refresh", new_callable=AsyncMock
        ) as mock_refresh:
            await coordinator.async_connect_scale_service()

        assert coordinator._user_initiated_disconnect is False
        mock_bookoo_scale.connect.assert_called_once()
        mock_ensure_queue.assert_called_once()
        mock_refresh.assert_called_once()


@pytest.mark.asyncio
async def test_async_connect_scale_service_already_connected(
    coordinator: BookooCoordinator,
    mock_bookoo_scale: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test async_connect_scale_service when already connected."""
    mock_bookoo_scale.connected = True
    mock_bookoo_scale.connect = AsyncMock()

    await coordinator.async_connect_scale_service()

    assert "Scale already connected. Skipping connect service action." in caplog.text
    mock_bookoo_scale.connect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_to_raise",
    [AIOBookooError("Connect failed"), Exception("Unexpected connect error")],
)
async def test_async_connect_scale_service_failure(
    coordinator: BookooCoordinator,
    mock_bookoo_scale: MagicMock,
    exception_to_raise: Exception,
    caplog: pytest.LogCaptureFixture,
):
    """Test async_connect_scale_service connection failures."""
    mock_bookoo_scale.connected = False
    mock_bookoo_scale.connect = AsyncMock(side_effect=exception_to_raise)

    await coordinator.async_connect_scale_service()

    if isinstance(exception_to_raise, AIOBookooError):
        assert (
            f"Error connecting to Bookoo scale via service: {exception_to_raise}"
            in caplog.text
        )
    else:
        assert (
            f"Unexpected error during connect_scale service: {exception_to_raise}"
            in caplog.text
        )


@pytest.mark.asyncio
async def test_async_disconnect_scale_service_success(
    coordinator: BookooCoordinator,
    mock_bookoo_scale: MagicMock,
    hass: HomeAssistant,  # For async_request_refresh
):
    """Test successful call to async_disconnect_scale_service."""
    mock_bookoo_scale.connected = True
    coordinator._user_initiated_disconnect = False

    async def _mock_disconnect(*args, **kwargs):
        mock_bookoo_scale.connected = False  # Simulate successful disconnection

    mock_bookoo_scale.disconnect = AsyncMock(side_effect=_mock_disconnect)

    with patch.object(
        coordinator, "async_request_refresh", new_callable=AsyncMock
    ) as mock_refresh:
        await coordinator.async_disconnect_scale_service()

        assert coordinator._user_initiated_disconnect is True
        mock_bookoo_scale.disconnect.assert_called_once()
        mock_refresh.assert_called_once()


@pytest.mark.asyncio
async def test_async_disconnect_scale_service_already_disconnected(
    coordinator: BookooCoordinator,
    mock_bookoo_scale: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test async_disconnect_scale_service when already disconnected."""
    mock_bookoo_scale.connected = False
    coordinator._user_initiated_disconnect = False  # Check it gets set
    mock_bookoo_scale.disconnect = AsyncMock()

    await coordinator.async_disconnect_scale_service()

    assert coordinator._user_initiated_disconnect is True
    assert (
        "Scale already disconnected. Skipping disconnect service action." in caplog.text
    )
    mock_bookoo_scale.disconnect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_to_raise",
    [AIOBookooError("Disconnect failed"), Exception("Unexpected disconnect error")],
)
async def test_async_disconnect_scale_service_failure(
    coordinator: BookooCoordinator,
    mock_bookoo_scale: MagicMock,
    exception_to_raise: Exception,
    caplog: pytest.LogCaptureFixture,
):
    """Test async_disconnect_scale_service disconnection failures."""
    mock_bookoo_scale.connected = True
    mock_bookoo_scale.disconnect = AsyncMock(side_effect=exception_to_raise)

    await coordinator.async_disconnect_scale_service()

    if isinstance(exception_to_raise, AIOBookooError):
        assert (
            f"Error disconnecting from Bookoo scale via service: {exception_to_raise}"
            in caplog.text
        )
    else:
        assert (
            f"Unexpected error during disconnect_scale service: {exception_to_raise}"
            in caplog.text
        )


# --- Tests for _handle_characteristic_update ---


@pytest.mark.asyncio
async def test_handle_characteristic_update_command_bytes(
    coordinator: BookooCoordinator,
    hass: HomeAssistant,  # For async_update_listeners mock
):
    """Test _handle_characteristic_update with command char and byte data."""
    with patch.object(coordinator, "async_update_listeners") as mock_update_listeners:
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, b"\x01\x02"
        )
        mock_update_listeners.assert_called_once()


@pytest.mark.asyncio
async def test_handle_characteristic_update_command_none_data(
    coordinator: BookooCoordinator,
    hass: HomeAssistant,  # For async_update_listeners mock
):
    """Test _handle_characteristic_update with command char and None data."""
    with patch.object(coordinator, "async_update_listeners") as mock_update_listeners:
        coordinator._handle_characteristic_update(UPDATE_SOURCE_COMMAND_CHAR, None)
        mock_update_listeners.assert_called_once()


@pytest.mark.asyncio
async def test_handle_characteristic_update_command_auto_timer_start(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    hass: HomeAssistant,  # For async_create_task and async_update_listeners mock
):
    """Test _handle_characteristic_update with command char auto_timer start event."""
    mock_session_manager_instance.is_shot_active = False
    data = {"type": "auto_timer", "event": "start"}

    with patch.object(coordinator, "_reset_realtime_analytics") as mock_reset_analytics:
        with patch.object(
            coordinator, "async_update_listeners"
        ) as mock_update_listeners:
            coordinator._handle_characteristic_update(UPDATE_SOURCE_COMMAND_CHAR, data)
        await (
            hass.async_block_till_done()
        )  # Allow async_create_task for start_session to run

        mock_reset_analytics.assert_called_once()
        mock_session_manager_instance.start_session.assert_called_once_with(
            trigger="scale_auto_dict"
        )
        mock_update_listeners.assert_called()  # Called at least once


@pytest.mark.asyncio
async def test_handle_characteristic_update_command_auto_timer_stop(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    hass: HomeAssistant,  # For async_create_task and async_update_listeners mock
):
    """Test _handle_characteristic_update with command char auto_timer stop event."""
    mock_session_manager_instance.is_shot_active = True
    data = {"type": "auto_timer", "event": "stop"}

    with patch.object(coordinator, "async_update_listeners") as mock_update_listeners:
        coordinator._handle_characteristic_update(UPDATE_SOURCE_COMMAND_CHAR, data)
        await (
            hass.async_block_till_done()
        )  # Allow async_create_task for stop_session to run

        mock_session_manager_instance.stop_session.assert_called_once_with(
            stop_reason="scale_auto_dict"
        )
        mock_update_listeners.assert_called()  # Called at least once


@pytest.mark.asyncio
async def test_handle_characteristic_update_weight_active_shot(
    coordinator: BookooCoordinator,
    mock_session_manager_instance: MagicMock,
    mock_bookoo_scale: MagicMock,
    hass: HomeAssistant,  # For async_update_listeners mock
):
    """Test _handle_characteristic_update with weight char during an active shot."""
    start_time = datetime.now(timezone.utc) - timedelta(seconds=10)
    mock_session_manager_instance.is_shot_active = True
    mock_session_manager_instance.session_start_time_utc = start_time

    mock_bookoo_scale.weight = 15.5
    mock_bookoo_scale.flow_rate = 1.8
    mock_bookoo_scale.timer = 10.0

    with patch.object(
        coordinator, "_update_realtime_analytics_if_needed"
    ) as mock_update_analytics:
        with patch.object(
            coordinator, "async_update_listeners"
        ) as mock_update_listeners:
            coordinator._handle_characteristic_update(UPDATE_SOURCE_WEIGHT_CHAR, None)

        # Check that data adding methods were called on session_manager
        # Elapsed time will be roughly 10.0, allow some flexibility with ANY
        mock_session_manager_instance.add_weight_data.assert_called_once_with(ANY, 15.5)
        mock_session_manager_instance.add_flow_data.assert_called_once_with(ANY, 1.8)
        mock_session_manager_instance.add_scale_timer_data.assert_called_once_with(
            ANY, 10
        )

        mock_update_analytics.assert_called_once()
        mock_update_listeners.assert_called_once()


@pytest.mark.asyncio
async def test_handle_characteristic_update_weight_unexpected_data(
    coordinator: BookooCoordinator,
    caplog: pytest.LogCaptureFixture,
    hass: HomeAssistant,  # For async_update_listeners mock
):
    """Test _handle_characteristic_update with weight char and unexpected data."""
    with patch.object(coordinator, "async_update_listeners") as mock_update_listeners:
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_WEIGHT_CHAR, b"unexpected"
        )
        assert "Unexpected data with UPDATE_SOURCE_WEIGHT_CHAR" in caplog.text
        mock_update_listeners.assert_called_once()  # Still called via _handle_weight_char_update


@pytest.mark.asyncio
async def test_handle_characteristic_update_unknown_source(
    coordinator: BookooCoordinator, caplog: pytest.LogCaptureFixture
):
    """Test _handle_characteristic_update with an unknown source."""
    coordinator._handle_characteristic_update("UNKNOWN_SOURCE", None)
    assert "Unknown characteristic update source: UNKNOWN_SOURCE" in caplog.text


# --- Tests for _options_update_callback ---


@pytest.mark.asyncio
async def test_options_update_callback(
    coordinator: BookooCoordinator,
    hass: HomeAssistant,
    init_integration: ConfigEntry,  # Provides the mock_config_entry as init_integration
):
    """Test that the options update callback reloads config and updates BookooConfig."""
    entry = init_integration  # alias for clarity

    # Ensure hass.config_entries.async_reload is an AsyncMock
    hass.config_entries.async_reload = AsyncMock()

    # Current options (from conftest.py mock_config_entry's default options)
    # Let's assume default min_shot_duration is 10 from BookooConfig.from_config_entry default
    assert coordinator.bookoo_config.min_shot_duration == 10

    # New options to simulate an update
    new_options = entry.options.copy()
    new_min_shot_duration = 20
    new_options["minimum_shot_duration_seconds"] = new_min_shot_duration

    # Update the entry's options directly to simulate HA updating it
    entry.options = new_options

    # Call the callback
    # The callback itself calls _load_options() which updates self.bookoo_config
    # and then hass.config_entries.async_reload(self.config_entry.entry_id)
    await coordinator._options_update_callback(hass, entry)

    # Verify BookooConfig was updated internally by _load_options call within callback
    assert coordinator.bookoo_config.min_shot_duration == new_min_shot_duration

    # Verify that async_reload was called on the config_entry
    hass.config_entries.async_reload.assert_called_once_with(entry.entry_id)
