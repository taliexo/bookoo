# tests/e2e/test_performance.py
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util

from custom_components.bookoo.const import (
    DOMAIN,
    SERVICE_START_SHOT,
    SERVICE_STOP_SHOT,
    EVENT_BOOKOO_SHOT_COMPLETED,
)
from homeassistant.const import CONF_ADDRESS
from custom_components.bookoo.coordinator import BookooCoordinator
from custom_components.bookoo.session_manager import SessionManager
from custom_components.bookoo.types import FlowDataPoint

from pytest_homeassistant_custom_component.common import MockConfigEntry

# Define acceptable latency thresholds (in seconds)
# These are initial estimates and can be adjusted based on expected performance.
SERVICE_CALL_LATENCY_THRESHOLD = 0.5  # seconds
SHOT_EVENT_LATENCY_THRESHOLD = 1.0  # seconds


@pytest.fixture(autouse=True)
async def mock_bluetooth(enable_bluetooth: None) -> None:
    """Enable bluetooth fixture."""
    pass


@pytest.fixture
async def async_setup_bookoo_integration_perf(
    hass: HomeAssistant, mock_bluetooth
) -> MockConfigEntry:
    """Set up the Bookoo integration with a mock config entry for performance tests."""
    config_entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_ADDRESS: "00:00:00:00:00:00"},
        unique_id="test_bookoo_perf",
    )
    config_entry.add_to_hass(hass)

    # Mock the BookooScale methods to be very fast or controlled
    with patch(
        "custom_components.bookoo.aiobookoov2.BookooScale.connect",
        new_callable=AsyncMock,
    ) as mock_connect:
        with patch(
            "custom_components.bookoo.aiobookoov2.BookooScale.disconnect",
            new_callable=AsyncMock,
        ):
            with patch(
                "custom_components.bookoo.aiobookoov2.BookooScale.tare_and_start_timer",
                new_callable=AsyncMock,
            ) as mock_tare_start:
                with patch(
                    "custom_components.bookoo.aiobookoov2.BookooScale.stop_timer",
                    new_callable=AsyncMock,
                ) as mock_stop_timer:
                    with patch(
                        "custom_components.bookoo.BookooCoordinator._async_update_data",
                        new_callable=AsyncMock,
                    ):  # Prevent coordinator's background updates
                        mock_connect.return_value = (
                            True  # Simulate successful connection
                        )
                        mock_tare_start.return_value = True
                        mock_stop_timer.return_value = True

                        assert await async_setup_component(hass, DOMAIN, {DOMAIN: {}})
                        await hass.async_block_till_done()

    return config_entry


@pytest.mark.asyncio
async def test_service_call_latency(
    hass: HomeAssistant, async_setup_bookoo_integration_perf: MockConfigEntry
):
    """Test the latency of start_shot and stop_shot service calls."""
    coordinator = hass.data[DOMAIN][async_setup_bookoo_integration_perf.entry_id]
    assert isinstance(coordinator, BookooCoordinator)

    # Test start_shot latency
    # Ensure scale is 'connected' for start_shot to proceed quickly
    coordinator.scale.is_connected = True
    coordinator.session_manager.is_shot_active = False  # Ensure no shot is active

    start_time = time.perf_counter()
    await hass.services.async_call(DOMAIN, SERVICE_START_SHOT, blocking=True)
    end_time = time.perf_counter()
    start_shot_latency = end_time - start_time
    print(f"Start shot service call latency: {start_shot_latency:.4f}s")
    assert start_shot_latency < SERVICE_CALL_LATENCY_THRESHOLD

    # Test stop_shot latency (assuming a shot was started)
    # The mock tare_and_start_timer in coordinator.async_start_shot_service would have been called.
    # The mock for BookooScale.stop_timer will be called by coordinator.async_stop_shot_service
    coordinator.session_manager.is_shot_active = (
        True  # Simulate shot is active for stop to proceed
    )
    coordinator.session_manager.session_start_time_utc = dt_util.utcnow()

    start_time = time.perf_counter()
    await hass.services.async_call(DOMAIN, SERVICE_STOP_SHOT, blocking=True)
    end_time = time.perf_counter()
    stop_shot_latency = end_time - start_time
    print(f"Stop shot service call latency: {stop_shot_latency:.4f}s")
    assert stop_shot_latency < SERVICE_CALL_LATENCY_THRESHOLD


@pytest.mark.asyncio
async def test_shot_completion_event_latency(
    hass: HomeAssistant, async_setup_bookoo_integration_perf: MockConfigEntry
):
    """Test the latency from shot completion trigger to event firing."""
    coordinator = hass.data[DOMAIN][async_setup_bookoo_integration_perf.entry_id]
    assert isinstance(coordinator, BookooCoordinator)
    session_manager = coordinator.session_manager
    assert isinstance(session_manager, SessionManager)

    # Ensure scale is 'connected'
    coordinator.scale.is_connected = True
    session_manager.is_shot_active = False

    # Setup to listen for the event
    event_fired = asyncio.Event()
    event_data = None

    async def event_listener(event):
        nonlocal event_data
        event_data = event.data
        event_fired.set()

    hass.bus.async_listen(EVENT_BOOKOO_SHOT_COMPLETED, event_listener)

    # Start a shot (minimal setup)
    await session_manager.start_session(trigger="perf_test")
    assert session_manager.is_shot_active

    # Simulate some data to make it a valid short shot (not aborted too short)
    session_manager.session_flow_profile.append(
        FlowDataPoint(elapsed_time=1.0, flow_rate=1.0)
    )
    session_manager.session_flow_profile.append(
        FlowDataPoint(elapsed_time=2.0, flow_rate=1.0)
    )
    session_manager.session_flow_profile.append(
        FlowDataPoint(
            elapsed_time=session_manager.config.min_shot_duration + 1, flow_rate=0.0
        )
    )
    coordinator.scale.weight = 10.0  # Mock final weight

    # Trigger shot stop and measure time to event
    time_stop_triggered = time.perf_counter()
    await session_manager.stop_session(stop_reason="perf_test_stop")

    try:
        await asyncio.wait_for(
            event_fired.wait(), timeout=SHOT_EVENT_LATENCY_THRESHOLD + 0.5
        )  # Add buffer to timeout
    except asyncio.TimeoutError:
        pytest.fail("EVENT_BOOKOO_SHOT_COMPLETED was not fired within timeout.")

    time_event_received = time.perf_counter()
    event_latency = time_event_received - time_stop_triggered

    print(f"Shot completion event latency: {event_latency:.4f}s")
    assert event_latency < SHOT_EVENT_LATENCY_THRESHOLD
    assert event_data is not None
    assert event_data["status"] != "aborted_too_short"
