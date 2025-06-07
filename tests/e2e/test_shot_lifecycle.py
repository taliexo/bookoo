# tests/e2e/test_shot_lifecycle.py
import pytest  # Keep for potential future use
import asyncio
from custom_components.bookoo.const import DOMAIN, EVENT_BOOKOO_SHOT_COMPLETED
from aiobookoov2.const import UPDATE_SOURCE_WEIGHT_CHAR
from homeassistant.core import callback


async def test_complete_shot_lifecycle(hass, init_integration, mock_bookoo_scale):
    """Test a complete shot from start to finish."""
    events = []

    @callback
    def capture_event(event):
        events.append(event)

    hass.bus.async_listen(EVENT_BOOKOO_SHOT_COMPLETED, capture_event)

    await hass.services.async_call(
        DOMAIN, "start_shot", blocking=True, context={"user_id": "test_user"}
    )
    await hass.async_block_till_done()

    coordinator = hass.data[DOMAIN][init_integration.entry_id]

    for i in range(30):  # Simulate 30 seconds of updates
        # Update mock scale's internal state that the coordinator will read
        mock_bookoo_scale.weight = i * 1.2
        mock_bookoo_scale.flow_rate = 2.0 if i > 5 else 0.5
        mock_bookoo_scale.timer = float(i)

        # Simulate the scale library calling the coordinator's callback
        coordinator._handle_characteristic_update(UPDATE_SOURCE_WEIGHT_CHAR, None)

        # Allow Home Assistant to process any resulting tasks or event listeners
        await hass.async_block_till_done()
        await asyncio.sleep(0.01)  # Optional small delay

    await hass.services.async_call(
        DOMAIN, "stop_shot", blocking=True, context={"user_id": "test_user"}
    )
    await hass.async_block_till_done()

    assert len(captured_events) == 1, (
        "Shot completed event was not fired or was fired multiple times."
    )
    event_data = captured_events[0].data

    # Check final weight based on the last simulated update
    # Last weight update: mock_bookoo_scale.weight = 29 * 1.2 = 34.8
    assert event_data["final_weight_grams"] == pytest.approx(34.8, abs=0.01), (
        f"Expected final weight ~34.8, got {event_data['final_weight_grams']}"
    )

    # Check the scale's timer value from the profile
    # Last timer update: mock_bookoo_scale.timer = float(29) = 29.0
    # The profile contains list of dicts: {"elapsed_time": ..., "timer_value": ...}
    assert event_data["scale_timer_profile"], "Scale timer profile should not be empty."
    assert event_data["scale_timer_profile"][-1]["timer_value"] == 29, (
        f"Expected last scale timer value 29, got {event_data['scale_timer_profile'][-1]['timer_value']}"
    )

    # Check wall-clock duration (should be relatively short due to asyncio.sleep(0.01))
    # The loop runs 30 times with 0.01s sleep, so at least 0.3s. Add some buffer for processing.
    assert event_data["duration_seconds"] > 0.2, (
        f"Expected wall-clock duration > 0.2s, got {event_data['duration_seconds']}"
    )
    assert event_data["duration_seconds"] < 5.0, (
        f"Expected wall-clock duration < 5.0s (sanity check), got {event_data['duration_seconds']}"
    )

    assert event_data["status"] == "completed", (
        f"Expected shot status 'completed', got {event_data['status']}"
    )
    assert len(event_data["flow_profile"]) > 0, "Flow profile should not be empty."
