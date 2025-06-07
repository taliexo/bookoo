# tests/integration/test_entities.py


async def test_sensor_entities(hass, init_integration, mock_bookoo_scale):
    """Test sensor entity creation and updates."""
    # Check weight sensor
    state = hass.states.get("sensor.test_bookoo_scale_weight")
    assert state is not None
    assert state.state == "0.0"

    # Update scale weight
    mock_bookoo_scale.weight = 18.5
    if hasattr(mock_bookoo_scale, "notify_callback") and callable(
        mock_bookoo_scale.notify_callback
    ):
        mock_bookoo_scale.notify_callback()
    else:
        coordinator = hass.data[init_integration.domain][init_integration.entry_id]
        if (
            coordinator
            and hasattr(coordinator, "async_update_listeners")
            and callable(coordinator.async_update_listeners)
        ):
            coordinator.async_update_listeners()

    await hass.async_block_till_done()

    state = hass.states.get("sensor.test_bookoo_scale_weight")
    assert state.state == "18.5"
