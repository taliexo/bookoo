"""Tests for the Bookoo sensor platform."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.bookoo.const import DOMAIN
from custom_components.bookoo.coordinator import BookooCoordinator
from custom_components.bookoo.sensor import async_setup_entry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant

# Fixtures from test_coordinator.py might be reusable or adaptable here
# For simplicity, we'll define some basic ones or assume they are available
# if conftest.py is structured to share them.


@pytest.fixture
def mock_bookoo_scale_for_sensor():
    """Fixture for a mock BookooScale tailored for sensor tests."""
    scale = MagicMock()
    scale.weight = 10.0
    scale.flow_rate = 2.0
    scale.timer = 30.0
    scale.device_state = MagicMock()
    scale.device_state.battery_level = 90
    scale.mac = "sensor_test_mac"
    scale.address = scale.mac
    scale.connected = True
    scale.async_send_command = AsyncMock()
    scale.async_connect = AsyncMock(return_value=True)
    return scale


@pytest.fixture
async def sensor_coordinator(
    hass: HomeAssistant, mock_bookoo_scale_for_sensor, mock_config_entry_for_sensor
):
    """Fixture for a BookooCoordinator instance for sensor tests."""
    # Use the existing mock_config_entry or create a specific one if needed
    # mock_config_entry_for_sensor = MockConfigEntry(domain=DOMAIN, data={"address": "test_address_sensor"}, unique_id="sensor_test_unique_id")

    with patch(
        "custom_components.bookoo.coordinator.BookooScale",
        return_value=mock_bookoo_scale_for_sensor,
    ):
        coordinator = BookooCoordinator(hass, mock_config_entry_for_sensor)
        coordinator.last_shot_data = {}  # Initialize
        # Set initial values for realtime analytics for predictability in tests
        coordinator.realtime_channeling_status = "None"
        coordinator.realtime_pre_infusion_active = False
        coordinator.realtime_pre_infusion_duration = None
        coordinator.realtime_extraction_uniformity = 0.0
        coordinator.realtime_shot_quality_score = 0.0
        mock_config_entry_for_sensor.runtime_data = (
            coordinator  # Store coordinator in runtime_data
        )
        yield coordinator


@pytest.fixture
def mock_config_entry_for_sensor(hass: HomeAssistant):  # Added hass fixture
    """Fixture for a mock ConfigEntry for sensor tests."""
    return MockConfigEntry(
        domain=DOMAIN,
        data={"address": "test_address_sensor", "is_valid_scale": True},
        options={},
        title="Test Bookoo Scale Sensors",
        unique_id="sensor_test_unique_id",
    )


async def test_setup_sensors(
    hass: HomeAssistant,
    sensor_coordinator: BookooCoordinator,
    mock_config_entry_for_sensor,
):
    """Test the setup of sensor entities."""
    async_add_entities = MagicMock()

    await async_setup_entry(hass, mock_config_entry_for_sensor, async_add_entities)

    # Check that entities were added
    assert async_add_entities.call_count > 0
    # Further checks can be added to verify specific entities were created
    # For example, by inspecting async_add_entities.call_args
    # print(async_add_entities.call_args.args[0]) # To see the list of entities

    # Check for the quality score sensor specifically
    found_quality_score_sensor = False
    for entity in async_add_entities.call_args.args[0]:
        if entity.entity_description.key == "current_shot_quality_score":
            found_quality_score_sensor = True
            assert entity.native_unit_of_measurement == PERCENTAGE
            assert (
                entity.name == "Current Shot Quality Score"
            )  # This will depend on strings.json and translation_key
            break
    assert found_quality_score_sensor, "Current Shot Quality Score sensor not found"


async def test_current_shot_quality_score_sensor(
    hass: HomeAssistant,
    sensor_coordinator: BookooCoordinator,
    mock_config_entry_for_sensor,
):
    """Test the current_shot_quality_score sensor."""
    async_add_entities_mock = MagicMock()
    await async_setup_entry(hass, mock_config_entry_for_sensor, async_add_entities_mock)

    quality_score_sensor = None
    for entity in async_add_entities_mock.call_args.args[0]:
        if entity.entity_description.key == "current_shot_quality_score":
            quality_score_sensor = entity
            break

    assert quality_score_sensor is not None, "Quality score sensor not created"

    # Test case 1: Initial value (should be 0.0 from coordinator init)
    sensor_coordinator.realtime_shot_quality_score = 0.0
    quality_score_sensor.async_write_ha_state()  # Or trigger coordinator update
    await hass.async_block_till_done()
    assert quality_score_sensor.native_value == 0.0

    # Test case 2: Update coordinator value
    sensor_coordinator.realtime_shot_quality_score = 75.56
    # Simulate coordinator update that would trigger sensor update
    # In a real scenario, this happens via coordinator.async_update_listeners() or data update
    # For direct testing, we might need to call a method on the sensor or mock its update mechanism
    # For now, let's assume direct update or that async_write_ha_state is sufficient for test
    quality_score_sensor.async_write_ha_state()
    await hass.async_block_till_done()
    assert quality_score_sensor.native_value == 75.6  # Check rounding

    # Test case 3: Value is None
    sensor_coordinator.realtime_shot_quality_score = None
    quality_score_sensor.async_write_ha_state()
    await hass.async_block_till_done()
    assert quality_score_sensor.native_value is None


# TODO: Add tests for other sensors:
# - current_shot_channeling_status
# - current_shot_pre_infusion_duration
# - current_shot_extraction_uniformity
# - weight, flow_rate, timer, current_shot_duration
# - last_shot_* sensors
# - battery sensor (from RESTORE_SENSORS)
