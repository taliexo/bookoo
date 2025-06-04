"""Tests for the Bookoo sensor platform."""

from unittest.mock import AsyncMock, MagicMock, patch
from homeassistant.helpers.entity_platform import EntityPlatform
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry  # type: ignore[import-untyped]

from custom_components.bookoo.const import DOMAIN
from custom_components.bookoo.coordinator import BookooCoordinator
from custom_components.bookoo.sensor import async_setup_entry
from homeassistant.const import PERCENTAGE
from homeassistant.core import HomeAssistant


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
def mock_config_entry_for_sensor(hass: HomeAssistant):
    """Fixture for a basic mock ConfigEntry for sensor tests."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"address": "test_address_sensor", "is_valid_scale": True},
        options={},
        title="Test Bookoo Scale Sensors",
        unique_id="sensor_test_unique_id",
    )
    entry.runtime_data = None  # Initialize runtime_data
    return entry


@pytest.fixture
async def configured_config_entry_with_coordinator(
    hass: HomeAssistant, mock_bookoo_scale_for_sensor, mock_config_entry_for_sensor
):
    """Fixture that creates a coordinator and attaches it to a config entry."""
    with patch(
        "custom_components.bookoo.coordinator.BookooScale",
        return_value=mock_bookoo_scale_for_sensor,
    ):
        coordinator = BookooCoordinator(hass, mock_config_entry_for_sensor)
        coordinator.last_shot_data = None  # Initialize
        # Set initial values for realtime analytics for predictability in tests
        coordinator.realtime_channeling_status = "None"
        coordinator.realtime_pre_infusion_active = False
        coordinator.realtime_pre_infusion_duration = None
        coordinator.realtime_extraction_uniformity = 0.0
        coordinator.realtime_shot_quality_score = 0.0

        mock_config_entry_for_sensor.runtime_data = (
            coordinator  # Set runtime_data on the entry
        )

    # If your coordinator has an async first refresh method that needs to be called:
    # await coordinator.async_config_entry_first_refresh()

    return mock_config_entry_for_sensor  # Return the config entry, now with runtime_data set


@pytest.mark.asyncio
async def test_setup_sensors(
    hass: HomeAssistant,
    configured_config_entry_with_coordinator,  # This is the coroutine fixture
):
    """Test the setup of sensor entities."""
    async_add_entities = MagicMock()

    actual_config_entry = await configured_config_entry_with_coordinator
    # Pass the config entry that now has the coordinator in its runtime_data
    await async_setup_entry(hass, actual_config_entry, async_add_entities)

    # Check that entities were added
    assert async_add_entities.call_count > 0

    # Check for the quality score sensor specifically
    found_quality_score_sensor = False
    for entity in async_add_entities.call_args.args[0]:
        if entity.entity_description.key == "current_shot_quality_score":
            found_quality_score_sensor = True
            assert entity.native_unit_of_measurement == PERCENTAGE
            # The name assertion might be tricky due to translations,
            # consider checking translation_key or a more stable property if needed.
            # assert entity.name == "Current Shot Quality Score"
            break
    assert found_quality_score_sensor, "Current Shot Quality Score sensor not found"


@pytest.mark.asyncio
async def test_current_shot_quality_score_sensor(
    hass: HomeAssistant,
    configured_config_entry_with_coordinator,  # This is the coroutine fixture
):
    """Test the current_shot_quality_score sensor."""
    async_add_entities_mock = MagicMock()

    actual_config_entry = await configured_config_entry_with_coordinator
    # The coordinator is now on actual_config_entry.runtime_data
    sensor_coordinator = actual_config_entry.runtime_data
    assert sensor_coordinator is not None  # Ensure coordinator exists

    await async_setup_entry(hass, actual_config_entry, async_add_entities_mock)

    quality_score_sensor = None
    for entity in async_add_entities_mock.call_args.args[0]:
        if entity.entity_description.key == "current_shot_quality_score":
            quality_score_sensor = entity
            break

    assert quality_score_sensor is not None, "Quality score sensor not created"

    # Manually assign hass and a mock platform to the sensor instance
    quality_score_sensor.hass = hass
    mock_platform = MagicMock(spec=EntityPlatform)
    mock_platform.platform_name = DOMAIN  # Typically the integration's domain
    mock_platform.domain = DOMAIN  # Used in some name generation
    mock_platform.component_translations = {}  # Mock component_translations
    mock_platform.platform_translations = {}  # Mock platform_translations
    quality_score_sensor.platform = mock_platform
    # Manually set entity_id for the test, as EntityRegistry doesn't run
    if (
        quality_score_sensor.entity_description
        and quality_score_sensor.entity_description.key
    ):
        quality_score_sensor.entity_id = (
            f"sensor.{quality_score_sensor.entity_description.key}"
        )
    else:
        # Fallback if key is not available, though it should be for this sensor
        quality_score_sensor.entity_id = "sensor.test_quality_score"

    # Test case 1: Initial value (should be 0.0 from coordinator init in fixture)
    # The coordinator's realtime_shot_quality_score was set to 0.0 in the fixture
    # sensor_coordinator.realtime_shot_quality_score is already 0.0 from fixture setup
    assert quality_score_sensor.native_value == 0.0

    # Test case 2: Update coordinator value
    sensor_coordinator.realtime_shot_quality_score = 75.56
    assert quality_score_sensor.native_value == 75.6  # Check rounding

    # Test case 3: Value is None
    sensor_coordinator.realtime_shot_quality_score = None
    assert quality_score_sensor.native_value is None


# TODO: Add tests for other sensors:
# - current_shot_channeling_status
# - current_shot_pre_infusion_duration
# - current_shot_extraction_uniformity
# - weight, flow_rate, timer, current_shot_duration
# - last_shot_* sensors
# - battery sensor (from RESTORE_SENSORS)
