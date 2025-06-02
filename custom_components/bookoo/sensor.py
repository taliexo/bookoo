"""Sensor platform for Bookoo."""

from collections.abc import Callable  # noqa: I001
from dataclasses import dataclass

from aiobookoo.bookooscale import BookooDeviceState
from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorExtraStoredData,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, UnitOfMass, UnitOfVolumeFlowRate, UnitOfTime
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util  # Added for time calculations
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .coordinator import (
    BookooConfigEntry,
    BookooCoordinator,
)  # Added BookooCoordinator import
from .entity import BookooEntity

# Coordinator is used to centralize the data updates
PARALLEL_UPDATES = 0


@dataclass(kw_only=True, frozen=True)
class BookooSensorEntityDescription(SensorEntityDescription):
    """Description for Bookoo sensor entities."""

    value_fn: Callable[
        [BookooCoordinator], int | float | str | None
    ]  # Changed BookooScale to BookooCoordinator, added str for status


@dataclass(kw_only=True, frozen=True)
class BookooDynamicUnitSensorEntityDescription(BookooSensorEntityDescription):
    """Description for Bookoo sensor entities with dynamic units."""

    unit_fn: Callable[[BookooDeviceState], str] | None = None


SENSORS: tuple[BookooSensorEntityDescription, ...] = (
    BookooDynamicUnitSensorEntityDescription(
        key="weight",
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.GRAMS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.scale.weight,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="flow_rate",
        device_class=SensorDeviceClass.VOLUME_FLOW_RATE,
        native_unit_of_measurement=UnitOfVolumeFlowRate.MILLILITERS_PER_SECOND,
        suggested_display_precision=1,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.scale.flow_rate,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="timer",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=2,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.scale.timer,
    ),
    # Current Shot Sensor
    BookooSensorEntityDescription(
        key="current_shot_duration",
        translation_key="current_shot_duration",  # Needs strings.json
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:timer-outline",
        value_fn=lambda coordinator: (
            (dt_util.utcnow() - coordinator.session_start_time_utc).total_seconds()
            if coordinator.is_shot_active and coordinator.session_start_time_utc
            else 0.0
        ),
    ),
    # Last Shot Sensors
    BookooSensorEntityDescription(
        key="last_shot_duration",
        translation_key="last_shot_duration",  # Needs strings.json
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:history",
        value_fn=lambda coordinator: coordinator.last_shot_data.get("duration_seconds")
        if coordinator.last_shot_data
        else None,
    ),
    BookooSensorEntityDescription(
        key="last_shot_final_weight",
        translation_key="last_shot_final_weight",  # Needs strings.json
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.GRAMS,
        suggested_display_precision=1,
        icon="mdi:weight-gram",
        value_fn=lambda coordinator: coordinator.last_shot_data.get(
            "final_weight_grams"
        )
        if coordinator.last_shot_data
        else None,
    ),
    BookooSensorEntityDescription(
        key="last_shot_start_time",
        translation_key="last_shot_start_time",  # Needs strings.json
        device_class=SensorDeviceClass.TIMESTAMP,
        icon="mdi:clock-start",
        value_fn=lambda coordinator: (
            dt_util.parse_datetime(coordinator.last_shot_data["start_time_utc"])
            if coordinator.last_shot_data
            and coordinator.last_shot_data.get("start_time_utc")
            else None
        ),
    ),
    BookooSensorEntityDescription(
        key="last_shot_status",
        translation_key="last_shot_status",  # Needs strings.json
        icon="mdi:list-status",
        value_fn=lambda coordinator: coordinator.last_shot_data.get("status")
        if coordinator.last_shot_data
        else None,
    ),
)

RESTORE_SENSORS: tuple[BookooSensorEntityDescription, ...] = (
    BookooSensorEntityDescription(
        key="battery",
        device_class=SensorDeviceClass.BATTERY,
        native_unit_of_measurement=PERCENTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.scale.device_state.battery_level
        if coordinator.scale.device_state
        else None,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BookooConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensors."""

    coordinator = entry.runtime_data
    entities: list[SensorEntity] = [
        BookooSensor(coordinator, entity_description) for entity_description in SENSORS
    ]
    entities.extend(
        BookooRestoreSensor(coordinator, entity_description)
        for entity_description in RESTORE_SENSORS
    )
    async_add_entities(entities)


class BookooSensor(BookooEntity, SensorEntity):
    """Representation of an Bookoo sensor."""

    entity_description: BookooDynamicUnitSensorEntityDescription

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit of measurement of this entity."""
        if (
            self._scale.device_state is not None
            and self.entity_description.unit_fn is not None
        ):
            return self.entity_description.unit_fn(self._scale.device_state)
        return self.entity_description.native_unit_of_measurement

    @property
    def native_value(self) -> int | float | str | None:  # Added str for status
        """Return the state of the entity."""
        return self.entity_description.value_fn(self.coordinator)  # Pass coordinator


class BookooRestoreSensor(BookooEntity, RestoreSensor):
    """Representation of an Bookoo sensor with restore capabilities."""

    entity_description: BookooSensorEntityDescription
    _restored_data: SensorExtraStoredData | None = None

    async def async_added_to_hass(self) -> None:
        """Handle entity which will be added."""
        await super().async_added_to_hass()

        self._restored_data = await self.async_get_last_sensor_data()
        if self._restored_data is not None:
            self._attr_native_value = self._restored_data.native_value
            self._attr_native_unit_of_measurement = (
                self._restored_data.native_unit_of_measurement
            )

        if self.coordinator.scale.device_state is not None:
            self._attr_native_value = self.entity_description.value_fn(self.coordinator)

    @callback
    def _handle_coordinator_update(self) -> None:
        """Handle updated data from the coordinator."""
        # This method is called by the coordinator when data is updated.
        # For BookooRestoreSensor (battery), we explicitly update its _attr_native_value.
        # Other sensors derive their state directly from the coordinator in their native_value property.
        if self.coordinator.scale.device_state is not None:
            # This specifically targets the BookooRestoreSensor for battery
            # Ensure this logic only applies if it IS a battery sensor to avoid errors if description changes
            if self.entity_description.key == "battery":
                self._attr_native_value = self.entity_description.value_fn(
                    self.coordinator
                )
        self.async_schedule_update_ha_state()

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return super().available or self._restored_data is not None
