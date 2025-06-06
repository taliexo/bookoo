"""Sensor platform for Bookoo."""

from collections.abc import Callable  # noqa: I001
from dataclasses import dataclass
from datetime import datetime

from aiobookoov2.bookooscale import BookooDeviceState
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
from homeassistant.util import dt as dt_util
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .coordinator import (
    BookooConfigEntry,
    BookooCoordinator,
)
from .entity import BookooEntity

# Import typing for casts and Optional if not already present from other stdlib imports
import typing

# Coordinator is used to centralize the data updates
PARALLEL_UPDATES = 0


@dataclass(kw_only=True, frozen=True)
class BookooSensorEntityDescription(SensorEntityDescription):
    """Describes a Bookoo sensor entity.

    Attributes:
        value_fn: Callable that takes the BookooCoordinator and returns the sensor's state.
    """

    value_fn: Callable[[BookooCoordinator], int | float | str | datetime | None]


@dataclass(kw_only=True, frozen=True)
class BookooDynamicUnitSensorEntityDescription(BookooSensorEntityDescription):
    """Describes a Bookoo sensor entity with potentially dynamic units.

    Attributes:
        unit_fn: Optional callable that takes BookooDeviceState and returns the unit string.
                 (Currently not used, native_unit_of_measurement is primary).
    """

    unit_fn: Callable[[BookooDeviceState], str] | None = None


SENSORS: tuple[BookooSensorEntityDescription, ...] = (
    BookooDynamicUnitSensorEntityDescription(
        key="weight",
        translation_key="weight",
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.GRAMS,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.scale.weight,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="flow_rate",
        translation_key="flow_rate",
        device_class=SensorDeviceClass.VOLUME_FLOW_RATE,
        native_unit_of_measurement=UnitOfVolumeFlowRate.MILLILITERS_PER_SECOND,
        suggested_display_precision=1,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.scale.flow_rate,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="timer",
        translation_key="timer",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=2,
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.scale.timer,
    ),
    # Current Shot Sensor
    BookooDynamicUnitSensorEntityDescription(
        key="current_shot_duration",
        translation_key="current_shot_duration",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:timer-outline",
        value_fn=lambda coordinator: (
            (
                dt_util.utcnow() - coordinator.session_manager.session_start_time_utc
            ).total_seconds()
            if coordinator.session_manager.is_shot_active
            and coordinator.session_manager.session_start_time_utc
            else 0.0
        ),
    ),
    # Last Shot Sensors
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_duration",
        translation_key="last_shot_duration",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:history",
        value_fn=lambda coordinator: coordinator.last_shot_data.duration_seconds
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_final_weight",
        translation_key="last_shot_final_weight",
        device_class=SensorDeviceClass.WEIGHT,
        native_unit_of_measurement=UnitOfMass.GRAMS,
        suggested_display_precision=1,
        icon="mdi:weight-gram",
        value_fn=lambda coordinator: coordinator.last_shot_data.final_weight_grams
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_start_time",
        translation_key="last_shot_start_time",
        device_class=SensorDeviceClass.TIMESTAMP,
        icon="mdi:clock-start",
        value_fn=lambda coordinator: (
            dt_util.parse_datetime(coordinator.last_shot_data.start_time_utc)
            if coordinator.last_shot_data
            and coordinator.last_shot_data.start_time_utc  # Direct attribute access implies existence
            else None
        ),
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_status",
        translation_key="last_shot_status",
        icon="mdi:list-status",
        value_fn=lambda coordinator: coordinator.last_shot_data.status
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_channeling_status",
        translation_key="last_shot_channeling_status",
        icon="mdi:chart-scatter-plot",
        value_fn=lambda coordinator: coordinator.last_shot_data.channeling_status
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_pre_infusion_duration",
        translation_key="last_shot_pre_infusion_duration",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:timelapse",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.last_shot_data.pre_infusion_duration_seconds
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_extraction_uniformity",
        translation_key="last_shot_extraction_uniformity",
        native_unit_of_measurement=PERCENTAGE,
        suggested_display_precision=1,
        icon="mdi:chart-bell-curve-cumulative",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            coordinator.last_shot_data.extraction_uniformity_metric * 100.0
            if coordinator.last_shot_data
            and coordinator.last_shot_data.extraction_uniformity_metric is not None
            else None
        ),
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_average_flow_rate",
        translation_key="last_shot_average_flow_rate",
        device_class=SensorDeviceClass.VOLUME_FLOW_RATE,
        native_unit_of_measurement=UnitOfVolumeFlowRate.MILLILITERS_PER_SECOND,
        suggested_display_precision=1,
        icon="mdi:water-pump",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.last_shot_data.average_flow_rate_gps
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_peak_flow_rate",
        translation_key="last_shot_peak_flow_rate",
        device_class=SensorDeviceClass.VOLUME_FLOW_RATE,
        native_unit_of_measurement=UnitOfVolumeFlowRate.MILLILITERS_PER_SECOND,
        suggested_display_precision=1,
        icon="mdi:water-pump-outline",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.last_shot_data.peak_flow_rate_gps
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_time_to_first_flow",
        translation_key="last_shot_time_to_first_flow",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:speedometer-slow",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.last_shot_data.time_to_first_flow_seconds
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_time_to_peak_flow",
        translation_key="last_shot_time_to_peak_flow",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:speedometer-medium",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: coordinator.last_shot_data.time_to_peak_flow_seconds
        if coordinator.last_shot_data
        else None,
    ),
    BookooDynamicUnitSensorEntityDescription(
        key="last_shot_quality_score",
        translation_key="last_shot_quality_score",
        native_unit_of_measurement=PERCENTAGE,
        suggested_display_precision=1,
        icon="mdi:chart-gantt",
        state_class=SensorStateClass.MEASUREMENT,
        value_fn=lambda coordinator: (
            round(coordinator.last_shot_data.shot_quality_score, 1)
            if coordinator.last_shot_data
            and coordinator.last_shot_data.shot_quality_score is not None
            else None
        ),
    ),
    BookooSensorEntityDescription(
        key="last_shot_next_shot_recommendation",
        translation_key="last_shot_next_shot_recommendation",
        icon="mdi:lightbulb-on-outline",
        value_fn=lambda coordinator: coordinator.last_shot_data.next_shot_recommendation
        if coordinator.last_shot_data
        else None,
    ),
    # Real-time Analytics Sensors
    BookooSensorEntityDescription(
        key="current_shot_channeling_status",
        translation_key="current_shot_channeling_status",
        icon="mdi:chart-scatter-plot",
        value_fn=lambda coordinator: coordinator.realtime_channeling_status,
    ),
    BookooSensorEntityDescription(
        key="current_shot_pre_infusion_duration",
        translation_key="current_shot_pre_infusion_duration",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        suggested_display_precision=1,
        icon="mdi:timelapse",
        value_fn=lambda coordinator: coordinator.realtime_pre_infusion_duration,
    ),
    BookooSensorEntityDescription(
        key="current_shot_extraction_uniformity",
        translation_key="current_shot_extraction_uniformity",
        native_unit_of_measurement=PERCENTAGE,  # Will be 0.0-1.0, displayed as %
        suggested_display_precision=1,
        icon="mdi:chart-bell-curve-cumulative",
        state_class=SensorStateClass.MEASUREMENT,  # Good for % values
        value_fn=lambda coordinator: (
            coordinator.realtime_extraction_uniformity * 100.0
            if coordinator.realtime_extraction_uniformity is not None
            else None
        ),  # Convert 0.0-1.0 to 0-100 for HA percentage
    ),
    BookooSensorEntityDescription(
        key="current_shot_quality_score",
        translation_key="current_shot_quality_score",
        native_unit_of_measurement=PERCENTAGE,
        icon="mdi:chart-gantt",  # Or mdi:gauge, mdi:speedometer
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=1,
        value_fn=lambda coordinator: round(coordinator.realtime_shot_quality_score, 1)
        if coordinator.realtime_shot_quality_score is not None
        else None,
    ),
)

RESTORE_SENSORS: tuple[BookooSensorEntityDescription, ...] = (
    BookooSensorEntityDescription(
        key="battery",
        translation_key="battery",
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
    """Set up Bookoo sensor entities based on the config entry.

    Creates sensor entities for various scale readings, shot metrics,
    and real-time analytics.
    """

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
    """Representation of a generic Bookoo sensor.

    This class serves as the base for sensors that derive their state directly
    from the coordinator via a `value_fn` defined in their entity description.
    """

    entity_description: BookooSensorEntityDescription

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the unit of measurement of this entity."""
        if (
            self.coordinator.scale.device_state is not None
            and hasattr(self.entity_description, "unit_fn")
            and isinstance(
                self.entity_description, BookooDynamicUnitSensorEntityDescription
            )
            and self.entity_description.unit_fn is not None
        ):
            # Now we are sure it's BookooDynamicUnitSensorEntityDescription and unit_fn is not None (due to the isinstance and not None check)
            # The hasattr is a bit redundant if isinstance is used, but safe.
            return self.entity_description.unit_fn(self.coordinator.scale.device_state)

        # Otherwise, defer to the native_unit_of_measurement from the entity_description.
        # This covers cases where entity_description is BookooSensorEntityDescription (no unit_fn)
        # or BookooDynamicUnitSensorEntityDescription but unit_fn is None.
        # Explicitly cast the type for mypy
        return typing.cast(
            str | None, self.entity_description.native_unit_of_measurement
        )

    @property
    def native_value(
        self,
    ) -> int | float | str | datetime | None:
        """Return the state of the entity."""
        return self.entity_description.value_fn(self.coordinator)  # Pass coordinator


class BookooRestoreSensor(BookooEntity, RestoreSensor):
    """Representation of a Bookoo sensor that can restore its state.

    Used for sensors like battery level that should retain their last known
    value across Home Assistant restarts if the device is temporarily unavailable.
    """

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
