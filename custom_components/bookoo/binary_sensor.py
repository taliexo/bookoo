"""Binary sensor platform for Bookoo scales."""

from collections.abc import Callable
from dataclasses import dataclass

from .coordinator import BookooCoordinator  # Add this import

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
    BinarySensorEntityDescription,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .coordinator import BookooConfigEntry
from .entity import BookooEntity

# Coordinator is used to centralize the data updates
PARALLEL_UPDATES = 0


@dataclass(kw_only=True, frozen=True)
class BookooBinarySensorEntityDescription(BinarySensorEntityDescription):
    """Description for Bookoo binary sensor entities."""

    is_on_fn: Callable[[BookooCoordinator], bool]


BINARY_SENSORS: tuple[BookooBinarySensorEntityDescription, ...] = (
    BookooBinarySensorEntityDescription(
        key="connected",
        translation_key="connected",
        device_class=BinarySensorDeviceClass.CONNECTIVITY,
        is_on_fn=lambda coordinator: coordinator.scale.connected,  # Changed to use coordinator
    ),
    BookooBinarySensorEntityDescription(
        key="shot_in_progress",
        translation_key="shot_in_progress",  # Needs entry in strings.json
        icon="mdi:timer-sand",
        is_on_fn=lambda coordinator: coordinator.session_manager.is_shot_active,
    ),
    BookooBinarySensorEntityDescription(
        key="current_shot_pre_infusion_active",
        translation_key="current_shot_pre_infusion_active",
        icon="mdi:water-opacity",  # Indicates water presence/flow state
        # Consider device_class=BinarySensorDeviceClass.RUNNING if it makes sense for pre-infusion state
        is_on_fn=lambda coordinator: coordinator.realtime_pre_infusion_active,
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BookooConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up binary sensors."""

    coordinator = entry.runtime_data
    async_add_entities(
        BookooBinarySensor(coordinator, description) for description in BINARY_SENSORS
    )


class BookooBinarySensor(BookooEntity, BinarySensorEntity):
    """Representation of an Bookoo binary sensor."""

    entity_description: BookooBinarySensorEntityDescription

    @property
    def is_on(self) -> bool:
        """Return true if the binary sensor is on."""
        return self.entity_description.is_on_fn(
            self.coordinator
        )  # Changed to use coordinator
