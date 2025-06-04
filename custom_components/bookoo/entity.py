"""Base class for Bookoo entities."""

from homeassistant.helpers.device_registry import (
    CONNECTION_BLUETOOTH,
    DeviceInfo,
    format_mac,
)
from homeassistant.helpers.entity import EntityDescription
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN
from .coordinator import BookooCoordinator


# @dataclass # Intentionally leaving @dataclass off for now due to test collection issues
class BookooEntity(CoordinatorEntity[BookooCoordinator]):
    """Common base class for Bookoo entities."""

    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: BookooCoordinator,
        entity_description: EntityDescription,
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self.entity_description = entity_description

        # Common Bookoo entity attributes initialization (inlined)
        scale = coordinator.scale
        formatted_mac = format_mac(scale.mac)
        self._attr_unique_id = f"{formatted_mac}_{entity_description.key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, formatted_mac)},
            name=scale.name,  # Use scale's name for the device name
            manufacturer="Bookoo",
            model=scale.model,
            connections={(CONNECTION_BLUETOOTH, scale.mac)},
        )

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return bool(super().available and self.coordinator.scale.connected)
