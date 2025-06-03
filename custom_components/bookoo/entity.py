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


class BookooEntitySetupLogic:
    """Provides setup logic for Bookoo entities."""

    @staticmethod
    def initialize_common_attributes(
        entity_instance,  # The actual entity instance (e.g., BookooEntity)
        coordinator: BookooCoordinator,
        entity_description: EntityDescription,
    ) -> None:
        """Initialize common Bookoo entity attributes for the given instance."""
        scale = coordinator.scale
        formatted_mac = format_mac(scale.mac)
        entity_instance._attr_unique_id = f"{formatted_mac}_{entity_description.key}"
        entity_instance._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, formatted_mac)},
            name=scale.name,  # Use scale's name for the device name
            manufacturer="Bookoo",
            model=scale.model,
            suggested_area="Kitchen",
            connections={(CONNECTION_BLUETOOTH, scale.mac)},
        )


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
        super().__init__(coordinator)  # Initialize CoordinatorEntity
        self.entity_description = entity_description
        BookooEntitySetupLogic.initialize_common_attributes(
            self, coordinator, entity_description
        )

    @property
    def available(self) -> bool:
        """Return True if entity is available."""
        return bool(super().available and self.coordinator.scale.connected)
