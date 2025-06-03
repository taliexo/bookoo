"""Initialize the Bookoo component."""

from homeassistant.core import HomeAssistant
import typing

from .storage import async_init_db

from .const import DOMAIN, SERVICE_START_SHOT, SERVICE_STOP_SHOT, PLATFORMS
from .coordinator import BookooConfigEntry, BookooCoordinator


async def async_setup_entry(hass: HomeAssistant, entry: BookooConfigEntry) -> bool:
    """Set up bookoo as config entry."""

    coordinator = BookooCoordinator(hass, entry)
    await async_init_db(hass)  # Initialize the database
    await coordinator.async_config_entry_first_refresh()

    entry.runtime_data = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Register services
    hass.services.async_register(
        DOMAIN,
        SERVICE_START_SHOT,
        coordinator.async_start_shot_service,  # Link to coordinator method
    )
    hass.services.async_register(
        DOMAIN,
        SERVICE_STOP_SHOT,
        coordinator.async_stop_shot_service,  # Link to coordinator method
    )

    return True


async def async_unload_entry(hass: HomeAssistant, entry: BookooConfigEntry) -> bool:
    """Unload a config entry."""

    return typing.cast(
        bool, await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    )
