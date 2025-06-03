"""Button entities for Bookoo scales."""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

# from aiobookoo.bookooscale import BookooScale # No longer directly needed by press_fn type hint
from .coordinator import BookooCoordinator  # Import coordinator for press_fn type hint

from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .coordinator import BookooConfigEntry  # Ensure BookooCoordinator is imported
from .entity import BookooEntity
from .const import (
    DOMAIN,
    SERVICE_START_SHOT,
    SERVICE_STOP_SHOT,
)  # Added for service calls

PARALLEL_UPDATES = 0


@dataclass(kw_only=True, frozen=True)
class BookooButtonEntityDescription(ButtonEntityDescription):
    """Description for bookoo button entities."""

    press_fn: Callable[[BookooCoordinator], Coroutine[Any, Any, None]] | None = (
        None  # Made optional
    )


BUTTONS: tuple[BookooButtonEntityDescription, ...] = (
    BookooButtonEntityDescription(
        key="tare",
        translation_key="tare",
        icon="mdi:target",  # Added icon for consistency
        press_fn=lambda coordinator: coordinator.scale.tare(),
    ),
    BookooButtonEntityDescription(
        key="reset_timer",
        translation_key="reset_timer",
        icon="mdi:timer-refresh-outline",  # Added icon
        press_fn=lambda coordinator: coordinator.scale.reset_timer(),
    ),
    BookooButtonEntityDescription(
        key="start",
        translation_key="start",
        icon="mdi:play",  # Added icon
        press_fn=lambda coordinator: coordinator.scale.start_timer(),
    ),
    BookooButtonEntityDescription(
        key="stop",
        translation_key="stop",
        icon="mdi:stop",  # Added icon
        press_fn=lambda coordinator: coordinator.scale.stop_timer(),
    ),
    BookooButtonEntityDescription(
        key="tare_and_start",
        translation_key="tare_and_start",
        icon="mdi:target-account",  # Example icon, adjust as needed
        press_fn=lambda coordinator: coordinator.scale.tare_and_start_timer(),
    ),
    BookooButtonEntityDescription(
        key="start_shot_session",
        translation_key="start_shot_session",  # Needs entry in strings.json
        icon="mdi:play-circle-outline",
        # press_fn is now handled by key in async_press
        # press_fn=lambda coordinator: coordinator.async_start_shot_service(), # This call is no longer direct
    ),
    BookooButtonEntityDescription(
        key="stop_shot_session",
        translation_key="stop_shot_session",  # Needs entry in strings.json
        icon="mdi:stop-circle-outline",
        # press_fn is now handled by key in async_press
        # press_fn=lambda coordinator: coordinator.async_stop_shot_service(), # This call is no longer direct
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BookooConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up button entities and services."""

    coordinator = entry.runtime_data
    async_add_entities(
        BookooButton(coordinator, description) for description in BUTTONS
    )


class BookooButton(BookooEntity, ButtonEntity):
    """Representation of an Bookoo button."""

    entity_description: BookooButtonEntityDescription

    async def async_press(self) -> None:
        """Handle the button press."""
        if self.entity_description.key == "start_shot_session":
            await self.hass.services.async_call(
                DOMAIN, SERVICE_START_SHOT, blocking=True
            )
        elif self.entity_description.key == "stop_shot_session":
            await self.hass.services.async_call(
                DOMAIN, SERVICE_STOP_SHOT, blocking=True
            )
        elif (
            hasattr(self.entity_description, "press_fn")
            and self.entity_description.press_fn
        ):
            await self.entity_description.press_fn(self.coordinator)
        else:
            # Fallback or error if no press_fn and not a special handled key
            # This case should ideally not be reached if BUTTONS tuple is correctly defined
            pass
