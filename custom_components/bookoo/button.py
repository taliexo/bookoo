"""Button entities for Bookoo scales."""

import logging  # Added for _LOGGER
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddEntitiesCallback

# Removed unused SERVICE_START_SHOT, SERVICE_STOP_SHOT, DOMAIN as buttons now call coordinator directly
# from .const import (
#     DOMAIN,
#     SERVICE_START_SHOT,
#     SERVICE_STOP_SHOT,
# )
from .coordinator import (
    BookooConfigEntry,
    BookooCoordinator,
)
from .entity import BookooEntity

PARALLEL_UPDATES = 0
_LOGGER = logging.getLogger(__name__)  # Added logger


@dataclass(kw_only=True, frozen=True)
class BookooButtonEntityDescription(ButtonEntityDescription):
    """Describes a Bookoo button entity.

    Attributes:
        press_fn: Coroutine function to call when the button is pressed.
                  It receives the BookooCoordinator instance.
    """

    # press_fn is now mandatory for this pattern
    press_fn: Callable[[BookooCoordinator], Coroutine[Any, Any, None]]


BUTTONS: tuple[BookooButtonEntityDescription, ...] = (
    BookooButtonEntityDescription(
        key="tare",
        translation_key="tare",
        icon="mdi:target",
        press_fn=lambda coordinator: coordinator.scale.tare(),
    ),
    BookooButtonEntityDescription(
        key="reset_timer",
        translation_key="reset_timer",
        icon="mdi:timer-refresh-outline",
        press_fn=lambda coordinator: coordinator.scale.reset_timer(),
    ),
    BookooButtonEntityDescription(
        key="start",
        translation_key="start",
        icon="mdi:play",
        press_fn=lambda coordinator: coordinator.scale.start_timer(),
    ),
    BookooButtonEntityDescription(
        key="stop",
        translation_key="stop",
        icon="mdi:stop",
        press_fn=lambda coordinator: coordinator.scale.stop_timer(),
    ),
    BookooButtonEntityDescription(
        key="tare_and_start",
        translation_key="tare_and_start",
        icon="mdi:target-account",
        press_fn=lambda coordinator: coordinator.scale.tare_and_start_timer(),
    ),
    BookooButtonEntityDescription(
        key="start_shot_session",
        translation_key="start_shot_session",
        icon="mdi:play-circle-outline",
        # Pass a dummy ServiceCall object or ensure coordinator method handles None
        press_fn=lambda coordinator: coordinator.async_start_shot_service(
            ServiceCall(
                hass=coordinator.hass, domain="bookoo", service="start_shot_session"
            )
        ),
    ),
    BookooButtonEntityDescription(
        key="stop_shot_session",
        translation_key="stop_shot_session",
        icon="mdi:stop-circle-outline",
        # Pass a dummy ServiceCall object or ensure coordinator method handles None
        press_fn=lambda coordinator: coordinator.async_stop_shot_service(
            ServiceCall(
                hass=coordinator.hass, domain="bookoo", service="stop_shot_session"
            )
        ),
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: BookooConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Bookoo button entities based on the config entry.

    Creates button entities for various scale commands and session controls.
    """

    coordinator = entry.runtime_data
    async_add_entities(
        BookooButton(coordinator, description) for description in BUTTONS
    )


class BookooButton(BookooEntity, ButtonEntity):
    """Representation of a Bookoo button entity.

    Each button corresponds to an action that can be performed on the
    Bookoo scale or related to shot sessions.
    """

    entity_description: BookooButtonEntityDescription

    async def async_press(self) -> None:
        """Handle the button press.

        Calls the `press_fn` associated with this button's description,
        which executes the corresponding action on the coordinator or scale.
        Logs an error if the scale is not connected or if the action fails.
        """
        # press_fn is now guaranteed by BookooButtonEntityDescription
        # No need to check hasattr or if it's None, unless it was made optional again.
        # Assuming press_fn is mandatory as per the dataclass change.
        try:
            await self.entity_description.press_fn(self.coordinator)
        except Exception as e:
            _LOGGER.error(
                "Error pressing button %s: %s",
                self.entity_description.key,
                e,
                exc_info=True,
            )
            # Optionally, re-raise or handle specific exceptions if needed
            raise HomeAssistantError(
                f"Error pressing button {self.entity_description.key}: {e}"
            ) from e
