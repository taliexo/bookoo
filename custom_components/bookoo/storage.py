"""Handles storage for Bookoo shot history using Home Assistant's Store."""

import logging
from typing import Any, cast

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .types import (
    BookooShotCompletedEventDataModel,  # Assuming this is the Pydantic model
)

_LOGGER = logging.getLogger(__name__)

STORAGE_KEY = "bookoo_shot_history"
STORAGE_VERSION = 1

# Consider adding a check for old SQLite DB and logging a one-time migration notice.
_LOGGER.warning(
    "Bookoo integration now uses Home Assistant's internal storage for shot history. "
    "Existing data from bookoo_shots.db (SQLite) will not be automatically migrated. "
    "New shots will be saved to the new format."
)


def _get_store(hass: HomeAssistant) -> Store[list[dict[str, Any]]]:
    """Get the Store instance for shot history.

    Initializes and returns a Home Assistant Store object for persisting
    shot data as a list of dictionaries.
    """
    # The Store will hold a list of shot data dictionaries.
    return cast(Store[list[dict[str, Any]]], Store(hass, STORAGE_VERSION, STORAGE_KEY))


async def async_add_shot_record(
    hass: HomeAssistant, shot_data: BookooShotCompletedEventDataModel
) -> None:
    """Add a shot record to the persistent Store.

    Loads the current history, appends the new shot data (after converting
    the Pydantic model to a dictionary), and saves the updated history.

    Args:
        hass: The HomeAssistant instance.
        shot_data: The Pydantic model instance of the completed shot.
    """
    store = _get_store(hass)
    try:
        shot_history = await store.async_load() or []
    except Exception as e:  # pylint: disable=broad-except
        _LOGGER.error(
            "Error loading shot history from store: %s. Initializing new history.",
            e,
            exc_info=True,
        )
        shot_history = []

    # Convert Pydantic model to dict for storage, if it's not already a dict
    # If BookooShotCompletedEventDataModel is a Pydantic model, use .model_dump()
    shot_data_dict = shot_data.model_dump(
        mode="json"
    )  # mode='json' ensures datetimes are ISO strings

    shot_history.append(shot_data_dict)

    try:
        await store.async_save(shot_history)
        _LOGGER.debug(
            "Successfully added shot record to HA Store: %s",
            shot_data_dict.get(
                "start_time_utc"
            ),  # Use start_time_utc from Pydantic model
        )
    except Exception as e:  # pylint: disable=broad-except
        _LOGGER.error(
            "Error saving shot record to HA Store %s: %s",
            shot_data_dict.get("start_time_utc"),
            e,
            exc_info=True,
        )


async def async_get_shot_history(
    hass: HomeAssistant, limit: int | None = None
) -> list[BookooShotCompletedEventDataModel]:
    """Retrieve shot history from the Store, optionally limited.

    Loads shot data dictionaries from the Store, validates them back into
    Pydantic models, and returns a list of these models.

    Args:
        hass: The HomeAssistant instance.
        limit: Optional maximum number of recent shots to return.

    Returns:
        A list of BookooShotCompletedEventDataModel instances.
    """
    store = _get_store(hass)
    try:
        shot_history_dicts = await store.async_load() or []
    except Exception as e:  # pylint: disable=broad-except
        _LOGGER.error(
            "Error loading shot history from store: %s. Returning empty list.",
            e,
            exc_info=True,
        )
        return []

    # Validate and parse back to Pydantic models
    validated_shots: list[BookooShotCompletedEventDataModel] = []
    for shot_dict in shot_history_dicts:
        try:
            validated_shots.append(BookooShotCompletedEventDataModel(**shot_dict))
        except Exception as e:  # pylint: disable=broad-except
            _LOGGER.warning(
                "Skipping invalid shot data during retrieval: %s. Error: %s",
                shot_dict.get("start_time_utc"),
                e,
            )
            continue  # Skip invalid records

    if limit is not None and limit > 0:
        return validated_shots[-limit:]  # Return the most recent 'limit' shots
    return validated_shots


async def async_delete_shot_history(hass: HomeAssistant) -> None:
    """Delete all shot history from the Store.

    Removes the entire storage file containing shot history.

    Args:
        hass: The HomeAssistant instance.
    """
    store = _get_store(hass)
    try:
        await store.async_remove()  # Removes the entire store file
        _LOGGER.info("Successfully deleted all Bookoo shot history from HA Store.")
    except Exception as e:  # pylint: disable=broad-except
        _LOGGER.error(
            "Error deleting Bookoo shot history from HA Store: %s", e, exc_info=True
        )


# Note: `async_init_db` is no longer needed as Store handles its own initialization.
