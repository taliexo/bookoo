import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
import pytest_asyncio  # Import the module itself
from aiobookoov2.bookooscale import BookooScale
from homeassistant.core import HomeAssistant  # Import HomeAssistant

from custom_components.bookoo.const import (
    CONF_IS_VALID_SCALE,
    DOMAIN,
)

# Ensure these are in const.py
from custom_components.bookoo.coordinator import BookooConfigEntry, BookooCoordinator

# This is needed to allow hass.data[DOMAIN] to be set up
# If your tests use pytest-homeassistant-custom-component, it might handle this.
pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture
def mock_config_entry() -> MagicMock:
    """Mock ConfigEntry."""
    entry = MagicMock(spec=BookooConfigEntry)
    entry.entry_id = "test_entry_id"
    entry.domain = DOMAIN
    entry.data = {"address": "XX:XX:XX:XX:XX:XX", CONF_IS_VALID_SCALE: True}
    entry.options = {}  # Add default options if your coordinator uses them
    entry.title = "Bookoo Test Scale"  # Add a title
    entry.runtime_data = None  # Will be set by async_setup_entry
    return entry


@pytest.fixture
def mock_scale(mock_config_entry: MagicMock) -> MagicMock:
    """Mock BookooScale instance."""
    # Use spec for method signature checking, but not spec_set for attribute assignment flexibility.
    scale = MagicMock(spec=BookooScale, instance=True)

    # Set standard attributes
    scale.mac = mock_config_entry.data["address"]

    # Assign PropertyMocks for attributes that behave like properties
    scale.name = PropertyMock(return_value=f"Bookoo {scale.mac}")
    scale.model = PropertyMock(return_value="TestModel")
    scale.connected = PropertyMock(return_value=False)
    scale.connected = PropertyMock(return_value=False)
    scale.weight = PropertyMock(return_value=0.0)
    scale.flow_rate = PropertyMock(return_value=0.0)
    scale.timer = PropertyMock(return_value=0.0)
    scale.device_state = PropertyMock(
        return_value=None
    )  # Mock initial device state as None or a default

    # Assign AsyncMocks for async methods
    scale.async_connect = AsyncMock(return_value=True)
    scale.async_disconnect = AsyncMock(return_value=True)
    scale.process_queue = (
        AsyncMock()
    )  # Configure further in tests if specific behavior is needed
    # Ensure that the return value of process_queue is awaitable if it's awaited in code
    # For example, if process_queue is an async generator, its mock might need special setup.
    # If it's a simple async method that returns None:
    scale.process_queue.return_value = None
    scale.process_queue_task = None  # Initialize as None, like in the actual class

    return scale


@pytest_asyncio.fixture
async def coordinator(
    hass: HomeAssistant,  # Provided by pytest-homeassistant-custom-component
    mock_config_entry: MagicMock,
    mock_scale: MagicMock,
    mocker: MagicMock,  # pytest-mock fixture
) -> BookooCoordinator:
    """Mock BookooCoordinator instance."""

    mocker.patch(
        "custom_components.bookoo.coordinator.BookooScale", return_value=mock_scale
    )

    # Assuming BookooCoordinator.__init__ is synchronous.
    coord = BookooCoordinator(hass, mock_config_entry)

    # Mock async_config_entry_first_refresh. Tests can await this if they call it.
    coord.async_config_entry_first_refresh = AsyncMock()  # type: ignore[method-assign]

    # Store the coordinator in actual_hass.data.
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][mock_config_entry.entry_id] = coord
    mock_config_entry.runtime_data = coord

    return coord


# Add project root to sys.path to allow imports like 'custom_components.bookoo'
project_root = Path(__file__).parent.parent  # Go up one level from 'tests' dir
sys.path.insert(0, str(project_root))

# pytest-homeassistant-custom-component generally handles mocking the HA environment.
# Manual shims below are removed to avoid conflicts.
