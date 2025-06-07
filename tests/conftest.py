# tests/conftest.py
import os

os.environ["BLEAK_FORCE_COREBLUETOOTH"] = "1"

import sys
from unittest.mock import MagicMock

# Aggressively mock dbus_next and dbus_next.aio at the global level.
# This runs as soon as conftest.py is imported by pytest, before test collection for specific files.
_mock_dbus_module = MagicMock()
_mock_dbus_aio_module = MagicMock()
_mock_dbus_module.aio = _mock_dbus_aio_module
_mock_dbus_aio_module.MessageBus = MagicMock()

# Attempt to delete any pre-existing dbus_next modules before applying mocks
# This is to handle cases where they might be partially loaded or in a strange state.
if "dbus_next.aio" in sys.modules:
    del sys.modules["dbus_next.aio"]
if "dbus_next" in sys.modules:
    del sys.modules["dbus_next"]

sys.modules["dbus_next"] = _mock_dbus_module
sys.modules["dbus_next.aio"] = _mock_dbus_aio_module

import pytest
from unittest.mock import (
    AsyncMock,
)  # Use alias to avoid conflict with pytest's patch & import AsyncMock


from tests.fixtures.bluetooth_data import MOCK_SERVICE_INFO

# MOCK_DEVICE_INFO is imported and available for other tests/fixtures if needed.


@pytest.fixture
def mock_bluetooth(enable_bluetooth):
    """Mock bluetooth discovery."""
    with patch(
        "homeassistant.components.bluetooth.async_discovered_service_info",
        return_value=[MOCK_SERVICE_INFO],  # Uses imported MOCK_SERVICE_INFO
    ):
        yield


@pytest.fixture
def mock_bookoo_scale():
    """Mock BookooScale instance."""
    # Path to BookooScale needs to be absolute from the perspective of where pytest runs
    # or relative to a path in sys.path. Assuming custom_components is in sys.path or discovered.
    with patch("custom_components.bookoo.coordinator.BookooScale") as mock:
        instance = mock.return_value
        instance.connected = True
        instance.mac = "AA:BB:CC:DD:EE:FF"
        instance.name = "Test Bookoo Scale"
        instance.model = "Themis"
        instance.weight = 0.0
        instance.flow_rate = 0.0
        instance.timer = 0.0
        instance.device_state = MagicMock(battery_level=85)
        instance.connect = AsyncMock()
        instance.disconnect = AsyncMock()
        instance.tare = AsyncMock()
        instance.reset_timer = AsyncMock()
        instance.start_timer = AsyncMock()
        instance.stop_timer = AsyncMock()
        instance.tare_and_start_timer = AsyncMock()
        instance.notify_callback = MagicMock()
        yield instance


@pytest.fixture
async def init_integration(hass, mock_config_entry, mock_bookoo_scale):
    """Set up the Bookoo integration."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    return mock_config_entry
