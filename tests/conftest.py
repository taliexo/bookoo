# tests/conftest.py
import os
import sys
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from homeassistant import config_entries
from homeassistant.core import HomeAssistant

# Set up test environment
os.environ["BLEAK_FORCE_COREBLUETOOTH"] = "1"
os.environ["HASS_NO_BLUETOOTH"] = "1"  # Disable Bluetooth in tests

# Add the custom_components directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Mock dbus_next and dbus_next.aio at the global level
_mock_dbus_module = MagicMock()
_mock_dbus_aio_module = MagicMock()
_mock_dbus_module.aio = _mock_dbus_aio_module
_mock_dbus_aio_module.MessageBus = MagicMock()

# Clean up any existing modules
for module in ["dbus_next", "dbus_next.aio"]:
    if module in sys.modules:
        del sys.modules[module]

sys.modules["dbus_next"] = _mock_dbus_module
sys.modules["dbus_next.aio"] = _mock_dbus_aio_module


# Import test fixtures


# Mock the bluetooth module
@pytest.fixture(autouse=True)
def mock_bluetooth(enable_bluetooth):
    """Auto mock bluetooth."""
    # Create a mock Bluetooth adapter
    mock_adapter = MagicMock()
    mock_adapter.address = "00:00:00:00:00:00"
    mock_adapter.hci = 0
    mock_adapter.name = "hci0"
    mock_adapter.version = "5.63"
    mock_adapter.history = {}

    # Create a mock Bluetooth manager
    mock_manager = MagicMock()
    mock_manager.async_setup = AsyncMock()
    mock_manager.async_stop = AsyncMock()
    mock_manager.scanner_adv_history = {}
    mock_manager.async_all_discovered_addresses = AsyncMock(return_value=[])
    mock_manager.async_discovered_service_info = AsyncMock(return_value=[])

    # Mock bluetooth components with the manager and adapter
    with (
        patch.multiple(
            "homeassistant.components.bluetooth",
            async_get_advertisement_callback=AsyncMock(),
            async_register_callback=AsyncMock(),
            async_ble_device_from_address=AsyncMock(),
            async_rediscover_address=AsyncMock(),
            async_track_unavailable=AsyncMock(),
            async_register_scanner=AsyncMock(),
            async_discovered_service_info=AsyncMock(return_value=[]),
            async_scanner_count=AsyncMock(return_value=1),
            async_scanner_devices_by_address=AsyncMock(return_value=[]),
            # async_get_bluetooth_adapters=AsyncMock(return_value=[mock_adapter]), # Removed, will be set directly
            # async_get_bluetooth_adapter_from_address=AsyncMock(return_value=mock_adapter), # Removed, will be set directly
        ),
        patch.multiple(
            "homeassistant.components.bluetooth.manager",
            BluetoothManager=MagicMock(),
        ),
        patch.multiple(
            "bluetooth_adapters.systems.linux",
            LinuxAdapters=MagicMock(),
        ),
        patch.multiple(
            "bluetooth_adapters.dbus",
            BlueZDBusObjects=MagicMock(),
        ),
    ):
        # Directly mock bluetooth functions on the module
        # This ensures they are set after enable_bluetooth might have configured the module.
        import homeassistant.components.bluetooth as bluetooth_module

        bluetooth_module.async_get_bluetooth_adapters = AsyncMock(
            return_value=[mock_adapter]
        )
        bluetooth_module.async_get_bluetooth_adapter_from_address = AsyncMock(
            return_value=mock_adapter
        )
        yield


# Mock the BookooScale class
@pytest.fixture
def mock_bookoo_scale():
    """Mock BookooScale instance."""
    with patch("aiobookoov2.bookooscale.BookooScale") as mock:
        instance = mock.return_value
        instance.connected = False
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


# Mock config entry
class MockConfigEntry(config_entries.ConfigEntry):
    """Mock ConfigEntry for testing."""

    def __init__(self, **kwargs):
        """Initialize the mock config entry."""
        self.data = kwargs.get("data", {})
        self.entry_id = kwargs.get("entry_id", "test_entry_id")
        self.domain = kwargs.get("domain", "bookoo")
        self.title = kwargs.get("title", "Test Bookoo Scale")
        self.unique_id = kwargs.get("unique_id", "AA:BB:CC:DD:EE:FF")
        self.source = kwargs.get("source", config_entries.SOURCE_USER)
        self.state = config_entries.ConfigEntryState.LOADED
        self.pref_disable_new_entities = False
        self.pref_disable_polling = False
        self.options = {}
        self._hass = None

    def add_to_hass(self, hass):
        """Add the config entry to hass."""
        self._hass = hass
        if not hasattr(hass.config_entries, "_entries"):
            hass.config_entries._entries = []
        hass.config_entries._entries.append(self)

    async def async_unload(self, hass):
        """Unload the config entry."""
        return True


@pytest.fixture
def mock_config_entry():
    """Create a mock config entry."""
    return MockConfigEntry(
        domain="bookoo",
        data={
            "name": "Test Bookoo Scale",
            "address": "AA:BB:CC:DD:EE:FF",
        },
        unique_id="AA:BB:CC:DD:EE:FF",
    )


# Mock the async_setup_entry function for the bookoo integration
@pytest.fixture
def mock_bookoo_integration():
    """Mock the bookoo integration setup."""
    with (
        patch(
            "custom_components.bookoo.async_setup_entry",
            return_value=True,
        ) as mock_setup_entry,
        patch(
            "custom_components.bookoo.config_flow.is_bookoo_scale",
            return_value=True,
        ) as mock_is_bookoo_scale,
    ):
        yield {
            "mock_setup_entry": mock_setup_entry,
            "mock_is_bookoo_scale": mock_is_bookoo_scale,
        }


# Setup integration fixture
@pytest.fixture
async def setup_integration(hass: HomeAssistant, mock_config_entry, mock_bookoo_scale):
    """Set up the Bookoo integration for testing."""
    # Add the config entry
    mock_config_entry.add_to_hass(hass)

    # Setup the integration
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Return the mock config entry and any mocks we want to check
    yield {
        "config_entry": mock_config_entry,
        "mock_bookoo_scale": mock_bookoo_scale,
    }

    # Clean up any lingering timers
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass
