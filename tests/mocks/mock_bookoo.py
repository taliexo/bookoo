"""Mock for the Bookoo integration."""

from unittest.mock import MagicMock, AsyncMock


class MockBookooScale:
    """Mock for the BookooScale class."""

    def __init__(self, *args, **kwargs):
        """Initialize the mock."""
        self.connected = False
        self.mac = kwargs.get("address", "AA:BB:CC:DD:EE:FF")
        self.name = kwargs.get("name", "Test Bookoo Scale")
        self.model = "Themis"
        self.weight = 0.0
        self.flow_rate = 0.0
        self.timer = 0.0
        self.device_state = MagicMock(battery_level=85)
        self.connect = AsyncMock()
        self.disconnect = AsyncMock()
        self.tare = AsyncMock()
        self.reset_timer = AsyncMock()
        self.start_timer = AsyncMock()
        self.stop_timer = AsyncMock()
        self.tare_and_start_timer = AsyncMock()
        self.notify_callback = MagicMock()


class MockBookooCoordinator:
    """Mock for the BookooCoordinator class."""

    def __init__(self, *args, **kwargs):
        """Initialize the mock."""
        self.hass = kwargs.get("hass")
        self.entry = kwargs.get("entry")
        self.scale = MockBookooScale()
        self.async_config_entry_first_refresh = AsyncMock()
        self.async_start_shot_service = AsyncMock()
        self.async_stop_shot_service = AsyncMock()
        self.async_connect_scale_service = AsyncMock()
        self.async_disconnect_scale_service = AsyncMock()
        self.async_update_listeners = MagicMock()
        self._handle_characteristic_update = MagicMock()


class MockBookooConfigEntry:
    """Mock for the BookooConfigEntry class."""

    def __init__(self, *args, **kwargs):
        """Initialize the mock."""
        self.data = kwargs.get("data", {})
        self.entry_id = "test_entry_id"
        self.domain = "bookoo"
        self.title = "Test Bookoo Scale"
        self.unique_id = self.data.get("address", "AA:BB:CC:DD:EE:FF")
        self.runtime_data = None
