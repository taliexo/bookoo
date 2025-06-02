import sys
from unittest.mock import MagicMock

# --- Start Mocks for Home Assistant ---
mock_hass_const = MagicMock()
mock_hass_const.Platform = MagicMock()
mock_hass_const.EVENT_HOMEASSISTANT_STOP = "mock_ha_stop"
mock_hass_const.CONF_ADDRESS = "address"
mock_hass_const.CONF_IS_VALID_SCALE = "is_valid_scale"
mock_hass_const.STATE_UNAVAILABLE = "unavailable"
mock_hass_const.STATE_UNKNOWN = "unknown"

mock_hass_core = MagicMock()


class MockDataUpdateCoordinator:
    def __init__(self, hass, logger, *, name, update_interval, config_entry=None):
        self.hass = hass
        self.logger = logger
        self.name = name
        self.update_interval = update_interval
        self.config_entry = config_entry
        self.data = None
        self._listeners = []

    async def _async_update_data(self):
        pass

    def async_add_listener(self, update_callback, context=None):
        pass

    def async_update_listeners(self):
        pass

    @classmethod
    def __class_getitem__(cls, item):
        return cls


mock_hass_helpers_update_coordinator = MagicMock()
mock_hass_helpers_update_coordinator.DataUpdateCoordinator = MockDataUpdateCoordinator
mock_hass_helpers_update_coordinator.UpdateFailed = type(
    "UpdateFailed", (Exception,), {}
)


class MockConfigEntry:
    @classmethod
    def __class_getitem__(cls, item):
        """Allow class to be subscripted (e.g., MockConfigEntry[Any])."""
        return cls


mock_hass_config_entries = MagicMock()
mock_hass_config_entries.ConfigEntry = MockConfigEntry

mock_dt_util = MagicMock()
mock_util = MagicMock()
mock_util.dt = mock_dt_util

sys.modules["homeassistant.const"] = mock_hass_const
sys.modules["homeassistant.core"] = mock_hass_core
sys.modules["homeassistant.helpers.update_coordinator"] = (
    mock_hass_helpers_update_coordinator
)
sys.modules["homeassistant.config_entries"] = mock_hass_config_entries
sys.modules["homeassistant.util"] = mock_util
sys.modules["homeassistant.util.dt"] = mock_dt_util
# --- End Mocks for Home Assistant ---
