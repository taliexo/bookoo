import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path to allow imports like 'custom_components.bookoo'
project_root = Path(__file__).parent.parent  # Go up one level from 'tests' dir
sys.path.insert(0, str(project_root))


# --- Start Mocks for Home Assistant ---
mock_hass_const = MagicMock()
mock_hass_const.Platform = MagicMock()
mock_hass_const.EVENT_HOMEASSISTANT_STOP = "mock_ha_stop"
mock_hass_const.CONF_ADDRESS = "address"
mock_hass_const.CONF_IS_VALID_SCALE = "is_valid_scale"
mock_hass_const.STATE_UNAVAILABLE = "unavailable"
mock_hass_const.STATE_UNKNOWN = "unknown"

mock_hass_core = MagicMock()


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
sys.modules["homeassistant.config_entries"] = mock_hass_config_entries
sys.modules["homeassistant.util"] = mock_util
sys.modules["homeassistant.util.dt"] = mock_dt_util
# --- End Mocks for Home Assistant ---
