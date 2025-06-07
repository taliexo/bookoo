# tests/integration/test_config_flow.py
from homeassistant import config_entries
from custom_components.bookoo.const import DOMAIN
# MOCK_SERVICE_INFO is now expected to be provided by conftest.py


async def test_bluetooth_discovery_flow(hass, mock_bluetooth, MOCK_SERVICE_INFO):
    """Test the Bluetooth discovery flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN,
        context={"source": config_entries.SOURCE_BLUETOOTH},
        data=MOCK_SERVICE_INFO,  # Uses MOCK_SERVICE_INFO from conftest.py
    )
    assert result["type"] == "form"
    assert result["step_id"] == "bluetooth_confirm"


async def test_user_flow(hass, mock_bluetooth):
    """Test the user configuration flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == "form"
    assert result["step_id"] == "user"
