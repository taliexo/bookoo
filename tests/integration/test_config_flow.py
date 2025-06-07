"""Test the Bookoo config flow."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from homeassistant import config_entries
from homeassistant.components.bluetooth import BluetoothServiceInfoBleak
from homeassistant.data_entry_flow import FlowResultType

from custom_components.bookoo.const import DOMAIN


class MockBookooScale:
    """Mock BookooScale class for testing."""

    def __init__(self):
        """Initialize the mock."""
        self.connected = False
        self.mac = "AA:BB:CC:DD:EE:FF"
        self.name = "Test Bookoo Scale"
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


# Mock data for testing
MOCK_BLUETOOTH_DATA = {
    "name": "BOOKOO_SCALE",
    "address": "AA:BB:CC:DD:EE:FF",
    "rssi": -60,
    "manufacturer_data": {},
    "service_data": {
        "00000ffe-0000-1000-8000-00805f9b34fb": {
            "0000ff11-0000-1000-8000-00805f9b34fb": b"\x01",
            "0000ff12-0000-1000-8000-00805f9b34fb": b"\x02",
        }
    },
    "service_uuids": ["00000ffe-0000-1000-8000-00805f9b34fb"],
    "source": "local",
    "device": None,
    "advertisement": None,
    "connectable": True,
    "time": 0,
    "tx_power": 0,
}


@pytest.fixture
def mock_setup_entry():
    """Mock setting up a config entry."""
    with patch(
        "custom_components.bookoo.async_setup_entry", return_value=True
    ) as mock_setup:
        yield mock_setup


@pytest.fixture
def mock_bookoo_scale():
    """Mock the BookooScale class."""
    with patch("aiobookoov2.bookooscale.BookooScale") as mock:
        mock.return_value = MockBookooScale()
        yield mock.return_value


async def test_bluetooth_discovery_flow(
    hass, mock_bluetooth, mock_setup_entry, mock_bookoo_scale
):
    """Test the Bluetooth discovery flow."""
    # Create a mock BluetoothServiceInfoBleak object
    service_info = BluetoothServiceInfoBleak(
        name="BOOKOO_SCALE",
        address="AA:BB:CC:DD:EE:FF",
        rssi=-60,
        manufacturer_data={},
        service_data={
            "00000ffe-0000-1000-8000-00805f9b34fb": {
                "0000ff11-0000-1000-8000-00805f9b34fb": b"\x01",
                "0000ff12-0000-1000-8000-00805f9b34fb": b"\x02",
            }
        },
        service_uuids=["00000ffe-0000-1000-8000-00805f9b34fb"],
        source="local",
        device=None,
        advertisement=None,
        connectable=True,
        time=0,
        tx_power=0,
    )

    # Mock the discovered service info
    with patch(
        "homeassistant.components.bluetooth.async_discovered_service_info",
        return_value=[service_info],
    ) as _mock_discovered_service_info:
        # Start the discovery flow
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_BLUETOOTH},
            data=service_info,
        )

        # Check that we're showing the confirmation form
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "bluetooth_confirm"
        assert result["description_placeholders"]["name"] == "BOOKOO_SCALE"

        # Submit the form
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], user_input={}
        )

        # Check that the entry was created
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "BOOKOO_SCALE"
        assert result["data"] == {
            "name": "BOOKOO_SCALE",
            "address": "AA:BB:CC:DD:EE:FF",
        }

        # Verify the setup was called
        assert len(mock_setup_entry.mock_calls) == 1


async def test_user_flow(hass, mock_bluetooth, mock_setup_entry, mock_bookoo_scale):
    """Test the user configuration flow."""
    # Create a mock BluetoothServiceInfoBleak object
    service_info = BluetoothServiceInfoBleak(
        name="BOOKOO_SCALE",
        address="AA:BB:CC:DD:EE:FF",
        rssi=-60,
        manufacturer_data={},
        service_data={
            "00000ffe-0000-1000-8000-00805f9b34fb": {
                "0000ff11-0000-1000-8000-00805f9b34fb": b"\x01",
                "0000ff12-0000-1000-8000-00805f9b34fb": b"\x02",
            }
        },
        service_uuids=["00000ffe-0000-1000-8000-00805f9b34fb"],
        source="local",
        device=None,
        advertisement=None,
        connectable=True,
        time=0,
        tx_power=0,
    )

    # Mock the discovered service info
    with patch(
        "homeassistant.components.bluetooth.async_discovered_service_info",
        return_value=[service_info],
    ) as _mock_discovered_service_info:
        # Start the user flow
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )

        # Check that we're showing the device selection form
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"

        # Select a device
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            user_input={"address": "AA:BB:CC:DD:EE:FF"},
        )

        # Check that the entry was created
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["title"] == "BOOKOO_SCALE"
        assert result["data"] == {
            "name": "BOOKOO_SCALE",
            "address": "AA:BB:CC:DD:EE:FF",
        }

        # Verify the setup was called
        assert len(mock_setup_entry.mock_calls) == 1
