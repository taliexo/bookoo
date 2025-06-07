"""Mock Bluetooth data for Bookoo Scale tests."""

# Example MAC address, consistent across mocks for a single device
MOCK_MAC_ADDRESS = "AA:BB:CC:DD:EE:FF"

MOCK_DEVICE_INFO = {
    "address": MOCK_MAC_ADDRESS,
    "name": "Bookoo Themis",  # Example name
    "rssi": -60,
    "details": {},
    # Add other fields as necessary for homeassistant.components.bluetooth.models.BluetoothDevice
    # For example, if your component uses specific advertisement data, mock it here.
}

MOCK_SERVICE_INFO = {
    "address": MOCK_MAC_ADDRESS,
    "name": "Bookoo Themis",  # Should match MOCK_DEVICE_INFO if from the same device
    "rssi": -60,
    "service_uuids": [
        "0000180d-0000-1000-8000-00805f9b34fb"
    ],  # Example: Heart Rate Service UUID
    "manufacturer_data": {
        76: b"\x01\x02\x03\x04\x05\x06"
    },  # Example manufacturer data (Apple, Inc.)
    "service_data": {
        "0000180d-0000-1000-8000-00805f9b34fb": b"\x00\x01\x02"  # Example service data for the UUID
    },
    "source": "local",
    # Add other fields as necessary for homeassistant.components.bluetooth.models.BluetoothServiceInfoBleak
}
