"""Mock Bluetooth components for testing."""

from unittest.mock import AsyncMock, patch

# Mock Bluetooth components
mock_async_get_advertisement_callback = AsyncMock()
mock_async_register_callback = AsyncMock()
mock_async_ble_device_from_address = AsyncMock()
mock_async_get_bluetooth_adapters = AsyncMock()
mock_async_get_bluetooth_adapter_from_address = AsyncMock()
mock_async_get_bluetooth_advertisement_callback = AsyncMock()
mock_async_rediscover_address = AsyncMock()
mock_async_track_unavailable = AsyncMock()
mock_async_register_scanner = AsyncMock()
mock_async_discovered_service_info = AsyncMock()
mock_async_scanner_count = AsyncMock(return_value=1)
mock_async_scanner_devices_by_address = AsyncMock()


def apply_bluetooth_mocks():
    """Apply all Bluetooth mocks."""
    return (
        patch(
            "homeassistant.components.bluetooth.async_get_advertisement_callback",
            mock_async_get_advertisement_callback,
        ),
        patch(
            "homeassistant.components.bluetooth.async_register_callback",
            mock_async_register_callback,
        ),
        patch(
            "homeassistant.components.bluetooth.async_ble_device_from_address",
            mock_async_ble_device_from_address,
        ),
        patch(
            "homeassistant.components.bluetooth.async_get_bluetooth_adapters",
            mock_async_get_bluetooth_adapters,
        ),
        patch(
            "homeassistant.components.bluetooth.async_get_bluetooth_adapter_from_address",
            mock_async_get_bluetooth_adapter_from_address,
        ),
        patch(
            "homeassistant.components.bluetooth.async_get_bluetooth_advertisement_callback",
            mock_async_get_bluetooth_advertisement_callback,
        ),
        patch(
            "homeassistant.components.bluetooth.async_rediscover_address",
            mock_async_rediscover_address,
        ),
        patch(
            "homeassistant.components.bluetooth.async_track_unavailable",
            mock_async_track_unavailable,
        ),
        patch(
            "homeassistant.components.bluetooth.async_register_scanner",
            mock_async_register_scanner,
        ),
        patch(
            "homeassistant.components.bluetooth.async_discovered_service_info",
            mock_async_discovered_service_info,
        ),
        patch(
            "homeassistant.components.bluetooth.async_scanner_count",
            mock_async_scanner_count,
        ),
        patch(
            "homeassistant.components.bluetooth.async_scanner_devices_by_address",
            mock_async_scanner_devices_by_address,
        ),
    )
