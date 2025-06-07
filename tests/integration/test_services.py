# tests/integration/test_services.py
import pytest
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from custom_components.bookoo.const import (
    DOMAIN,
    SERVICE_START_SHOT,
    SERVICE_STOP_SHOT,
    SERVICE_CONNECT_SCALE,
    SERVICE_DISCONNECT_SCALE,
)
from custom_components.bookoo.coordinator import BookooCoordinator

from pytest_homeassistant_custom_component.common import MockConfigEntry


@pytest.fixture(autouse=True)
async def mock_bluetooth(enable_bluetooth: None) -> None:
    """Enable bluetooth fixture."""
    # This fixture is often needed for custom components using Bluetooth
    pass


@pytest.fixture
async def async_setup_bookoo_integration(
    hass: HomeAssistant, mock_bluetooth
) -> MockConfigEntry:
    """Set up the Bookoo integration with a mock config entry."""
    config_entry = MockConfigEntry(
        domain=DOMAIN, data={}, unique_id="test_bookoo_device"
    )
    config_entry.add_to_hass(hass)

    # Mock the BookooScale connection during setup to avoid actual Bluetooth interactions
    with patch(
        "aiobookoov2.bookooscale.BookooScale.connect",
        new_callable=AsyncMock,
    ):
        with patch(
            "custom_components.bookoo.BookooCoordinator._async_update_data",
            new_callable=AsyncMock,
        ):  # Prevent coordinator from running _async_update_data
            assert await async_setup_component(hass, DOMAIN, {DOMAIN: {}})
            await hass.async_block_till_done()

    return config_entry


@pytest.mark.asyncio
async def test_service_start_shot(
    hass: HomeAssistant, async_setup_bookoo_integration: MockConfigEntry
):
    """Test the start_shot service call."""
    with patch.object(
        BookooCoordinator, "async_start_shot_service", new_callable=AsyncMock
    ) as mock_service_method:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_START_SHOT,
            blocking=True,
        )
        mock_service_method.assert_called_once()


@pytest.mark.asyncio
async def test_service_stop_shot(
    hass: HomeAssistant, async_setup_bookoo_integration: MockConfigEntry
):
    """Test the stop_shot service call."""
    with patch.object(
        BookooCoordinator, "async_stop_shot_service", new_callable=AsyncMock
    ) as mock_service_method:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_STOP_SHOT,
            blocking=True,
        )
        mock_service_method.assert_called_once()


@pytest.mark.asyncio
async def test_service_connect_scale(
    hass: HomeAssistant, async_setup_bookoo_integration: MockConfigEntry
):
    """Test the connect_scale service call."""
    with patch.object(
        BookooCoordinator, "async_connect_scale_service", new_callable=AsyncMock
    ) as mock_service_method:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_CONNECT_SCALE,
            blocking=True,
        )
        mock_service_method.assert_called_once()


@pytest.mark.asyncio
async def test_service_disconnect_scale(
    hass: HomeAssistant, async_setup_bookoo_integration: MockConfigEntry
):
    """Test the disconnect_scale service call."""
    with patch.object(
        BookooCoordinator, "async_disconnect_scale_service", new_callable=AsyncMock
    ) as mock_service_method:
        await hass.services.async_call(
            DOMAIN,
            SERVICE_DISCONNECT_SCALE,
            blocking=True,
        )
        mock_service_method.assert_called_once()
