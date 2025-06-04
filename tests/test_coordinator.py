"""Tests for the Bookoo coordinator."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from _pytest.logging import LogCaptureFixture
from aiobookoov2.const import (
    UPDATE_SOURCE_COMMAND_CHAR,
    UPDATE_SOURCE_WEIGHT_CHAR,
)
from aiobookoov2.exceptions import BookooDeviceNotFound, BookooError
from freezegun.api import FrozenDateTimeFactory  # Added import
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.update_coordinator import UpdateFailed
from pytest_mock import MockerFixture

from custom_components.bookoo.coordinator import BookooCoordinator
from custom_components.bookoo.session_manager import SessionManager

# Fixtures are in conftest.py


class TestBookooCoordinator:
    """Test cases for BookooCoordinator."""

    def _assert_log_message(
        self,
        caplog: LogCaptureFixture,
        level: str,
        logger_name: str,
        message_substring: str,
    ):
        """Helper to assert a log message is present."""
        found_log = False
        for record in caplog.records:
            if (
                record.levelname == level
                and record.name == logger_name
                and message_substring in record.message
            ):
                found_log = True
                break
        assert found_log, f"Expected {level} log from {logger_name} containing '{message_substring}' not found. Logs: {caplog.text}"
        caplog.clear()

    @pytest.mark.asyncio
    async def test_coordinator_initialization(
        self,
        coordinator: BookooCoordinator,
        mock_scale: MagicMock,
        hass: MagicMock,
    ):
        """Test coordinator initialization."""
        assert coordinator.scale is mock_scale
        assert coordinator.hass is hass
        assert coordinator.name == f"Bookoo {mock_scale.mac}"
        assert isinstance(coordinator.session_manager, SessionManager)
        assert coordinator.last_shot_data is None
        assert coordinator.realtime_channeling_status == "Undetermined"

    @pytest.mark.asyncio
    async def test_handle_char_update_weight_data_active_shot(
        self,
        coordinator: BookooCoordinator,
        hass: MagicMock,  # Used by SessionManager
        freezer: FrozenDateTimeFactory,
        caplog,
        mocker: MagicMock,  # pytest-mock fixture, though often not explicitly typed as MagicMock
    ):
        """Test _handle_characteristic_update with weight data during an active shot."""
        caplog.set_level(logging.DEBUG)
        coordinator.session_manager.is_shot_active = True
        coordinator.session_manager.session_start_time_utc = datetime.now(timezone.utc)

        mocker.patch.object(
            coordinator.scale, "weight", new_callable=PropertyMock, return_value=10.0
        )
        mocker.patch.object(
            coordinator.scale, "flow_rate", new_callable=PropertyMock, return_value=1.5
        )
        mocker.patch.object(
            coordinator.scale, "timer", new_callable=PropertyMock, return_value=5.0
        )

        coordinator._handle_characteristic_update(UPDATE_SOURCE_WEIGHT_CHAR, None)
        await asyncio.sleep(0)

        assert (
            cast(MagicMock, coordinator.session_manager.add_flow_data).call_count == 1
        )
        call_args = cast(MagicMock, coordinator.session_manager.add_flow_data).call_args
        assert call_args is not None
        assert call_args.args[0] == pytest.approx(0.0)
        assert call_args.args[1] == pytest.approx(1.5)

        assert (
            cast(MagicMock, coordinator.session_manager.add_weight_data).call_count == 1
        )
        call_args_weight = cast(
            MagicMock, coordinator.session_manager.add_weight_data
        ).call_args
        assert call_args_weight is not None
        assert call_args_weight.args[0] == pytest.approx(0.0)
        assert call_args_weight.args[1] == pytest.approx(10.0)

        assert (
            cast(MagicMock, coordinator.session_manager.add_scale_timer_data).call_count
            == 1
        )
        call_args_timer = cast(
            MagicMock, coordinator.session_manager.add_scale_timer_data
        ).call_args
        assert call_args_timer is not None
        assert call_args_timer.args[0] == pytest.approx(0.0)
        assert call_args_timer.args[1] == pytest.approx(5.0)

        cast(
            MagicMock, coordinator.shot_analyzer.detect_channeling
        ).assert_called_once_with(coordinator.session_manager.session_flow_profile)
        cast(
            MagicMock, coordinator.shot_analyzer.identify_pre_infusion
        ).assert_called_once_with(
            coordinator.session_manager.session_flow_profile,
            coordinator.session_manager.session_scale_timer_profile,
        )
        cast(
            MagicMock, coordinator.shot_analyzer.calculate_extraction_uniformity
        ).assert_called_once_with(coordinator.session_manager.session_flow_profile)
        assert coordinator.realtime_channeling_status == "Undetermined"
        cast(AsyncMock, coordinator.async_update_listeners).assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_char_update_weight_data_no_active_shot(
        self, coordinator: BookooCoordinator, caplog
    ):
        """Test _handle_characteristic_update with weight data when no shot is active."""
        caplog.set_level(logging.DEBUG)
        coordinator.session_manager.is_shot_active = False

        cast(MagicMock, coordinator.session_manager.add_flow_data).reset_mock()
        cast(MagicMock, coordinator.session_manager.add_weight_data).reset_mock()
        cast(MagicMock, coordinator.session_manager.add_scale_timer_data).reset_mock()
        cast(MagicMock, coordinator.shot_analyzer.detect_channeling).reset_mock()
        cast(MagicMock, coordinator.shot_analyzer.identify_pre_infusion).reset_mock()
        cast(
            MagicMock, coordinator.shot_analyzer.calculate_extraction_uniformity
        ).reset_mock()
        cast(AsyncMock, coordinator.async_update_listeners).reset_mock()

        with patch.object(
            coordinator, "_update_shot_quality_score", new_callable=MagicMock
        ) as mock_update_quality:
            coordinator._handle_characteristic_update(UPDATE_SOURCE_WEIGHT_CHAR, None)
            await asyncio.sleep(0)

            cast(
                MagicMock, coordinator.session_manager.add_flow_data
            ).assert_not_called()
            cast(
                MagicMock, coordinator.session_manager.add_weight_data
            ).assert_not_called()
            cast(
                MagicMock, coordinator.session_manager.add_scale_timer_data
            ).assert_not_called()

            cast(
                MagicMock, coordinator.shot_analyzer.detect_channeling
            ).assert_not_called()
            cast(
                MagicMock, coordinator.shot_analyzer.identify_pre_infusion
            ).assert_not_called()
            cast(
                MagicMock,
                coordinator.shot_analyzer.calculate_extraction_uniformity,
            ).assert_not_called()
            mock_update_quality.assert_not_called()

            cast(AsyncMock, coordinator.async_update_listeners).assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_char_update_command_char_auto_timer_start_stop(
        self, coordinator: BookooCoordinator, hass: MagicMock, caplog
    ):
        """Test _handle_characteristic_update for command char with auto timer events."""
        caplog.set_level(logging.DEBUG)
        decoded_start_event = {"type": "auto_timer", "event": "start"}

        coordinator.session_manager.is_shot_active = False
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_start_event
        )
        await asyncio.sleep(0)
        cast(MagicMock, coordinator._reset_realtime_analytics).assert_called_once()
        cast(
            AsyncMock, coordinator.session_manager.start_session
        ).assert_called_once_with(trigger="scale_auto_dict")

        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.session_manager.start_session).reset_mock()
        cast(MagicMock, coordinator._reset_realtime_analytics).reset_mock()
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_start_event
        )
        await asyncio.sleep(0)
        cast(MagicMock, coordinator._reset_realtime_analytics).assert_not_called()
        cast(AsyncMock, coordinator.session_manager.start_session).assert_not_called()

        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.session_manager.start_session).reset_mock()
        cast(MagicMock, coordinator._reset_realtime_analytics).reset_mock()
        other_decoded_event = {"type": "other_event", "event": "some_value"}
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, other_decoded_event
        )
        await asyncio.sleep(0)
        cast(AsyncMock, coordinator.session_manager.start_session).assert_not_called()

        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.session_manager.start_session).reset_mock()
        cast(MagicMock, coordinator._reset_realtime_analytics).reset_mock()
        coordinator._handle_characteristic_update(UPDATE_SOURCE_COMMAND_CHAR, None)
        await asyncio.sleep(0)
        cast(AsyncMock, coordinator.session_manager.start_session).assert_not_called()
        cast(AsyncMock, coordinator.async_update_listeners).assert_called()

    @pytest.mark.asyncio
    async def test_handle_char_update_decoded_auto_stop(
        self, coordinator: BookooCoordinator, hass: MagicMock, caplog
    ):
        """Test _handle_characteristic_update for command char with auto timer stop event."""
        caplog.set_level(logging.DEBUG)
        decoded_stop_event = {"type": "auto_timer", "event": "stop"}

        coordinator.session_manager.is_shot_active = True
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_stop_event
        )
        await asyncio.sleep(0)
        cast(
            AsyncMock, coordinator.session_manager.stop_session
        ).assert_called_once_with(stop_reason="scale_auto_dict")

        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_stop_event
        )
        await asyncio.sleep(0)
        hass.async_create_task.assert_not_called()  # hass.async_create_task is from the fixture, not directly called here
        cast(AsyncMock, coordinator.session_manager.stop_session).assert_not_called()

        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        other_decoded_event = {"type": "other_event", "event": "some_value"}
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, other_decoded_event
        )
        await asyncio.sleep(0)
        cast(AsyncMock, coordinator.session_manager.stop_session).assert_not_called()

        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        coordinator._handle_characteristic_update(UPDATE_SOURCE_COMMAND_CHAR, None)
        await asyncio.sleep(0)
        cast(AsyncMock, coordinator.session_manager.stop_session).assert_not_called()
        cast(AsyncMock, coordinator.async_update_listeners).assert_called()

    @pytest.mark.asyncio
    async def test_async_start_shot_service(
        self,
        coordinator: BookooCoordinator,
        hass: MagicMock,
        caplog,
        mocker: MockerFixture,
    ):
        """Test the async_start_shot_service method."""
        caplog.set_level(logging.INFO)
        mock_call = MagicMock(spec=ServiceCall)

        coordinator.session_manager.is_shot_active = False

        # Mock _reset_realtime_analytics to check its call
        mock_reset_analytics = mocker.patch.object(
            coordinator, "_reset_realtime_analytics"
        )

        await coordinator.async_start_shot_service(mock_call)
        mock_reset_analytics.assert_called_once()
        cast(
            AsyncMock, coordinator.session_manager.start_session
        ).assert_called_once_with(trigger="ha_service")
        cast(AsyncMock, coordinator.scale.tare_and_start_timer).assert_called_once()

        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.session_manager.start_session).reset_mock()
        cast(AsyncMock, coordinator.scale.tare_and_start_timer).reset_mock()
        cast(MagicMock, coordinator._reset_realtime_analytics).reset_mock()
        with patch.object(
            logging.getLogger("custom_components.bookoo.coordinator"), "warning"
        ) as mock_log_warning:
            await coordinator.async_start_shot_service(mock_call)
            mock_log_warning.assert_called_once_with(
                "%s: Start shot service called, but a shot is already active.",
                coordinator.name,
            )
        cast(AsyncMock, coordinator.scale.tare_and_start_timer).assert_not_called()
        cast(AsyncMock, coordinator.session_manager.start_session).assert_not_called()
        cast(MagicMock, coordinator._reset_realtime_analytics).assert_not_called()

        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.session_manager.start_session).reset_mock()
        cast(AsyncMock, coordinator.scale.tare_and_start_timer).reset_mock()
        cast(MagicMock, coordinator._reset_realtime_analytics).reset_mock()

        original_side_effect = BookooError("Scale command failed")
        cast(
            AsyncMock, coordinator.scale.tare_and_start_timer
        ).side_effect = original_side_effect

        with patch.object(
            logging.getLogger("custom_components.bookoo.coordinator"), "error"
        ) as mock_log_error:
            await coordinator.async_start_shot_service(mock_call)
            mock_log_error.assert_called_once()
            args, _ = mock_log_error.call_args
            assert (
                args[0] == "%s: Error sending Tare & Start Timer command to scale: %s"
            )
            assert args[1] == coordinator.name
            assert args[2] is original_side_effect
        cast(MagicMock, coordinator._reset_realtime_analytics).assert_called_once()
        cast(
            AsyncMock, coordinator.session_manager.start_session
        ).assert_called_once_with(trigger="ha_service")

    @pytest.mark.asyncio
    async def test_async_update_data_connect_disconnect(
        self,
        coordinator: BookooCoordinator,
        mock_scale: MagicMock,
        hass: MagicMock,
        caplog: LogCaptureFixture,
        mocker: MockerFixture,
    ):
        # Scenario 3: BookooError during connection
        caplog.set_level(logging.WARNING)  # Errors are expected
        mocker.patch.object(
            coordinator.scale,
            "connected",
            new_callable=PropertyMock,
            return_value=False,
        )
        mock_ac_bookoo_error = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        bookoo_error_instance = BookooError("Test BookooError")
        mock_ac_bookoo_error.side_effect = bookoo_error_instance
        mock_ac_bookoo_error.reset_mock()
        # Re-assign side_effect as reset_mock clears it.
        mock_ac_bookoo_error.side_effect = bookoo_error_instance

        with pytest.raises(
            UpdateFailed,
            match=f"Error communicating with Bookoo scale: {bookoo_error_instance}",
        ):
            await coordinator._async_update_data()

        mock_ac_bookoo_error.assert_called_once()
        self._assert_log_message(
            caplog,
            "WARNING",
            "custom_components.bookoo.coordinator",
            f"{coordinator.name}: Error communicating with Bookoo scale: {bookoo_error_instance}",
        )

        # Scenario 4: BookooDeviceNotFound during connection
        mocker.patch.object(
            coordinator.scale,
            "connected",
            new_callable=PropertyMock,
            return_value=False,
        )
        mock_ac_not_found = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        mock_ac_not_found.side_effect = BookooDeviceNotFound("Test DeviceNotFound")
        mock_ac_not_found.return_value = None
        mock_ac_not_found.reset_mock()
        with pytest.raises(
            UpdateFailed, match="Bookoo scale device not found: Test DeviceNotFound"
        ):
            await coordinator._async_update_data()
        mock_ac_not_found.assert_called_once()
        self._assert_log_message(
            caplog,
            "WARNING",
            "custom_components.bookoo.coordinator",
            f"{coordinator.name}: Bookoo scale device not found: Test DeviceNotFound",
        )

        # Scenario 5: process_queue fails with BookooError (shot active)
        mocker.patch.object(
            coordinator.scale, "connected", new_callable=PropertyMock, return_value=True
        )  # Scale is connected for this scenario
        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        cast(AsyncMock, coordinator.scale.process_queue).side_effect = BookooError(
            "Test BookooError during queue"
        )
        with pytest.raises(
            UpdateFailed,
            match="Error processing Bookoo scale data: Test BookooError during queue",
        ):
            await coordinator._async_update_data()
        self._assert_log_message(
            caplog,
            "WARNING",
            "custom_components.bookoo.coordinator",
            f"{coordinator.name}: Error processing Bookoo scale data: Test BookooError during queue",
        )
        cast(
            AsyncMock, coordinator.session_manager.stop_session
        ).assert_called_once_with(stop_reason="disconnected_bookoo_error_during_update")
        self._assert_log_message(
            caplog,
            "WARNING",
            "custom_components.bookoo.coordinator",
            f"{coordinator.name}: Scale disconnected/failed during an active shot. Ending session (reason suffix: bookoo_error_during_update).",
        )

        # Scenario 6: process_queue fails with BookooDeviceNotFound (shot active)
        mocker.patch.object(
            coordinator.scale, "connected", new_callable=PropertyMock, return_value=True
        )
        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        cast(
            AsyncMock, coordinator.scale.process_queue
        ).side_effect = BookooDeviceNotFound("Device disappeared")
        with pytest.raises(
            UpdateFailed, match="Bookoo scale device not found: Device disappeared"
        ):
            await coordinator._async_update_data()
        cast(
            AsyncMock, coordinator.session_manager.stop_session
        ).assert_called_once_with(
            stop_reason="disconnected_device_not_found_during_update"
        )
        self._assert_log_message(
            caplog,
            "WARNING",
            "custom_components.bookoo.coordinator",
            f"{coordinator.name}: Scale disconnected/failed during an active shot. Ending session (reason suffix: device_not_found_during_update).",
        )

        # Scenario 7: Reconnect after failure (shot not active)
        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        mocker.patch.object(
            coordinator.scale,
            "connected",
            new_callable=PropertyMock,
            return_value=False,
        )  # Attempting to reconnect
        mock_ac3 = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        mock_ac3.return_value = True
        mock_ac3.reset_mock()
        cast(AsyncMock, coordinator.scale.process_queue).side_effect = None
        await coordinator._async_update_data()
        cast(AsyncMock, coordinator.session_manager.stop_session).assert_not_called()
        mock_ac3.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_update_data_initial_connect_success(
        self,
        coordinator: BookooCoordinator,
        caplog: LogCaptureFixture,
        mocker: MockerFixture,
    ):
        """Test _async_update_data for initial successful connection."""
        caplog.set_level(logging.INFO)

        # Patch 'connected' to be False initially for this test run
        mocker.patch.object(
            coordinator.scale,
            "connected",
            new_callable=PropertyMock,
            return_value=False,
        )

        mock_async_connect = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        mock_async_connect.return_value = True  # Simulate successful connection
        mock_async_connect.reset_mock()

        mock_create_task = mocker.patch.object(
            coordinator.config_entry, "async_create_background_task"
        )

        await coordinator._async_update_data()

        mock_async_connect.assert_called_once()
        # Assert that the 'connected' attribute on the mock scale was set to True by the coordinator
        assert coordinator.scale.connected is True
        self._assert_log_message(
            caplog,
            "INFO",
            "custom_components.bookoo.coordinator",
            f"{coordinator.name}: Successfully connected to scale.",
        )
        mock_create_task.assert_called_once()
        assert coordinator.scale.process_queue_task == mock_create_task.return_value

    @pytest.mark.asyncio
    async def test_async_update_data_connect_failure(
        self,
        coordinator: BookooCoordinator,
        caplog: LogCaptureFixture,
        mocker: MockerFixture,
    ):
        """Test _async_update_data when async_connect returns False (simulating a connection failure)."""
        caplog.set_level(logging.WARNING)

        mocker.patch.object(
            coordinator.scale,
            "connected",
            new_callable=PropertyMock,
            return_value=False,
        )

        mock_async_connect = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        mock_async_connect.return_value = False  # Simulate connection failure
        mock_async_connect.reset_mock()

        # Prevent actual task creation if logic were to somehow reach it
        mocker.patch.object(coordinator.config_entry, "async_create_background_task")

        with pytest.raises(UpdateFailed) as excinfo:
            await coordinator._async_update_data()

        # Check the specific UpdateFailed message propagated from _handle_specific_update_exception
        # which wraps the asyncio.TimeoutError raised by _attempt_bookoo_connection
        expected_error_detail = (
            "Connection attempt failed (async_connect returned False)"
        )
        assert f"Timeout connecting to Bookoo scale: {expected_error_detail}" in str(
            excinfo.value
        )

        mock_async_connect.assert_called_once()
        self._assert_log_message(
            caplog,
            "WARNING",
            "custom_components.bookoo.coordinator",
            f"{coordinator.name}: Timeout connecting to Bookoo scale: {expected_error_detail}",
        )

    @pytest.mark.asyncio
    async def test_async_stop_shot_service(
        self,
        coordinator: BookooCoordinator,
        hass: MagicMock,
        caplog,
        mocker: MockerFixture,  # Added MockerFixture
    ):
        """Test the async_stop_shot_service method."""
        caplog.set_level(logging.INFO)
        mock_call = MagicMock(spec=ServiceCall)

        # Patch coordinator.session_manager.stop_session to be an AsyncMock
        mock_sm_stop_session = mocker.patch.object(
            coordinator.session_manager, "stop_session", new_callable=AsyncMock
        )

        coordinator.session_manager.is_shot_active = True
        await coordinator.async_stop_shot_service(mock_call)

        cast(AsyncMock, coordinator.scale.stop_timer).assert_called_once()
        mock_sm_stop_session.assert_called_once_with(stop_reason="ha_service")

        # Reset mocks for the next part of the test
        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.scale.stop_timer).reset_mock()
        mock_sm_stop_session.reset_mock()
        with patch.object(
            logging.getLogger("custom_components.bookoo.coordinator"), "warning"
        ) as mock_log_warning:
            await coordinator.async_stop_shot_service(mock_call)
            mock_log_warning.assert_called_once_with(
                "%s: Stop shot service called, but no shot is active.", coordinator.name
            )
        cast(AsyncMock, coordinator.scale.stop_timer).assert_not_called()
        mock_sm_stop_session.assert_not_called()  # Assert on our mock

        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.scale.stop_timer).reset_mock()
        mock_sm_stop_session.reset_mock()

        original_side_effect = BookooError("Scale stop command failed")
        cast(AsyncMock, coordinator.scale.stop_timer).side_effect = original_side_effect

        with patch.object(
            logging.getLogger("custom_components.bookoo.coordinator"), "error"
        ) as mock_log_error:
            await coordinator.async_stop_shot_service(mock_call)
            mock_log_error.assert_called_once()
            args, _ = mock_log_error.call_args
            assert args[0] == "%s: Error sending Stop Timer command to scale: %s"
            assert args[1] == coordinator.name
            assert args[2] is original_side_effect
        cast(
            AsyncMock, coordinator.session_manager.stop_session
        ).assert_called_once_with(stop_reason="ha_service_scale_error")

    @pytest.mark.asyncio
    async def test_async_options_update_listener(
        self,
        coordinator: BookooCoordinator,
        hass: HomeAssistant,  # This is the generator type from pytest-homeassistant-custom-component
        mocker: MockerFixture,
    ):
        """Test the options update listener reloads config and triggers reload."""
        # Resolve the hass async_generator if it is one
        actual_hass = hass
        if hasattr(hass, "__anext__"):  # Check if it's an async generator
            actual_hass = await anext(hass)

        # Ensure actual_hass is now a HomeAssistant instance, not a generator
        assert not hasattr(
            actual_hass, "__anext__"
        ), "hass should be resolved to HomeAssistant instance"

        # Mock the config entry reload and BookooConfig.from_config_entry
        actual_hass.config_entries.async_reload = AsyncMock()  # type: ignore[method-assign]
        mock_bookoo_config_from_entry = mocker.patch(
            "custom_components.bookoo.coordinator.BookooConfig.from_config_entry",
            return_value=coordinator.bookoo_config,  # Return existing config for simplicity, or a new one
        )

        # Simulate the options update callback
        await coordinator._options_update_callback(
            actual_hass, coordinator.config_entry
        )

        # Assert that BookooConfig.from_config_entry was called
        mock_bookoo_config_from_entry.assert_called_once_with(coordinator.config_entry)

        # Assert that the config entry reload was triggered
        cast(
            AsyncMock, actual_hass.config_entries.async_reload
        ).assert_called_once_with(coordinator.config_entry.entry_id)
