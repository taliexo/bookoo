"""Tests for the Bookoo coordinator."""

import asyncio
import logging
import pytest
from typing import cast
from pytest_mock import MockerFixture
from _pytest.logging import LogCaptureFixture
from datetime import datetime, timezone
from unittest.mock import (
    AsyncMock,
    MagicMock,
    PropertyMock,
    patch,
)

from freezegun import freeze_time

from homeassistant.core import ServiceCall
from homeassistant.helpers.update_coordinator import UpdateFailed

from custom_components.bookoo.coordinator import BookooCoordinator
from custom_components.bookoo.session_manager import SessionManager

from aiobookoov2.exceptions import BookooError, BookooDeviceNotFound
from aiobookoov2.const import (
    UPDATE_SOURCE_COMMAND_CHAR,
    UPDATE_SOURCE_WEIGHT_CHAR,
)


# Fixtures are in conftest.py


class TestBookooCoordinator:
    """Test cases for BookooCoordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization(
        self,
        coordinator: BookooCoordinator,
        mock_scale: MagicMock,
        mock_hass: MagicMock,
    ):
        """Test coordinator initialization."""
        assert coordinator.scale is mock_scale
        assert coordinator.hass is mock_hass
        assert coordinator.name == f"Bookoo {mock_scale.mac}"
        assert isinstance(coordinator.session_manager, SessionManager)
        assert coordinator.last_shot_data is None
        assert coordinator.realtime_channeling_status == "Undetermined"

    @pytest.mark.asyncio
    async def test_handle_char_update_weight_data_active_shot(
        self,
        coordinator: BookooCoordinator,
        mock_hass: MagicMock,  # Used by SessionManager
        freezer: freeze_time,
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
            MagicMock, coordinator.mock_shot_analyzer.detect_channeling
        ).assert_called_once_with(coordinator.session_manager.session_flow_profile)
        cast(
            MagicMock, coordinator.mock_shot_analyzer.identify_pre_infusion
        ).assert_called_once_with(
            coordinator.session_manager.session_flow_profile,
            coordinator.session_manager.session_scale_timer_profile,
        )
        cast(
            MagicMock, coordinator.mock_shot_analyzer.calculate_extraction_uniformity
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
        cast(MagicMock, coordinator.mock_shot_analyzer.detect_channeling).reset_mock()
        cast(
            MagicMock, coordinator.mock_shot_analyzer.identify_pre_infusion
        ).reset_mock()
        cast(
            MagicMock, coordinator.mock_shot_analyzer.calculate_extraction_uniformity
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
                MagicMock, coordinator.mock_shot_analyzer.detect_channeling
            ).assert_not_called()
            cast(
                MagicMock, coordinator.mock_shot_analyzer.identify_pre_infusion
            ).assert_not_called()
            cast(
                MagicMock,
                coordinator.mock_shot_analyzer.calculate_extraction_uniformity,
            ).assert_not_called()
            mock_update_quality.assert_not_called()

            cast(AsyncMock, coordinator.async_update_listeners).assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_char_update_command_char_auto_timer_start_stop(
        self, coordinator: BookooCoordinator, mock_hass: MagicMock, caplog
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
        self, coordinator: BookooCoordinator, mock_hass: MagicMock, caplog
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
        mock_hass.async_create_task.assert_not_called()  # mock_hass.async_create_task is from the fixture, not directly called here
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
        self, coordinator: BookooCoordinator, mock_hass: MagicMock, caplog
    ):
        """Test the async_start_shot_service method."""
        caplog.set_level(logging.INFO)
        mock_call = MagicMock(spec=ServiceCall)

        coordinator.session_manager.is_shot_active = False
        await coordinator.async_start_shot_service(mock_call)
        cast(MagicMock, coordinator._reset_realtime_analytics).assert_called_once()
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
        mock_hass: MagicMock,
        caplog: LogCaptureFixture,
        mocker: MockerFixture,
    ):
        caplog.set_level(logging.INFO)

        mocker.patch.object(
            coordinator.scale,
            "is_connected",
            new_callable=PropertyMock,
            return_value=False,
        )
        mock_ac1 = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        mock_ac1.return_value = True
        mock_ac1.reset_mock()
        # print(
        #     f"DEBUG test_async_update_data_connect_disconnect: BEFORE CALL: id(coordinator.scale)={id(coordinator.scale)}, coordinator.scale.is_connected={coordinator.scale.is_connected}, async_connect_id={id(coordinator.scale.async_connect)}, async_connect_return_value={cast(AsyncMock, coordinator.scale.async_connect).return_value}"
        # )
        await coordinator._async_update_data()
        mock_ac1.assert_called_once()
        found_log = False
        expected_message = f"{coordinator.name}: Successfully connected to scale."
        for record in caplog.records:
            if (
                record.levelname == "INFO"
                and record.name == "custom_components.bookoo.coordinator"
                and expected_message in record.message
            ):
                found_log = True
                break
        assert found_log, f"Expected log message '{expected_message}' not found in INFO logs from custom_components.bookoo.coordinator"
        caplog.clear()

        mocker.patch.object(
            coordinator.scale,
            "is_connected",
            new_callable=PropertyMock,
            return_value=False,
        )
        mock_ac2 = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        mock_ac2.return_value = False
        mock_ac2.reset_mock()
        with pytest.raises(
            UpdateFailed, match=f"Failed to connect to Bookoo scale {coordinator.name}"
        ):
            await coordinator._async_update_data()
        mock_ac2.assert_called_once()
        found_warning_log = False
        expected_warning_message = f"{coordinator.name}: Failed to connect to scale."
        for record in caplog.records:
            if (
                record.levelname == "WARNING"
                and record.name == "custom_components.bookoo.coordinator"
                and expected_warning_message in record.message
            ):
                found_warning_log = True
                break
        assert (
            found_warning_log
        ), f"Expected warning log '{expected_warning_message}' not found."
        caplog.clear()

        mocker.patch.object(
            coordinator.scale,
            "is_connected",
            new_callable=PropertyMock,
            return_value=True,
        )
        coordinator.session_manager.is_shot_active = True
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
        found_disconnect_log = False
        expected_disconnect_log = f"{coordinator.name}: Scale disconnected/failed during an active shot. Ending session (reason suffix: device_not_found_during_update)."
        for record in caplog.records:
            if (
                record.levelname == "WARNING"
                and record.name == "custom_components.bookoo.coordinator"
                and expected_disconnect_log in record.message
            ):
                found_disconnect_log = True
                break
        assert (
            found_disconnect_log
        ), f"Expected disconnect log '{expected_disconnect_log}' not found."
        caplog.clear()

        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        mocker.patch.object(
            coordinator.scale,
            "is_connected",
            new_callable=PropertyMock,
            return_value=False,
        )
        mock_ac3 = cast(AsyncMock, coordinator.scale.async_connect)  # type: ignore[attr-defined]
        mock_ac3.return_value = True
        mock_ac3.reset_mock()
        cast(AsyncMock, coordinator.scale.process_queue).side_effect = None

        await coordinator._async_update_data()
        cast(AsyncMock, coordinator.session_manager.stop_session).assert_not_called()
        mock_ac3.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_stop_shot_service(
        self, coordinator: BookooCoordinator, mock_hass: MagicMock, caplog
    ):
        """Test the async_stop_shot_service method."""
        caplog.set_level(logging.INFO)
        mock_call = MagicMock(spec=ServiceCall)

        coordinator.session_manager.is_shot_active = True
        await coordinator.async_stop_shot_service(mock_call)
        cast(AsyncMock, coordinator.scale.stop_timer).assert_called_once()
        cast(
            AsyncMock, coordinator.session_manager.stop_session
        ).assert_called_once_with(stop_reason="ha_service")

        coordinator.session_manager.is_shot_active = False
        cast(AsyncMock, coordinator.scale.stop_timer).reset_mock()
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()
        with patch.object(
            logging.getLogger("custom_components.bookoo.coordinator"), "warning"
        ) as mock_log_warning:
            await coordinator.async_stop_shot_service(mock_call)
            mock_log_warning.assert_called_once_with(
                "%s: Stop shot service called, but no shot is active.", coordinator.name
            )
        cast(AsyncMock, coordinator.scale.stop_timer).assert_not_called()
        cast(AsyncMock, coordinator.session_manager.stop_session).assert_not_called()

        coordinator.session_manager.is_shot_active = True
        cast(AsyncMock, coordinator.scale.stop_timer).reset_mock()
        cast(AsyncMock, coordinator.session_manager.stop_session).reset_mock()

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
        self, coordinator: BookooCoordinator, mock_hass: MagicMock
    ):
        """Test the options update listener reloads config and triggers reload."""
        mock_hass.config_entries.async_reload = AsyncMock()

        await coordinator._options_update_callback(mock_hass, coordinator.config_entry)

        cast(AsyncMock, mock_hass.config_entries.async_reload).assert_called_once_with(
            coordinator.config_entry.entry_id
        )

    @pytest.mark.asyncio
    async def test_reset_realtime_analytics(self, coordinator: BookooCoordinator):
        """Test the _reset_realtime_analytics method."""
        coordinator.realtime_channeling_status = "Mild Channeling"
        coordinator.realtime_pre_infusion_active = True
        coordinator.realtime_pre_infusion_duration = 5.0
        coordinator.realtime_extraction_uniformity = 0.75
        coordinator.realtime_shot_quality_score = 75.0

        coordinator._reset_realtime_analytics()

        assert coordinator.realtime_channeling_status == "Undetermined"
        assert not coordinator.realtime_pre_infusion_active
        assert coordinator.realtime_pre_infusion_duration is None
        assert coordinator.realtime_extraction_uniformity == 0.0
        assert coordinator.realtime_shot_quality_score == 0.0
