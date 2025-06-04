"""Tests for the SessionManager class in Bookoo integration."""

import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from aiobookoov2.bookooscale import BookooScale  # For type hinting mock
from homeassistant.config_entries import ConfigEntry  # For type hinting mock
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from pydantic import ValidationError

from custom_components.bookoo.const import (
    DEFAULT_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
    DEFAULT_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
    DEFAULT_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
    DEFAULT_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
    DEFAULT_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
    DEFAULT_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
    EVENT_BOOKOO_SHOT_COMPLETED,
    BookooConfig,
)
from custom_components.bookoo.coordinator import (
    BookooCoordinator,
)

# For type hinting mock
from custom_components.bookoo.session_manager import SessionManager
from custom_components.bookoo.types import (
    BookooShotCompletedEventDataModel,
    FlowDataPoint,
    ScaleTimerDataPoint,
    WeightDataPoint,
)

# Configure basic logging for tests if needed
# logging.basicConfig(level=logging.DEBUG)
_LOGGER = logging.getLogger(__name__)


@pytest.fixture
def mock_hass_session_manager():
    """Fixture for a mock HomeAssistant tailored for SessionManager tests."""
    hass = MagicMock()  # Removed spec=ActualHomeAssistant
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    hass.states = MagicMock()
    hass.states.get = MagicMock(return_value=None)
    hass.async_create_task = MagicMock(
        side_effect=lambda coro: asyncio.create_task(coro)
    )
    hass.async_add_executor_job = AsyncMock()  # For async_add_shot_record if it uses it
    return hass


@pytest.fixture
def mock_scale_session_manager():
    """Fixture for a mock BookooScale tailored for SessionManager tests."""
    scale = MagicMock(spec=BookooScale)
    scale.weight = 0.0
    return scale


@pytest.fixture
def mock_config_entry_session_manager():
    """Fixture for a mock ConfigEntry tailored for SessionManager tests."""
    entry = MagicMock(spec=ConfigEntry)
    entry.entry_id = "session_test_entry_id"
    entry.unique_id = "session_test_unique_id"
    entry.options = {}  # Default, can be overridden in tests
    return entry


@pytest.fixture
def bookoo_config_default(mock_config_entry_session_manager: ConfigEntry):
    """Fixture for a default BookooConfig instance."""
    # Ensure mock_config_entry_session_manager.options is populated if BookooConfig.from_config_entry expects it
    # For a truly default config, we can instantiate BookooConfig directly with its defaults
    return BookooConfig(
        min_shot_duration=10,
        linked_bean_weight_entity=None,
        linked_coffee_name_entity=None,
        enable_auto_stop_flow_cutoff=False,
        auto_stop_pre_infusion_ignore_duration=DEFAULT_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
        auto_stop_min_flow_for_stability=DEFAULT_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
        auto_stop_max_flow_variance_for_stability=DEFAULT_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
        auto_stop_min_duration_for_stability=DEFAULT_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
        auto_stop_flow_cutoff_threshold=DEFAULT_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
        auto_stop_min_duration_for_cutoff=DEFAULT_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
    )


@pytest.fixture
def mock_coordinator_session_manager(
    mock_hass_session_manager,
    mock_config_entry_session_manager,
    bookoo_config_default,
    mock_scale_session_manager,
):
    """Fixture for a mock BookooCoordinator tailored for SessionManager tests."""
    coordinator = MagicMock(spec=BookooCoordinator)
    coordinator.hass = mock_hass_session_manager
    coordinator.config_entry = mock_config_entry_session_manager
    coordinator.bookoo_config = bookoo_config_default
    coordinator.scale = mock_scale_session_manager
    coordinator.name = "TestBookooSMDevice"
    coordinator.async_update_listeners = MagicMock()  # Real method is not async
    # Initialize real-time analytics attributes that SessionManager might read/rely on
    coordinator.realtime_channeling_status = "Undetermined"
    coordinator.realtime_pre_infusion_active = False
    coordinator.realtime_pre_infusion_duration = None
    coordinator.realtime_extraction_uniformity = 0.0
    coordinator.realtime_shot_quality_score = 0.0
    # Mock last_shot_data attribute for SessionManager to update
    coordinator.last_shot_data = None
    return coordinator


@pytest_asyncio.fixture
async def session_manager(mock_hass_session_manager, mock_coordinator_session_manager):
    """Fixture for a SessionManager instance."""
    # Patch dt_util.utcnow for consistent time in tests
    with patch("custom_components.bookoo.session_manager.dt_util") as mock_dt_util_sm:
        mock_dt_util_sm.utcnow = MagicMock(
            return_value=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        )
        sm = SessionManager(mock_hass_session_manager, mock_coordinator_session_manager)
        yield sm


class TestSessionManager:
    """Test cases for SessionManager."""

    def test_initialization(
        self,
        session_manager: SessionManager,
        mock_hass_session_manager,
        mock_coordinator_session_manager,
    ):
        """Test SessionManager initializes correctly."""
        assert session_manager.hass is mock_hass_session_manager
        assert session_manager.coordinator is mock_coordinator_session_manager
        assert not session_manager.is_shot_active
        assert session_manager.session_start_time_utc is None
        assert session_manager.session_flow_profile == []
        assert session_manager.session_weight_profile == []
        assert session_manager.session_scale_timer_profile == []
        assert session_manager.session_input_parameters == {}
        assert session_manager.session_start_trigger is None
        assert session_manager.last_shot_data is None

    @pytest.mark.asyncio
    async def test_start_session_basic(
        self, session_manager: SessionManager, mock_coordinator_session_manager
    ):
        """Test basic session start."""
        test_trigger = "test_manual_start"

        # Patch dt_util.utcnow for this specific test's timing if needed, or rely on fixture patch
        # For this test, the fixture's patch of dt_util.utcnow is likely sufficient.

        await session_manager.start_session(trigger=test_trigger)

        assert session_manager.is_shot_active
        assert session_manager.session_start_time_utc is not None
        assert session_manager.session_start_time_utc == datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc
        )
        assert session_manager.session_start_trigger == test_trigger
        assert (
            session_manager.session_input_parameters == {}
        )  # No linked entities mocked yet
        mock_coordinator_session_manager.async_update_listeners.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_session_with_linked_entities(
        self,
        session_manager: SessionManager,
        mock_coordinator_session_manager,
        mock_hass_session_manager,
    ):
        """Test session start populates input_parameters from linked entities."""
        # Configure coordinator for this test scenario
        mock_coordinator_session_manager.bookoo_config = replace(
            mock_coordinator_session_manager.bookoo_config,
            linked_bean_weight_entity="input_number.test_bean_weight",
            linked_coffee_name_entity="input_text.test_coffee_name",
        )

        # Mock hass.states.get to return specific states
        def mock_states_get(entity_id):
            if entity_id == "input_number.test_bean_weight":
                mock_state = MagicMock()
                mock_state.state = "18.5"
                return mock_state
            if entity_id == "input_text.test_coffee_name":
                mock_state = MagicMock()
                mock_state.state = "Ethiopian Yirgacheffe"
                return mock_state
            return None

        mock_hass_session_manager.states.get = MagicMock(side_effect=mock_states_get)

        await session_manager.start_session(trigger="linked_entity_test")

        assert session_manager.is_shot_active
        assert session_manager.session_input_parameters == {
            "bean_weight": "18.5",
            "coffee_name": "Ethiopian Yirgacheffe",
        }
        mock_coordinator_session_manager.async_update_listeners.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_session_with_unavailable_linked_entities(
        self,
        session_manager: SessionManager,
        mock_coordinator_session_manager,
        mock_hass_session_manager,
    ):
        """Test session start handles unavailable/unknown linked entity states gracefully."""
        mock_coordinator_session_manager.bookoo_config = replace(
            mock_coordinator_session_manager.bookoo_config,
            linked_bean_weight_entity="input_number.unavailable_bean_weight",
            linked_coffee_name_entity="input_text.unknown_coffee_name",
        )

        def mock_states_get_unavailable(entity_id):
            if entity_id == "input_number.unavailable_bean_weight":
                mock_state = MagicMock()
                mock_state.state = STATE_UNAVAILABLE  # Home Assistant constant
                return mock_state
            if entity_id == "input_text.unknown_coffee_name":
                mock_state = MagicMock()
                mock_state.state = STATE_UNKNOWN  # Home Assistant constant
                return mock_state
            return None

        mock_hass_session_manager.states.get = MagicMock(
            side_effect=mock_states_get_unavailable
        )

        await session_manager.start_session(trigger="unavailable_entity_test")

        assert session_manager.is_shot_active
        assert (
            session_manager.session_input_parameters == {}
        )  # Should not populate if state is unknown/unavailable
        mock_coordinator_session_manager.async_update_listeners.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_flow_data(self, session_manager: SessionManager):
        """Test add_flow_data method."""
        # Test when session is not active
        session_manager.add_flow_data(elapsed_time=1.0, flow_rate=0.5)
        assert len(session_manager.session_flow_profile) == 0

        # Test when session is active
        await session_manager.start_session(trigger="test_flow_data")
        session_manager.add_flow_data(elapsed_time=1.1, flow_rate=0.6)
        assert len(session_manager.session_flow_profile) == 1
        assert isinstance(session_manager.session_flow_profile[0], FlowDataPoint)
        assert session_manager.session_flow_profile[0].elapsed_time == 1.1
        assert session_manager.session_flow_profile[0].flow_rate == 0.6

        session_manager.add_flow_data(elapsed_time=2.2, flow_rate=0.7)
        assert len(session_manager.session_flow_profile) == 2
        assert session_manager.session_flow_profile[1].elapsed_time == 2.2
        assert session_manager.session_flow_profile[1].flow_rate == 0.7

    @pytest.mark.asyncio
    async def test_add_weight_data(self, session_manager: SessionManager):
        """Test add_weight_data method."""
        # Test when session is not active
        session_manager.add_weight_data(elapsed_time=1.0, weight=10.0)
        assert len(session_manager.session_weight_profile) == 0

        # Test when session is active
        await session_manager.start_session(trigger="test_weight_data")
        session_manager.add_weight_data(elapsed_time=1.1, weight=10.1)
        assert len(session_manager.session_weight_profile) == 1
        assert isinstance(session_manager.session_weight_profile[0], WeightDataPoint)
        assert session_manager.session_weight_profile[0].elapsed_time == 1.1
        assert session_manager.session_weight_profile[0].weight == 10.1

        session_manager.add_weight_data(elapsed_time=2.2, weight=20.2)
        assert len(session_manager.session_weight_profile) == 2
        assert session_manager.session_weight_profile[1].elapsed_time == 2.2
        assert session_manager.session_weight_profile[1].weight == 20.2

    @pytest.mark.asyncio
    async def test_add_scale_timer_data(self, session_manager: SessionManager):
        """Test add_scale_timer_data method."""
        # Test when session is not active
        session_manager.add_scale_timer_data(elapsed_time=1.0, timer_value=1)
        assert len(session_manager.session_scale_timer_profile) == 0

        # Test when session is active
        await session_manager.start_session(trigger="test_timer_data")
        session_manager.add_scale_timer_data(elapsed_time=1.1, timer_value=2)
        assert len(session_manager.session_scale_timer_profile) == 1
        assert isinstance(
            session_manager.session_scale_timer_profile[0], ScaleTimerDataPoint
        )
        assert session_manager.session_scale_timer_profile[0].elapsed_time == 1.1
        assert session_manager.session_scale_timer_profile[0].timer_value == 2

        session_manager.add_scale_timer_data(elapsed_time=2.2, timer_value=3)
        assert len(session_manager.session_scale_timer_profile) == 2
        assert session_manager.session_scale_timer_profile[1].elapsed_time == 2.2
        assert session_manager.session_scale_timer_profile[1].timer_value == 3

    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.session_manager.async_add_shot_record",
        new_callable=AsyncMock,
    )
    async def test_stop_session_basic_completion(
        self,
        mock_async_add_shot_record: AsyncMock,
        session_manager: SessionManager,
        mock_hass_session_manager,
        mock_coordinator_session_manager,
    ):
        """Test stop_session for a basic successful shot completion."""
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)  # 30s duration

        # Setup coordinator and scale states
        mock_coordinator_session_manager.scale.weight = 36.0  # Final weight
        mock_coordinator_session_manager.realtime_channeling_status = "None"
        mock_coordinator_session_manager.realtime_pre_infusion_active = False
        mock_coordinator_session_manager.realtime_pre_infusion_duration = None
        mock_coordinator_session_manager.realtime_extraction_uniformity = 0.9
        mock_coordinator_session_manager.realtime_shot_quality_score = 90.0
        # Ensure _reset_realtime_analytics is a mock we can assert on
        mock_coordinator_session_manager._reset_realtime_analytics = MagicMock()

        with patch(
            "custom_components.bookoo.session_manager.dt_util.utcnow",
            side_effect=[start_time, end_time],
        ):
            await session_manager.start_session(trigger="test_stop_basic")

            # Add some profile data
            session_manager.add_flow_data(elapsed_time=5.0, flow_rate=0.5)
            session_manager.add_flow_data(elapsed_time=10.0, flow_rate=1.5)  # Peak flow
            session_manager.add_flow_data(elapsed_time=15.0, flow_rate=1.2)
            session_manager.add_flow_data(elapsed_time=20.0, flow_rate=1.0)
            session_manager.add_flow_data(elapsed_time=25.0, flow_rate=0.8)

            session_manager.add_weight_data(
                elapsed_time=25.0, weight=30.0
            )  # Not final, final is from scale.weight
            session_manager.add_scale_timer_data(elapsed_time=25.0, timer_value=25)

            await session_manager.stop_session(stop_reason="test_completed_normally")

        assert not session_manager.is_shot_active
        assert session_manager.last_shot_data is not None
        assert isinstance(
            session_manager.last_shot_data, BookooShotCompletedEventDataModel
        )

        # Verify key fields in the validated data
        event_payload = session_manager.last_shot_data
        assert event_payload.status == "completed"
        assert event_payload.stop_reason == "test_completed_normally"
        assert event_payload.duration_seconds == 30.0
        assert event_payload.final_weight_grams == 36.0
        assert event_payload.average_flow_rate_gps == round(36.0 / 30.0, 2)  # 1.2
        assert event_payload.peak_flow_rate_gps == 1.5
        assert (
            event_payload.time_to_first_flow_seconds == 5.0
        )  # Assuming 0.5 > FIRST_FLOW_THRESHOLD_GPS (0.2)
        assert event_payload.time_to_peak_flow_seconds == 10.0
        assert len(event_payload.flow_profile) == 5
        assert event_payload.channeling_status == "None"
        assert event_payload.shot_quality_score == 90.0

        # Check that coordinator's copy is also updated
        assert mock_coordinator_session_manager.last_shot_data == event_payload

        # Verify event firing
        print(
            f"DEBUG test_stop_session_basic_completion: type(event_payload)={type(event_payload)}, dir(event_payload)={dir(event_payload)}"
        )
        mock_hass_session_manager.bus.async_fire.assert_called_once_with(
            EVENT_BOOKOO_SHOT_COMPLETED, event_payload.model_dump()
        )

        # Verify storage call
        mock_async_add_shot_record.assert_called_once_with(
            mock_hass_session_manager, event_payload.model_dump()
        )

        # Verify state reset calls
        mock_coordinator_session_manager._reset_realtime_analytics.assert_called_once()
        mock_coordinator_session_manager.async_update_listeners.assert_called()
        # Check if internal profiles are reset (part of _reset_session_variables -> _reset_internal_session_state)
        assert session_manager.session_flow_profile == []
        assert session_manager.session_start_time_utc is None

    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.session_manager.async_add_shot_record",
        new_callable=AsyncMock,
    )
    async def test_stop_session_aborted_too_short(
        self,
        mock_async_add_shot_record: AsyncMock,
        session_manager: SessionManager,
        mock_hass_session_manager,
        mock_coordinator_session_manager,
    ):
        """Test stop_session when shot duration is less than min_shot_duration."""
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Shot duration 5s, default min_shot_duration in BookooConfig is 10s
        end_time = datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc)

        mock_coordinator_session_manager.bookoo_config = replace(
            mock_coordinator_session_manager.bookoo_config, min_shot_duration=20
        )
        mock_coordinator_session_manager.scale.weight = 5.0  # Some final weight
        mock_coordinator_session_manager._reset_realtime_analytics = MagicMock()

        with patch(
            "custom_components.bookoo.session_manager.dt_util.utcnow",
            side_effect=[start_time, end_time],
        ):
            await session_manager.start_session(trigger="test_aborted_short")
            # Add minimal data, as it's a short shot
            session_manager.add_flow_data(elapsed_time=1.0, flow_rate=0.1)
            await session_manager.stop_session(stop_reason="test_too_short")

        assert not session_manager.is_shot_active
        assert session_manager.last_shot_data is not None
        event_payload = session_manager.last_shot_data

        assert event_payload.status == "aborted_too_short"
        assert (
            event_payload.stop_reason == "test_too_short"
        )  # Original stop reason is preserved
        assert event_payload.duration_seconds == 5.0
        assert (
            event_payload.final_weight_grams == 0.0
        )  # For aborted_too_short, specific minimal data is set
        assert len(event_payload.flow_profile) == 0  # Minimal data for aborted

        mock_hass_session_manager.bus.async_fire.assert_not_called()  # No main event for aborted_too_short
        mock_async_add_shot_record.assert_called_once()  # Should still attempt to save minimal record
        # Check that the saved record matches the minimal structure for aborted_too_short
        saved_data_dict = mock_async_add_shot_record.call_args[0][1]
        assert saved_data_dict["status"] == "aborted_too_short"
        assert saved_data_dict["final_weight_grams"] == 0.0

        mock_coordinator_session_manager._reset_realtime_analytics.assert_called_once()
        mock_coordinator_session_manager.async_update_listeners.assert_called()
        assert session_manager.session_flow_profile == []

    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.session_manager.async_add_shot_record",
        new_callable=AsyncMock,
    )
    async def test_stop_session_disconnected(
        self,
        mock_async_add_shot_record: AsyncMock,
        session_manager: SessionManager,
        mock_hass_session_manager,
        mock_coordinator_session_manager,
    ):
        """Test stop_session when stop_reason is 'disconnected'."""
        start_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        # Disconnected after 15s, which is > min_shot_duration (default 10s)
        end_time = datetime(2024, 1, 1, 12, 0, 15, tzinfo=timezone.utc)

        mock_coordinator_session_manager.scale.weight = (
            15.0  # Some weight at time of disconnect
        )
        mock_coordinator_session_manager._reset_realtime_analytics = MagicMock()

        with patch(
            "custom_components.bookoo.session_manager.dt_util.utcnow",
            side_effect=[start_time, end_time],
        ):
            await session_manager.start_session(trigger="test_disconnected")
            session_manager.add_flow_data(elapsed_time=5.0, flow_rate=1.0)
            session_manager.add_weight_data(elapsed_time=10.0, weight=10.0)
            await session_manager.stop_session(stop_reason="disconnected")

        assert not session_manager.is_shot_active
        assert session_manager.last_shot_data is not None
        event_payload = session_manager.last_shot_data

        assert event_payload.status == "aborted_disconnected"
        assert event_payload.stop_reason == "disconnected"
        assert event_payload.duration_seconds == 15.0
        assert event_payload.final_weight_grams == 15.0  # Weight at time of disconnect
        assert (
            len(event_payload.flow_profile) == 1
        )  # Profile data up to disconnect is preserved

        # Event should still fire for disconnected, as it's a form of completion
        print(
            f"DEBUG test_stop_session_disconnected: type(event_payload)={type(event_payload)}, dir(event_payload)={dir(event_payload)}"
        )
        mock_hass_session_manager.bus.async_fire.assert_called_once_with(
            EVENT_BOOKOO_SHOT_COMPLETED, event_payload.model_dump()
        )
        mock_async_add_shot_record.assert_called_once()
        saved_data_dict = mock_async_add_shot_record.call_args[0][1]
        assert saved_data_dict["status"] == "aborted_disconnected"

        mock_coordinator_session_manager._reset_realtime_analytics.assert_called_once()
        mock_coordinator_session_manager.async_update_listeners.assert_called()

    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.session_manager.async_add_shot_record",
        new_callable=AsyncMock,
    )
    @patch(
        "custom_components.bookoo.session_manager._LOGGER.error"
    )  # Mock logger.error
    async def test_stop_session_pydantic_validation_failure(
        self,
        mock_logger_error: MagicMock,
        mock_async_add_shot_record: AsyncMock,
        session_manager: SessionManager,
        mock_hass_session_manager,
        mock_coordinator_session_manager,
    ):
        """Test stop_session when Pydantic validation of event data fails."""
        mock_coordinator_session_manager.scale.weight = 25.0
        mock_coordinator_session_manager._reset_realtime_analytics = MagicMock()

        # Intentionally create malformed data by making a required field an incorrect type
        # We'll achieve this by patching the dictionary creation part inside stop_session,
        # or by ensuring our mock data passed to the Pydantic model is bad.
        # For this test, we'll let stop_session build its dict, then intercept and modify it
        # before Pydantic validation, or more simply, ensure one of the inputs is bad.
        # The easiest way is to ensure a required field is missing when it's constructed.
        # Let's simulate 'final_weight_grams' being missing by making scale.weight None
        # and ensuring the Pydantic model requires it (which it does, it's not Optional).
        # However, the current code defaults final_weight_grams if scale.weight is None.
        # A more direct way to test validation failure is to make a field the wrong type.

        # Let's assume 'duration_seconds' is made a string by mistake in raw_event_data construction phase
        # This requires a bit more intricate patching if we want to simulate it perfectly *inside* the method.
        # A simpler approach for testing the except block:
        #   Patch BookooShotCompletedEventDataModel to raise ValidationError.

        # Define a helper function for the side_effect
        def raise_validation_error(*args, **kwargs):
            error_details = [
                {
                    "type": "float_type",  # Pydantic v2 error type for float parsing
                    "loc": ("duration_seconds",),
                    "msg": "Input should be a valid float",
                    "input": kwargs.get(
                        "duration_seconds", "N/A"
                    ),  # Example of capturing input if needed
                }
            ]
            # Use from_exception_data for constructing ValidationError
            raise ValidationError.from_exception_data(
                title=BookooShotCompletedEventDataModel, line_errors=error_details
            )

        with (
            patch(
                "custom_components.bookoo.session_manager.async_add_shot_record",
                new_callable=AsyncMock,
            ),
            patch(
                "custom_components.bookoo.session_manager.BookooShotCompletedEventDataModel",
                side_effect=raise_validation_error,  # Use the helper function
            ),
        ):
            await session_manager.start_session(trigger="test_validation_fail")
            await session_manager.stop_session(stop_reason="test_validation_fail_stop")

        assert not session_manager.is_shot_active  # Session should still be reset
        assert (
            session_manager.last_shot_data is None
        )  # Should not be set on validation error
        assert (
            mock_coordinator_session_manager.last_shot_data is None
        )  # Should not be set on validation error

        mock_logger_error.assert_called_once()
        # Check that the log message contains relevant info about validation failure
        args, _ = mock_logger_error.call_args
        # args[0] is the format string, args[1] is self.name, args[2] is the exception e
        assert "Shot data validation failed" in args[0]
        assert (
            args[1] == session_manager.coordinator.name
        )  # Check that the name is logged correctly
        assert isinstance(args[2], ValidationError)  # %s for exception

        mock_hass_session_manager.bus.async_fire.assert_not_called()
        mock_async_add_shot_record.assert_not_called()

        # State reset calls should still happen to clean up
        # _reset_realtime_analytics is called by _reset_session_variables in session_manager
        mock_coordinator_session_manager._reset_realtime_analytics.assert_called_once()
        mock_coordinator_session_manager.async_update_listeners.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_session_no_active_session(
        self,
        session_manager: SessionManager,
        mock_hass_session_manager: MagicMock,
        mock_coordinator_session_manager: MagicMock,
        caplog,
    ):
        """Test stop_session when no session is active."""
        # Ensure no session is active initially
        session_manager.is_shot_active = False
        session_manager.session_start_time_utc = None
        # Also ensure coordinator's last_shot_data is None initially for this test context
        mock_coordinator_session_manager.last_shot_data = None

        with patch(
            "custom_components.bookoo.session_manager.async_add_shot_record",
            new_callable=AsyncMock,
        ) as mock_add_record:
            await session_manager.stop_session(stop_reason="test_no_active")

            # Assertions: no state change, no event, no storage, specific log
            assert not session_manager.is_shot_active
            assert session_manager.last_shot_data is None
            assert (
                mock_coordinator_session_manager.last_shot_data is None
            )  # Should remain None

            # Check logs via caplog
            found_log = False
            for record in caplog.records:
                if (
                    record.name == "custom_components.bookoo.session_manager"
                    and record.levelno == logging.DEBUG
                    and "Stop session called but no active session or start time found."
                    in record.getMessage()
                ):
                    found_log = True
                    break
            assert (
                found_log
            ), "Expected log message about no active session or start time not found"

            mock_hass_session_manager.bus.async_fire.assert_not_called()
            mock_add_record.assert_not_called()
            mock_coordinator_session_manager._reset_realtime_analytics.assert_not_called()
            mock_coordinator_session_manager.async_update_listeners.assert_not_called()

    # Final test for SessionManager: _reset_internal_session_state
    def test_reset_internal_session_state(self, session_manager: SessionManager):
        """Test the _reset_internal_session_state method."""
        # Set up some state
        session_manager.is_shot_active = True
        session_manager.session_start_time_utc = datetime.now(timezone.utc)
        session_manager.session_flow_profile = [FlowDataPoint(1.0, 0.5)]
        session_manager.session_weight_profile = [WeightDataPoint(1.0, 10.0)]
        session_manager.session_scale_timer_profile = [ScaleTimerDataPoint(1.0, 1)]
        session_manager.session_input_parameters = {"key": "value"}
        session_manager.session_start_trigger = "test_trigger"
        # last_shot_data is not reset by this internal method, but by _reset_session_variables

        session_manager._reset_internal_session_state()

        assert (
            not session_manager.is_shot_active
        )  # is_shot_active is reset by _reset_session_variables
        assert session_manager.session_start_time_utc is None
        assert session_manager.session_flow_profile == []
        assert session_manager.session_weight_profile == []
        assert session_manager.session_scale_timer_profile == []
        assert session_manager.session_input_parameters == {}
        assert session_manager.session_start_trigger is None
