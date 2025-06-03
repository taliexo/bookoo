"""Tests for the Bookoo coordinator."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import logging
import pytest_asyncio
from freezegun import freeze_time
from datetime import datetime, timezone

from custom_components.bookoo.const import (
    EVENT_BOOKOO_SHOT_COMPLETED,
    OPTION_MIN_SHOT_DURATION,
    OPTION_LINKED_BEAN_WEIGHT_ENTITY,
    OPTION_LINKED_COFFEE_NAME_ENTITY,
    # DOMAIN, # Will add if ruff later says it's needed and missing
)
from aiobookoov2.const import UPDATE_SOURCE_COMMAND_CHAR, UPDATE_SOURCE_WEIGHT_CHAR
from custom_components.bookoo.coordinator import BookooCoordinator
from homeassistant.const import STATE_UNAVAILABLE, CONF_ADDRESS, CONF_IS_VALID_SCALE


# Mock homeassistant.util.dt if not using freezegun for all time aspects
# For example:
# MOCK_UTC_NOW = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_scale():
    """Fixture for a mock BookooScale."""
    scale = MagicMock()
    scale.weight = 0.0
    scale.flow_rate = 0.0
    scale.timer = 0.0
    scale.device_state = MagicMock()
    scale.device_state.battery_level = 100
    scale.async_send_command = AsyncMock()
    scale.tare_and_start_timer = AsyncMock()
    scale.stop_timer = AsyncMock()
    scale.mac = "test_mac_address"
    scale.address = scale.mac  # Assuming BookooScale stores address this way
    scale.async_connect = AsyncMock(
        return_value=True
    )  # Assume connected for most tests
    return scale


@pytest.fixture
def mock_config_entry():
    """Fixture for a mock ConfigEntry."""
    entry = MagicMock()  # Removed spec_set=True
    entry.entry_id = "test_entry_id"
    entry.unique_id = "test_unique_id"
    entry.options = {}
    entry.runtime_data = None
    entry.data = {
        CONF_ADDRESS: "test_address",
        CONF_IS_VALID_SCALE: True,
    }
    entry.title = "Test Bookoo Scale"
    # If the actual ConfigEntry class has specific methods that are called,
    # those might need to be mocked here too, e.g., entry.add_update_listener = MagicMock()
    return entry


@pytest.fixture
def mock_hass():
    """Fixture for a mock HomeAssistant."""
    hass = MagicMock()  # Removed spec_set=True from the top-level mock for now

    # Define bus and then its methods/attributes
    hass.bus = MagicMock()  # Can add spec_set=True here if needed for hass.bus itself
    hass.bus.async_fire = MagicMock()

    # Define states and then its methods/attributes
    hass.states = (
        MagicMock()
    )  # Can add spec_set=True here if needed for hass.states itself
    hass.states.get = MagicMock(return_value=None)

    async def run_coro_if_awaitable(coro):
        if hasattr(coro, "__await__"):
            return await coro
        return coro

    hass.async_create_task = MagicMock(
        side_effect=lambda coro: asyncio.create_task(coro)
    )
    hass.loop = MagicMock()  # Use a mock loop
    return hass


@pytest_asyncio.fixture
async def coordinator(mock_hass, mock_config_entry, mock_scale):
    """Fixture for a BookooCoordinator instance."""
    with (
        patch(
            "custom_components.bookoo.coordinator.BookooScale", return_value=mock_scale
        ) as _mock_bookoo_scale_init,
        patch(
            "custom_components.bookoo.coordinator.dt_util"
        ) as mock_dt_util_in_fixture,
    ):  # Patch dt_util
        # Make the mocked utcnow() respect freezegun
        mock_dt_util_in_fixture.utcnow = lambda: datetime.now(timezone.utc)
        coord = BookooCoordinator(mock_hass, mock_config_entry)
        # In the actual HA setup, the coordinator is stored in entry.runtime_data
        # For testing, we can assign it if the __init__ doesn't do it,
        # but BookooCoordinator's __init__ does not store itself on entry.runtime_data.
        # The HA framework does that after calling async_setup_entry.
        # For unit testing the coordinator directly, we just need the instance.

        # Ensure the coordinator's scale is our mock_scale.
        # The patch for BookooScale constructor should ensure this.
        # coord.scale is assigned in BookooCoordinator.__init__

        yield coord


class TestBookooCoordinator:
    """Test cases for BookooCoordinator."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization(
        self, coordinator: BookooCoordinator, mock_scale, mock_config_entry, mock_hass
    ):
        """Test coordinator initializes correctly."""
        assert coordinator.hass is mock_hass
        assert coordinator.config_entry == mock_config_entry
        assert coordinator.scale == mock_scale
        assert not coordinator.is_shot_active
        assert coordinator.session_start_time_utc is None
        assert coordinator.session_flow_profile == []
        assert coordinator.session_scale_timer_profile == []
        assert coordinator.session_input_parameters == {}
        assert coordinator.session_start_trigger is None
        assert coordinator.last_shot_data == {}
        assert coordinator.name == f"Bookoo {mock_scale.mac}"
        # Test that options are loaded (default values initially)
        assert (
            coordinator.min_shot_duration == 10
        )  # Default from __init__ if not in options
        assert coordinator.linked_bean_weight_entity_id is None
        assert coordinator.linked_coffee_name_entity_id is None

    @pytest.mark.asyncio
    async def test_handle_char_update_auto_start(
        self, coordinator: BookooCoordinator, mock_hass
    ):
        """Test auto-start from command characteristic."""
        # The coordinator's _handle_characteristic_update does:
        # self.hass.async_create_task(self._start_session(trigger="scale_auto"))
        # The mock_hass.async_create_task fixture is set up to execute the coroutine.

        with patch.object(
            coordinator, "_start_session", new_callable=AsyncMock
        ) as mock_start_session_method:
            # Scenario 1: No active shot, auto-start message received
            coordinator.is_shot_active = False
            auto_start_payload = bytes.fromhex(
                "030d01000000000000000000000000000000000f"
            )

            coordinator._handle_characteristic_update(
                UPDATE_SOURCE_COMMAND_CHAR, auto_start_payload
            )
            await asyncio.sleep(0)  # Allow event loop to run for any scheduled tasks

            mock_start_session_method.assert_not_called()  # Coordinator doesn't decode these raw command bytes to start session

            # Scenario 2: Shot already active, auto-start message received
            mock_start_session_method.reset_mock()
            coordinator.is_shot_active = True

            coordinator._handle_characteristic_update(
                UPDATE_SOURCE_COMMAND_CHAR, auto_start_payload
            )
            await asyncio.sleep(0)

            mock_start_session_method.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_session_populates_linked_inputs(
        self, coordinator: BookooCoordinator, mock_hass, mock_config_entry
    ):
        """Test the _start_session method, including reading linked inputs."""
        # Setup options for linked entities
        mock_config_entry.options = {
            OPTION_LINKED_BEAN_WEIGHT_ENTITY: "input_number.test_bean_weight",
            OPTION_LINKED_COFFEE_NAME_ENTITY: "input_text.test_coffee_name",
            OPTION_MIN_SHOT_DURATION: 7,  # Example min duration
        }
        coordinator._load_options()  # Reload options as coordinator is already initialized

        # Mock states for linked entities
        mock_bean_weight_state = MagicMock()
        mock_bean_weight_state.state = "18.5"
        mock_coffee_name_state = MagicMock()
        mock_coffee_name_state.state = "Ethiopia Yirgacheffe"

        def mock_states_get_side_effect(entity_id):
            if entity_id == "input_number.test_bean_weight":
                return mock_bean_weight_state
            if entity_id == "input_text.test_coffee_name":
                return mock_coffee_name_state
            return None

        mock_hass.states.get.side_effect = mock_states_get_side_effect

        await coordinator._start_session(trigger="test_inputs")

        assert coordinator.session_input_parameters.get("bean_weight") == "18.5"
        assert (
            coordinator.session_input_parameters.get("coffee_name")
            == "Ethiopia Yirgacheffe"
        )

    @pytest.mark.asyncio
    async def test_start_session_handles_missing_or_unavailable_linked_inputs(
        self, coordinator: BookooCoordinator, mock_hass, mock_config_entry
    ):
        """Test _start_session with missing or unavailable linked entities."""
        mock_config_entry.options = {
            OPTION_LINKED_BEAN_WEIGHT_ENTITY: "input_number.test_bean_weight_missing",  # Will return None from states.get
            OPTION_LINKED_COFFEE_NAME_ENTITY: "input_text.test_coffee_name_unavailable",  # Will return state UNAVAILABLE
        }
        coordinator._load_options()

        mock_coffee_unavailable_state = MagicMock()
        mock_coffee_unavailable_state.state = STATE_UNAVAILABLE

        def mock_states_get_side_effect(entity_id):
            if entity_id == "input_text.test_coffee_name_unavailable":
                return mock_coffee_unavailable_state
            return None  # For input_number.test_bean_weight_missing

        mock_hass.states.get.side_effect = mock_states_get_side_effect

        await coordinator._start_session(trigger="test_missing_inputs")

        assert "bean_weight" not in coordinator.session_input_parameters
        assert "coffee_name" not in coordinator.session_input_parameters

    @pytest.mark.asyncio
    @freeze_time("2023-01-01 12:00:00")
    async def test_stop_session_aborts_if_too_short(
        self, coordinator: BookooCoordinator, mock_hass, mock_config_entry, caplog
    ):
        """Test _stop_session aborts logging if shot duration is less than min_shot_duration."""
        caplog.set_level(logging.INFO, logger="custom_components.bookoo.coordinator")
        min_duration_config = 15
        mock_config_entry.options = {
            OPTION_MIN_SHOT_DURATION: min_duration_config,
            OPTION_LINKED_BEAN_WEIGHT_ENTITY: "input_number.test_bean_weight",
        }
        coordinator._load_options()
        assert coordinator.min_shot_duration == min_duration_config

        # Mock linked entity for input parameters
        mock_bean_weight_state = MagicMock()
        mock_bean_weight_state.state = "20.0"
        mock_hass.states.get.side_effect = (
            lambda entity_id: mock_bean_weight_state
            if entity_id == "input_number.test_bean_weight"
            else None
        )

        # Start a session
        await coordinator._start_session(trigger="test_short_shot_trigger")
        assert coordinator.is_shot_active
        expected_input_params = {"bean_weight": "20.0"}
        assert coordinator.session_input_parameters == expected_input_params

        # Simulate time passing for a short shot (less than min_duration_config)
        with freeze_time("2023-01-01 12:00:10"):  # 10 seconds < 15 seconds
            await coordinator._stop_session(stop_reason="test_too_short")

        # Assert event was NOT fired for full completion
        mock_hass.bus.async_fire.assert_not_called()
        # Well, it might be called by other things, let's be specific if EVENT_BOOKOO_SHOT_COMPLETED was not fired.
        # For now, let's check last_shot_data status

        assert not coordinator.is_shot_active
        assert coordinator.last_shot_data["status"] == "aborted_too_short"
        assert coordinator.last_shot_data["duration_seconds"] == 10.0
        assert coordinator.last_shot_data["input_parameters"] == expected_input_params
        assert (
            "Shot duration (10.00 s) is less than minimum configured (15 s). Aborting full log."
            in caplog.text
        )

    @pytest.mark.asyncio
    @freeze_time("2023-01-01 13:00:00")
    @patch(
        "custom_components.bookoo.coordinator.async_add_shot_record",
        new_callable=AsyncMock,
    )
    async def test_stop_session_logs_valid_shot_with_inputs(
        self,
        mock_coordinator_add_shot_record: AsyncMock,
        coordinator: BookooCoordinator,
        mock_hass,
        mock_config_entry,
        caplog,
    ):
        """Test _stop_session logs a valid shot and includes input_parameters in the event."""
        caplog.set_level(logging.INFO, logger="custom_components.bookoo.coordinator")
        min_duration_config = 10
        mock_config_entry.options = {
            OPTION_MIN_SHOT_DURATION: min_duration_config,
            OPTION_LINKED_BEAN_WEIGHT_ENTITY: "input_number.test_bean_weight",
            OPTION_LINKED_COFFEE_NAME_ENTITY: "input_text.test_coffee_name",
        }
        coordinator._load_options()

        # Mock linked entities
        mock_bean_weight_state = MagicMock()
        mock_bean_weight_state.state = "19.2"
        mock_coffee_name_state = MagicMock()
        mock_coffee_name_state.state = "Dark Roast Special"

        def mock_states_get_side_effect(entity_id):
            if entity_id == "input_number.test_bean_weight":
                return mock_bean_weight_state
            if entity_id == "input_text.test_coffee_name":
                return mock_coffee_name_state
            return None

        mock_hass.states.get.side_effect = mock_states_get_side_effect

        # Start a session
        await coordinator._start_session(trigger="test_valid_shot_trigger")
        assert coordinator.is_shot_active
        expected_input_params = {
            "bean_weight": "19.2",
            "coffee_name": "Dark Roast Special",
        }
        assert coordinator.session_input_parameters == expected_input_params

        # Simulate time passing for a valid shot
        with freeze_time("2023-01-01 13:00:25"):  # 25 seconds >= 10 seconds
            await coordinator._stop_session(stop_reason="test_valid_complete")

        assert not coordinator.is_shot_active
        mock_hass.bus.async_fire.assert_called_once()
        fired_event_name, fired_event_data = mock_hass.bus.async_fire.call_args[0]

        assert fired_event_name == EVENT_BOOKOO_SHOT_COMPLETED
        assert fired_event_data["status"] == "completed"
        assert fired_event_data["duration_seconds"] == 25.0
        assert fired_event_data["input_parameters"] == expected_input_params
        assert coordinator.last_shot_data == fired_event_data
        assert "Fired EVENT_BOOKOO_SHOT_COMPLETED" in caplog.text

    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator._stop_session",
        new_callable=AsyncMock,
    )
    async def test_handle_char_update_auto_stop(
        self, mock_stop_session: AsyncMock, coordinator: BookooCoordinator, mock_hass
    ):
        """Test auto-stop from command characteristic."""
        # Similar to auto-start, hass.async_create_task handles the coroutine execution.

        # Scenario 1: Active shot, auto-stop message received
        coordinator.is_shot_active = True
        auto_stop_payload = bytes.fromhex("030d00000000000000000000000000000000000e")

        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, auto_stop_payload
        )
        await asyncio.sleep(0)

        mock_stop_session.assert_not_called()  # Coordinator doesn't decode raw command bytes to stop session

        # Scenario 2: No active shot, auto-stop message received
        mock_stop_session.reset_mock()
        coordinator.is_shot_active = False

        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, auto_stop_payload
        )
        await asyncio.sleep(0)

        mock_stop_session.assert_not_called()

    @freeze_time(
        "2023-01-01 12:00:00 UTC"
    )  # Freeze time for consistent elapsed_seconds
    @pytest.mark.asyncio
    async def test_handle_char_update_weight_data_active_shot(
        self,
        coordinator: BookooCoordinator,
        mock_hass,
        caplog,  # Add caplog fixture
    ):
        """Test weight data handling during an active shot."""
        # Setup active shot and session_start_time_utc using frozen time
        # Patch dt_util.utcnow within the coordinator's module for when _start_session (or similar) would use it.
        # However, for this specific test of _handle_characteristic_update, it relies on an *existing* session_start_time_utc.
        caplog.set_level(
            logging.DEBUG, logger="custom_components.bookoo.coordinator"
        )  # Set log level
        with patch(
            "custom_components.bookoo.coordinator.dt_util"
        ) as mock_dt_util_coord:
            mock_dt_util_coord.utcnow.return_value = datetime(
                2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
            )
            coordinator.session_start_time_utc = (
                mock_dt_util_coord.utcnow()
            )  # Manually set for test clarity
        coordinator.is_shot_active = True
        coordinator.session_flow_profile = []  # Ensure it's empty before test
        coordinator.session_weight_profile = []  # Ensure it's empty before test
        coordinator.session_scale_timer_profile = []  # Ensure it's empty

        # Set the mock scale's attributes to what the coordinator should read
        coordinator.scale.weight = 10.0  # type: ignore[misc]
        coordinator.scale.flow_rate = 2.5  # type: ignore[misc]
        coordinator.scale.timer = 5.0  # type: ignore[misc]  # Stored in seconds on BookooScale mock

        # Data for weight char updates is read from self.scale attributes directly by the coordinator,
        # not from a decode function at this level.

        # Simulate time has passed for elapsed_seconds calculation
        with freeze_time("2023-01-01 12:00:05 UTC") as _:
            with patch(
                "custom_components.bookoo.coordinator.BookooCoordinator.async_update_listeners",
                new_callable=MagicMock,
                create=True,
            ) as mock_update_listeners:
                coordinator._handle_characteristic_update(
                    UPDATE_SOURCE_WEIGHT_CHAR, None
                )
                await asyncio.sleep(0)

                assert len(coordinator.session_flow_profile) == 1
                assert coordinator.session_flow_profile[0] == (5.0, 2.5)

                assert len(coordinator.session_weight_profile) == 1
                assert coordinator.session_weight_profile[0] == (5.0, 10.0)

                assert len(coordinator.session_scale_timer_profile) == 1
                assert coordinator.session_scale_timer_profile[0] == (
                    5.0,
                    5000,
                )  # Timer stored as ms in profile

            mock_update_listeners.assert_called_once()

        # Scenario: Shot not active, should not process weight data for profiles
        coordinator.is_shot_active = False
        coordinator.session_flow_profile = []
        coordinator.session_scale_timer_profile = []
        mock_update_listeners.reset_mock()

        with freeze_time("2023-01-01 12:00:10 UTC"):
            coordinator._handle_characteristic_update(UPDATE_SOURCE_WEIGHT_CHAR, None)
            await asyncio.sleep(0)

            assert len(coordinator.session_flow_profile) == 0
            assert len(coordinator.session_scale_timer_profile) == 0
            # aiobookoo_decode is not used at this level; decoding happens in aiobookoov2 or data is read directly.
            mock_update_listeners.assert_not_called()

    # --- Tests for Decoded Command Characteristic Handling ---

    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator._start_session",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_handle_char_update_decoded_auto_start(
        self,
        mock_start_session: AsyncMock,
        coordinator: BookooCoordinator,
        mock_hass,  # Ensure mock_hass is included if used by coordinator methods indirectly
    ):
        """Test auto-start from command characteristic via decoded event."""
        # Scenario 1: No active shot, correct decoded auto-start event received
        coordinator.is_shot_active = False
        decoded_start_event = {"type": "auto_timer", "event": "start"}
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_start_event
        )
        await asyncio.sleep(0)  # Allow any scheduled tasks to run
        mock_start_session.assert_called_once_with(trigger="scale_auto_dict")

        # Scenario 2: Shot already active, decoded auto-start event received
        mock_start_session.reset_mock()
        coordinator.is_shot_active = True
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_start_event
        )
        await asyncio.sleep(0)
        mock_start_session.assert_not_called()

        # Scenario 3: Decoded data is not the expected event type/event
        mock_start_session.reset_mock()
        coordinator.is_shot_active = False
        other_decoded_event = {"type": "other_event", "event": "some_value"}
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, other_decoded_event
        )
        await asyncio.sleep(0)
        mock_start_session.assert_not_called()

        # Scenario 4: Decoded data is None (if aiobookoov2 could produce this for command char)
        # Based on current coordinator logic, if data is None for command char, it's not explicitly processed.
        mock_start_session.reset_mock()
        coordinator.is_shot_active = False
        coordinator._handle_characteristic_update(UPDATE_SOURCE_COMMAND_CHAR, None)
        await asyncio.sleep(0)
        mock_start_session.assert_not_called()

    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator._stop_session",
        new_callable=AsyncMock,
    )
    @pytest.mark.asyncio
    async def test_handle_char_update_decoded_auto_stop(
        self,
        mock_stop_session: AsyncMock,
        coordinator: BookooCoordinator,
        mock_hass,
    ):
        """Test auto-stop from command characteristic via decoded event."""
        # Test with data already decoded as a dict
        decoded_stop_event = {"type": "auto_timer", "event": "stop"}

        # Scenario 1: Active shot, decoded auto-stop event received
        coordinator.is_shot_active = True
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_stop_event
        )
        await asyncio.sleep(0)
        mock_stop_session.assert_called_once_with(stop_reason="scale_auto_dict")

        # Scenario 2: No active shot, decoded auto-stop event received
        mock_stop_session.reset_mock()
        coordinator.is_shot_active = False
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, decoded_stop_event
        )
        await asyncio.sleep(0)
        mock_stop_session.assert_not_called()

        # Scenario 3: Decoded data is not the expected event
        mock_stop_session.reset_mock()
        coordinator.is_shot_active = True
        coordinator._handle_characteristic_update(
            UPDATE_SOURCE_COMMAND_CHAR, {"type": "other_event", "event": "some_value"}
        )
        await asyncio.sleep(0)
        mock_stop_session.assert_not_called()

        # Scenario 4: Decoded data is None (simulates aiobookoo_decode not parsing this specific command)
        mock_stop_session.reset_mock()
        coordinator.is_shot_active = True
        # Scenario 4: Decoded data is None (simulates aiobookoov2 not producing a dict for some command char data)
        # Based on current coordinator logic, if data is None for command char, _stop_session is not called.
        coordinator._handle_characteristic_update(UPDATE_SOURCE_COMMAND_CHAR, None)
        await asyncio.sleep(0)
        mock_stop_session.assert_not_called()  # mock_stop_session was reset before this scenario

    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator.async_update_listeners"
    )
    @freeze_time("2023-01-01 12:00:00 UTC")
    @pytest.mark.asyncio
    async def test_start_session(
        self,
        mock_update_listeners: MagicMock,
        coordinator: BookooCoordinator,
        mock_config_entry: MagicMock,
        mock_hass: MagicMock,
    ):
        """Test the _start_session method, including reading linked inputs."""
        # Configure linked entities in options
        mock_config_entry.options = {
            "linked_bean_weight_entity": "input_number.test_bean_weight",
            "linked_coffee_name_entity": "input_text.test_coffee_name",
            "non_existent_entity_link": "input_number.non_existent",  # Test handling of missing entities
        }
        coordinator._load_options()  # Reload options as coordinator is already initialized

        # Mock return values for hass.states.get
        def mock_states_get_side_effect(entity_id):
            if entity_id == "input_number.test_bean_weight":
                mock_state = MagicMock()
                mock_state.state = "18.5"
                return mock_state
            if entity_id == "input_text.test_coffee_name":
                mock_state = MagicMock()
                mock_state.state = "Test Coffee"
                return mock_state
            return None  # For non_existent_entity_link or other calls

        mock_hass.states.get.side_effect = mock_states_get_side_effect

        # Initial state for profiles to ensure they are reset
        coordinator.session_flow_profile = [(1.0, 1.0)]
        coordinator.session_scale_timer_profile = [(1.0, 1000)]
        coordinator.is_shot_active = False  # Ensure it starts as false

        await coordinator._start_session(trigger="pytest_trigger")

        assert coordinator.is_shot_active is True
        assert coordinator.session_start_time_utc == datetime(
            2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
        )
        assert coordinator.session_flow_profile == []  # Should be reset
        assert coordinator.session_scale_timer_profile == []  # Should be reset
        assert coordinator.session_start_trigger == "pytest_trigger"

        expected_input_params = {"bean_weight": "18.5", "coffee_name": "Test Coffee"}
        assert coordinator.session_input_parameters == expected_input_params

        mock_update_listeners.assert_called_once()

    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator.async_update_listeners"
    )
    @patch(
        "custom_components.bookoo.coordinator._LOGGER"
    )  # To check log messages if needed
    @freeze_time("2023-01-01 12:00:00 UTC")
    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.coordinator.async_add_shot_record",
        new_callable=AsyncMock,
    )
    async def test_stop_session_valid_shot(
        self,
        mock_coordinator_add_shot_record: AsyncMock,
        mock_logger: MagicMock,
        mock_update_listeners: MagicMock,
        coordinator: BookooCoordinator,
        mock_config_entry: MagicMock,
        mock_hass: MagicMock,
        mock_scale: MagicMock,
    ):
        """Test _stop_session for a valid, completed shot."""
        # Setup: Configure minimum duration
        min_duration = 5.0
        mock_config_entry.options = {"minimum_shot_duration_seconds": min_duration}
        # Assuming device_id for event is derived from config_entry.unique_id or a fixed value if not.
        # If coordinator.device_info is used, mock that instead or ensure unique_id is sufficient.
        # For now, let's assume unique_id is used or we mock device_id source if different.
        # coordinator.py uses self.config_entry.unique_id if available, else self.config_entry.entry_id for device_id
        mock_config_entry.unique_id = "test_device_id_for_event"

        # Setup: Start a session state
        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        coordinator.is_shot_active = True
        coordinator.session_start_time_utc = start_time
        coordinator.session_flow_profile = [(1.0, 1.5), (2.0, 2.5)]
        coordinator.session_scale_timer_profile = [(1.0, 1000), (2.0, 2000)]
        coordinator.session_input_parameters = {"bean_weight": "18.0"}  # Corrected key
        coordinator.session_start_trigger = "scale_auto"

        mock_scale.weight = 36.0  # Final weight

        # Action: Stop the session after a valid duration (e.g., 30 seconds later)
        with freeze_time("2023-01-01 12:00:30 UTC") as _:
            end_time_actual = datetime(
                2023, 1, 1, 12, 0, 30, tzinfo=timezone.utc
            )  # Correctly get frozen time
            await coordinator._stop_session(stop_reason="ha_service")

            # Assertions for event firing
            mock_hass.bus.async_fire.assert_called_once()
            event_call_args = mock_hass.bus.async_fire.call_args[0]
            assert event_call_args[0] == EVENT_BOOKOO_SHOT_COMPLETED

            event_data = event_call_args[1]
            assert event_data["device_id"] == mock_config_entry.unique_id
            assert event_data["entry_id"] == mock_config_entry.entry_id
            assert event_data["start_time_utc"] == start_time.isoformat()
            assert event_data["end_time_utc"] == end_time_actual.isoformat()
            assert event_data["duration_seconds"] == 30.0
            assert event_data["final_weight_grams"] == 36.0
            assert event_data["flow_profile"] == [(1.0, 1.5), (2.0, 2.5)]
            assert event_data["scale_timer_profile"] == [(1.0, 1000), (2.0, 2000)]
            assert event_data["input_parameters"] == {
                "bean_weight": "18.0"
            }  # Corrected key
            assert event_data["start_trigger"] == "scale_auto"
            assert event_data["stop_reason"] == "ha_service"
            assert event_data["status"] == "completed"

            # Assertions for state reset
            assert coordinator.is_shot_active is False
            assert coordinator.session_start_time_utc is None
            assert coordinator.session_flow_profile == []
            assert coordinator.session_scale_timer_profile == []
            assert coordinator.session_input_parameters == {}
            assert coordinator.session_start_trigger is None

            # Assertions for last_shot_data
            assert (
                coordinator.last_shot_data["start_time_utc"] == start_time.isoformat()
            )
            assert coordinator.last_shot_data["duration_seconds"] == 30.0
            assert coordinator.last_shot_data["status"] == "completed"

            mock_update_listeners.assert_called_once()

    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator.async_update_listeners"
    )
    @patch("custom_components.bookoo.coordinator._LOGGER")
    @freeze_time("2023-01-01 12:00:00 UTC")
    @pytest.mark.asyncio
    async def test_stop_session_too_short(
        self,
        mock_logger: MagicMock,
        mock_update_listeners: MagicMock,
        coordinator: BookooCoordinator,
        mock_config_entry: MagicMock,
        mock_hass: MagicMock,
        mock_scale: MagicMock,
    ):
        """Test _stop_session for a shot that is too short."""
        min_duration = 10.0
        mock_config_entry.options = {"minimum_shot_duration_seconds": min_duration}
        mock_config_entry.unique_id = "test_device_short_shot"

        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        coordinator.is_shot_active = True
        coordinator.session_start_time_utc = start_time
        coordinator.session_flow_profile = [(1.0, 0.5)]  # Some minimal data
        coordinator.session_scale_timer_profile = [(1.0, 1000)]
        coordinator.session_input_parameters = {}
        coordinator.session_start_trigger = "test"

        mock_scale.weight = 5.0  # Final weight for the short attempt

        # Stop the session after only 3 seconds
        with freeze_time("2023-01-01 12:00:03 UTC"):
            await coordinator._stop_session(stop_reason="manual_stop_short")

            mock_hass.bus.async_fire.assert_not_called()

            assert coordinator.is_shot_active is False
            assert coordinator.session_start_time_utc is None
            assert coordinator.session_flow_profile == []
            assert coordinator.session_scale_timer_profile == []
            assert coordinator.session_input_parameters == {}
            assert coordinator.session_start_trigger is None

            # Check last_shot_data for "aborted_too_short" status
            assert (
                coordinator.last_shot_data["start_time_utc"] == start_time.isoformat()
            )
            assert coordinator.last_shot_data["duration_seconds"] == 3.0
            assert (
                coordinator.last_shot_data["final_weight_grams"] == 0.0
            )  # Aborted short shots have 0.0 final weight
            assert coordinator.last_shot_data["status"] == "aborted_too_short"
            assert (
                coordinator.last_shot_data["flow_profile"] == []
            )  # Aborted short shots have empty profile
            assert (
                coordinator.last_shot_data["scale_timer_profile"] == []
            )  # Aborted short shots have empty profile
            assert coordinator.last_shot_data["input_parameters"] == {}

            mock_update_listeners.assert_called_once()
            # mock_logger.info.assert_any_call("Espresso shot session aborted (too short): %s seconds", 3.0)

    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator.async_update_listeners"
    )
    @patch("custom_components.bookoo.coordinator._LOGGER")
    @freeze_time("2023-01-01 12:00:00 UTC")
    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.coordinator.async_add_shot_record",
        new_callable=AsyncMock,
    )
    async def test_stop_session_disconnected_shot(
        self,
        mock_coordinator_add_shot_record: AsyncMock,
        mock_logger: MagicMock,
        mock_update_listeners: MagicMock,
        coordinator: BookooCoordinator,
        mock_config_entry: MagicMock,
        mock_hass: MagicMock,
        mock_scale: MagicMock,
    ):
        """Test _stop_session for a shot that ends due to disconnection."""
        min_duration = 10.0  # Min duration should not prevent event on disconnect
        mock_config_entry.options = {"minimum_shot_duration_seconds": min_duration}
        mock_config_entry.unique_id = "test_device_disconnected"

        start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        coordinator.is_shot_active = True
        coordinator.session_start_time_utc = start_time
        coordinator.session_flow_profile = [(1.0, 0.2)]
        coordinator.session_scale_timer_profile = [(1.0, 1000)]
        coordinator.session_input_parameters = {"info": "test_disconnect"}
        coordinator.session_start_trigger = "ble_auto"

        mock_scale.weight = 2.0  # Final weight at disconnect

        # Stop the session after only 3 seconds due to "disconnection"
        with freeze_time("2023-01-01 12:00:03 UTC") as _:
            end_time_actual = datetime(
                2023, 1, 1, 12, 0, 3, tzinfo=timezone.utc
            )  # Get actual frozen time
            await coordinator._stop_session(stop_reason="disconnected")

            # Event should still fire for disconnections, regardless of duration
            mock_hass.bus.async_fire.assert_called_once()
            event_call_args = mock_hass.bus.async_fire.call_args[0]
            assert event_call_args[0] == EVENT_BOOKOO_SHOT_COMPLETED

            event_data = event_call_args[1]
            assert event_data["device_id"] == mock_config_entry.unique_id
            assert event_data["entry_id"] == mock_config_entry.entry_id
            assert event_data["start_time_utc"] == start_time.isoformat()
            assert event_data["end_time_utc"] == end_time_actual.isoformat()
            assert event_data["duration_seconds"] == 3.0
            assert event_data["final_weight_grams"] == 2.0
            assert event_data["flow_profile"] == [(1.0, 0.2)]
            assert event_data["scale_timer_profile"] == [(1.0, 1000)]
            assert event_data["input_parameters"] == {"info": "test_disconnect"}
            assert event_data["start_trigger"] == "ble_auto"
            assert event_data["stop_reason"] == "disconnected"
            assert event_data["status"] == "aborted_disconnected"

            # Assertions for state reset
            assert coordinator.is_shot_active is False
            assert coordinator.session_start_time_utc is None
            # Check other state resets as in previous tests (profiles, etc.)
            assert coordinator.session_flow_profile == []
            assert coordinator.session_scale_timer_profile == []
            assert coordinator.session_input_parameters == {}
            assert coordinator.session_start_trigger is None

            # Assertions for last_shot_data
            assert coordinator.last_shot_data["status"] == "aborted_disconnected"
            assert coordinator.last_shot_data["duration_seconds"] == 3.0

            mock_update_listeners.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator._start_session",
        new_callable=AsyncMock,
    )
    async def test_async_start_shot_service(
        self,
        mock_start_session: AsyncMock,
        coordinator: BookooCoordinator,
        mock_scale: MagicMock,
        mock_hass: MagicMock,
    ):
        """Test the async_start_shot_service method."""
        await coordinator.async_start_shot_service()
        await asyncio.sleep(0)

        mock_start_session.assert_called_once_with(trigger="ha_service")
        mock_scale.tare_and_start_timer.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "custom_components.bookoo.coordinator.BookooCoordinator._stop_session",
        new_callable=AsyncMock,
    )
    async def test_async_stop_shot_service(
        self,
        mock_stop_session: AsyncMock,
        coordinator: BookooCoordinator,
        mock_scale: MagicMock,
        mock_hass: MagicMock,
    ):
        """Test the async_stop_shot_service method."""
        coordinator.is_shot_active = (
            True  # Ensure shot is active for stop service to proceed
        )
        await coordinator.async_stop_shot_service()
        await asyncio.sleep(0)

        mock_stop_session.assert_called_once_with(stop_reason="ha_service")
        mock_scale.stop_timer.assert_called_once_with()
