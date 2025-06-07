# tests/unit/test_session_manager.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta
import collections

from homeassistant.core import HomeAssistant, State
from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.exceptions import HomeAssistantError

from custom_components.bookoo.session_manager import SessionManager, MAX_PROFILE_POINTS
from custom_components.bookoo.coordinator import (
    BookooCoordinator,
)  # For type hinting mock
from custom_components.bookoo.const import (
    BookooConfig,
    DOMAIN,
)  # Assuming DOMAIN is needed for event name
from custom_components.bookoo.types import (
    BookooShotCompletedEventDataModel,
    FlowDataPoint,
    ScaleTimerDataPoint,
    WeightDataPoint,
)

# --- Fixtures ---


@pytest.fixture
def mock_hass() -> MagicMock:
    """Fixture for a mock HomeAssistant instance."""
    hass = MagicMock(spec=HomeAssistant)
    hass.states = MagicMock()
    hass.bus = MagicMock()
    hass.bus.async_fire = MagicMock()
    hass.config_entries = (
        MagicMock()
    )  # Added for _options_update_callback if SessionManager uses it indirectly
    return hass


@pytest.fixture
def mock_bookoo_config() -> MagicMock:
    """Fixture for a mock BookooConfig instance."""
    config = MagicMock(spec=BookooConfig)
    config.min_shot_duration = 10
    config.max_shot_duration = 45
    config.auto_stop_flow_cutoff_enabled = True
    config.auto_stop_flow_cutoff_threshold_gps = 0.2
    config.auto_stop_flow_cutoff_duration_seconds = 3
    config.auto_stop_initial_stable_flow_seconds = 5
    config.linked_bean_weight_entity = "sensor.bean_weight"
    config.linked_coffee_name_entity = "input_text.coffee_name"
    config.linked_grind_setting_entity = "input_text.grind_setting"
    config.linked_brew_temperature_entity = None  # Example of one not set
    return config


@pytest.fixture
def mock_scale() -> MagicMock:
    """Fixture for a mock aiobookoov2.BookooScale instance."""
    scale = MagicMock()
    scale.weight = 0.0
    scale.flow_rate = 0.0
    scale.timer = 0
    return scale


@pytest.fixture
def mock_coordinator(
    mock_hass: MagicMock, mock_bookoo_config: MagicMock, mock_scale: MagicMock
) -> MagicMock:
    """Fixture for a mock BookooCoordinator instance."""
    coordinator = MagicMock(spec=BookooCoordinator)
    coordinator.hass = mock_hass
    coordinator.bookoo_config = mock_bookoo_config
    coordinator.scale = mock_scale
    coordinator.name = "Test Bookoo Scale"
    coordinator.config_entry = MagicMock()
    coordinator.config_entry.unique_id = "test_unique_id"
    coordinator.config_entry.entry_id = "test_entry_id"
    coordinator.config_entry.domain = DOMAIN  # For event name construction
    coordinator.async_update_listeners = MagicMock()
    # Mock real-time analytics attributes that SessionManager might read
    coordinator.realtime_channeling_status = "None"
    coordinator.realtime_pre_infusion_detected = False
    coordinator.realtime_pre_infusion_duration_seconds = None
    coordinator.realtime_extraction_uniformity_metric = 0.0
    coordinator.realtime_shot_quality_score = 0.0
    return coordinator


@pytest.fixture
def session_manager(
    mock_hass: MagicMock, mock_coordinator: MagicMock
) -> SessionManager:
    """Fixture to create a SessionManager instance."""
    return SessionManager(mock_hass, mock_coordinator)


@pytest.fixture
def mock_utcnow(monkeypatch):
    """Fixture to mock dt_util.utcnow."""
    mock_now = MagicMock(
        return_value=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    )
    # Patch utcnow directly on the original module homeassistant.util.dt
    monkeypatch.setattr("homeassistant.util.dt.utcnow", mock_now)
    return mock_now


@pytest.fixture
async def mock_async_add_shot_record():
    """Patches custom_components.bookoo.session_manager.async_add_shot_record."""
    with patch(
        "custom_components.bookoo.session_manager.async_add_shot_record",
        new_callable=AsyncMock,
    ) as mock_add_record:
        yield mock_add_record


# --- Test Cases ---


def test_session_manager_initialization(session_manager: SessionManager):
    """Test the initial state of SessionManager."""
    assert session_manager.is_shot_active is False
    assert session_manager.session_start_time_utc is None
    assert len(session_manager.session_flow_profile) == 0
    assert session_manager.session_flow_profile.maxlen == MAX_PROFILE_POINTS
    assert len(session_manager.session_weight_profile) == 0
    assert session_manager.session_weight_profile.maxlen == MAX_PROFILE_POINTS
    assert len(session_manager.session_scale_timer_profile) == 0
    assert session_manager.session_scale_timer_profile.maxlen == MAX_PROFILE_POINTS
    assert session_manager.session_input_parameters == {}
    assert session_manager.session_start_trigger is None
    assert session_manager.last_shot_data is None
    assert session_manager._auto_stop_flow_stable_start_time is None
    assert session_manager._auto_stop_flow_below_cutoff_start_time is None
    assert isinstance(session_manager._session_lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_start_session_success(
    session_manager: SessionManager,
    mock_coordinator: MagicMock,
    mock_utcnow: MagicMock,
    mock_hass: MagicMock,  # For hass.states.get mock
):
    """Test successful start of a new shot session."""
    trigger = "test_trigger"

    # Mock _read_linked_entities to avoid actual HA state calls in this specific test unit
    with patch.object(session_manager, "_read_linked_entities") as mock_read_linked:
        await session_manager.start_session(trigger)

        assert session_manager.is_shot_active is True
        assert session_manager.session_start_time_utc == datetime(
            2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
        )
        assert session_manager.session_start_trigger == trigger
        mock_read_linked.assert_called_once()
        mock_coordinator.async_update_listeners.assert_called_once()

        # Check that internal state was reset (deques are empty)
        assert len(session_manager.session_flow_profile) == 0
        assert len(session_manager.session_weight_profile) == 0
        assert len(session_manager.session_scale_timer_profile) == 0
        assert (
            session_manager.session_input_parameters == {}
        )  # _read_linked_entities is mocked, so this remains empty


@pytest.mark.asyncio
async def test_start_session_already_active(session_manager: SessionManager):
    """Test attempting to start a session when one is already active."""
    session_manager.is_shot_active = True  # Simulate an active shot
    original_start_time = datetime(2023, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
    session_manager.session_start_time_utc = original_start_time

    with pytest.raises(HomeAssistantError) as exc_info:
        await session_manager.start_session("another_trigger")

    assert "shot_already_active" in str(exc_info.value.translation_key)
    # Ensure state hasn't changed from the previous active session
    assert session_manager.is_shot_active is True
    assert session_manager.session_start_time_utc == original_start_time


# --- Tests for _read_linked_input_to_params and _read_linked_entities ---


@pytest.mark.parametrize(
    "entity_id, state_value, expected_log_level, expected_param_value",
    [
        ("sensor.test_sensor", "valid_state", "DEBUG", "valid_state"),
        ("sensor.test_sensor_unavailable", STATE_UNAVAILABLE, "WARNING", None),
        ("sensor.test_sensor_unknown", STATE_UNKNOWN, "WARNING", None),
        (
            "sensor.non_existent_sensor",
            None,
            "WARNING",
            None,
        ),  # hass.states.get returns None
    ],
)
def test_read_linked_input_to_params(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    caplog: pytest.LogCaptureFixture,
    entity_id: str,
    state_value: str | None,
    expected_log_level: str,
    expected_param_value: str | None,
):
    """Test _read_linked_input_to_params with various scenarios."""
    param_key = "test_param"
    param_description = "Test Parameter"
    session_manager.session_input_parameters = {}  # Reset for each test run

    if state_value is None and entity_id == "sensor.non_existent_sensor":
        mock_hass.states.get.return_value = None
    elif state_value is not None:
        mock_hass.states.get.return_value = State(entity_id, state_value)
    else:  # For STATE_UNAVAILABLE, STATE_UNKNOWN where state_value is the state itself
        mock_hass.states.get.return_value = State(entity_id, state_value)

    session_manager._read_linked_input_to_params(
        entity_id, param_key, param_description
    )

    mock_hass.states.get.assert_called_with(entity_id)

    if expected_param_value is not None:
        assert (
            session_manager.session_input_parameters[param_key] == expected_param_value
        )
        assert (
            f"Logged {param_description}: {expected_param_value} from {entity_id}"
            in caplog.text
        )
    else:
        assert param_key not in session_manager.session_input_parameters
        assert (
            f"Could not read state for linked {param_description} entity: {entity_id}"
            in caplog.text
        )

    assert expected_log_level in caplog.text  # Check if the log level matches


def test_read_linked_input_to_params_no_entity_id(
    session_manager: SessionManager, mock_hass: MagicMock
):
    """Test _read_linked_input_to_params when entity_id is None."""
    session_manager.session_input_parameters = {}
    mock_hass.states.get.reset_mock()  # Ensure it's not called

    session_manager._read_linked_input_to_params(None, "test_param", "Test Parameter")

    assert "test_param" not in session_manager.session_input_parameters
    mock_hass.states.get.assert_not_called()


def test_read_linked_entities_all_set_and_valid(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_coordinator: MagicMock,
    mock_bookoo_config: MagicMock,
):
    """Test _read_linked_entities when all configured entities are set and valid."""
    # Configure mock_bookoo_config (already done by fixture, but ensure they are distinct for clarity)
    mock_bookoo_config.linked_bean_weight_entity = "sensor.bean_weight_test"
    mock_bookoo_config.linked_coffee_name_entity = "input_text.coffee_name_test"
    mock_bookoo_config.linked_grind_setting_entity = "input_text.grind_setting_test"
    mock_bookoo_config.linked_brew_temperature_entity = "sensor.brew_temp_test"

    mock_coordinator.bookoo_config = (
        mock_bookoo_config  # Ensure coordinator has this config
    )

    # Mock hass.states.get to return specific states
    def mock_states_get(entity_id):
        if entity_id == "sensor.bean_weight_test":
            return State(entity_id, "18.5")
        if entity_id == "input_text.coffee_name_test":
            return State(entity_id, "Ethiopian Yirgacheffe")
        if entity_id == "input_text.grind_setting_test":
            return State(entity_id, "7.5")
        if entity_id == "sensor.brew_temp_test":
            return State(entity_id, "92.0")
        return None

    mock_hass.states.get = MagicMock(side_effect=mock_states_get)
    session_manager.session_input_parameters = {}

    session_manager._read_linked_entities()

    assert session_manager.session_input_parameters["bean_weight"] == "18.5"
    assert (
        session_manager.session_input_parameters["coffee_name"]
        == "Ethiopian Yirgacheffe"
    )
    assert session_manager.session_input_parameters["grind_setting"] == "7.5"
    assert session_manager.session_input_parameters["brew_temperature"] == "92.0"
    assert mock_hass.states.get.call_count == 4


def test_read_linked_entities_mixed_validity(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_coordinator: MagicMock,
    mock_bookoo_config: MagicMock,
):
    """Test _read_linked_entities with a mix of valid, None, and unavailable entities."""
    mock_bookoo_config.linked_bean_weight_entity = "sensor.valid_bean_weight"
    mock_bookoo_config.linked_coffee_name_entity = None  # Not configured
    mock_bookoo_config.linked_grind_setting_entity = "input_text.unavailable_grind"
    mock_bookoo_config.linked_brew_temperature_entity = "sensor.unknown_brew_temp"

    mock_coordinator.bookoo_config = mock_bookoo_config

    def mock_states_get(entity_id):
        if entity_id == "sensor.valid_bean_weight":
            return State(entity_id, "20.0")
        if entity_id == "input_text.unavailable_grind":
            return State(entity_id, STATE_UNAVAILABLE)
        if entity_id == "sensor.unknown_brew_temp":
            return State(entity_id, STATE_UNKNOWN)
        return None

    mock_hass.states.get = MagicMock(side_effect=mock_states_get)
    session_manager.session_input_parameters = {}

    session_manager._read_linked_entities()

    assert session_manager.session_input_parameters["bean_weight"] == "20.0"
    assert "coffee_name" not in session_manager.session_input_parameters
    assert "grind_setting" not in session_manager.session_input_parameters
    assert "brew_temperature" not in session_manager.session_input_parameters
    # Called for valid_bean_weight, unavailable_grind, unknown_brew_temp (None entity_id is skipped)
    assert mock_hass.states.get.call_count == 3


# --- Tests for add_flow_data, add_weight_data, add_scale_timer_data ---


@pytest.mark.parametrize(
    "method_name, profile_name, data_point_type, sample_args, check_auto_stop_call",
    [
        ("add_flow_data", "session_flow_profile", FlowDataPoint, (1.0, 0.5), True),
        (
            "add_weight_data",
            "session_weight_profile",
            WeightDataPoint,
            (2.0, 10.0),
            False,
        ),
        (
            "add_scale_timer_data",
            "session_scale_timer_profile",
            ScaleTimerDataPoint,
            (3.0, 3),
            False,
        ),
    ],
)
def test_add_data_point_when_shot_active(
    session_manager: SessionManager,
    method_name: str,
    profile_name: str,
    data_point_type: type,
    sample_args: tuple,
    check_auto_stop_call: bool,
):
    """Test adding data points when a shot is active."""
    session_manager.is_shot_active = True
    # Ensure deques are empty before the call
    getattr(session_manager, profile_name).clear()

    method_to_call = getattr(session_manager, method_name)

    if check_auto_stop_call:
        with patch.object(
            session_manager, "_check_auto_stop_flow_cutoff"
        ) as mock_check_auto_stop:
            method_to_call(*sample_args)
            mock_check_auto_stop.assert_called_once_with(
                sample_args[0], sample_args[1]
            )  # elapsed_time, flow_rate
    else:
        method_to_call(*sample_args)

    profile_deque = getattr(session_manager, profile_name)
    assert len(profile_deque) == 1
    assert isinstance(profile_deque[0], data_point_type)
    if data_point_type is FlowDataPoint:
        assert profile_deque[0] == FlowDataPoint(
            elapsed_time=sample_args[0], flow_rate=sample_args[1]
        )
    elif data_point_type is WeightDataPoint:
        assert profile_deque[0] == WeightDataPoint(
            elapsed_time=sample_args[0], weight=sample_args[1]
        )
    elif data_point_type is ScaleTimerDataPoint:
        assert profile_deque[0] == ScaleTimerDataPoint(
            elapsed_time=sample_args[0], timer_value=sample_args[1]
        )


@pytest.mark.parametrize(
    "method_name, profile_name, sample_args, check_auto_stop_call",
    [
        ("add_flow_data", "session_flow_profile", (1.0, 0.5), True),
        ("add_weight_data", "session_weight_profile", (2.0, 10.0), False),
        ("add_scale_timer_data", "session_scale_timer_profile", (3.0, 3), False),
    ],
)
def test_add_data_point_when_shot_inactive(
    session_manager: SessionManager,
    method_name: str,
    profile_name: str,
    sample_args: tuple,
    check_auto_stop_call: bool,
):
    """Test that data points are not added when a shot is inactive."""
    session_manager.is_shot_active = False
    getattr(session_manager, profile_name).clear()
    method_to_call = getattr(session_manager, method_name)

    if check_auto_stop_call:
        with patch.object(
            session_manager, "_check_auto_stop_flow_cutoff"
        ) as mock_check_auto_stop:
            method_to_call(*sample_args)
            mock_check_auto_stop.assert_not_called()
    else:
        method_to_call(*sample_args)

    profile_deque = getattr(session_manager, profile_name)
    assert len(profile_deque) == 0


# --- Tests for _check_auto_stop_flow_cutoff ---


@pytest.mark.asyncio
async def test_check_auto_stop_disabled(
    session_manager: SessionManager, mock_bookoo_config: MagicMock, mock_hass: MagicMock
):
    """Test auto-stop logic when it's disabled in config."""
    mock_bookoo_config.auto_stop_flow_cutoff_enabled = False
    mock_hass.async_create_task = MagicMock()  # To check if stop_session is scheduled

    session_manager._check_auto_stop_flow_cutoff(10.0, 0.1)  # elapsed_time, flow_rate

    mock_hass.async_create_task.assert_not_called()
    assert session_manager._auto_stop_flow_stable_start_time is None
    assert session_manager._auto_stop_flow_below_cutoff_start_time is None


@pytest.mark.asyncio
async def test_check_auto_stop_flow_above_threshold(
    session_manager: SessionManager,
    mock_bookoo_config: MagicMock,
    mock_hass: MagicMock,
    mock_utcnow: MagicMock,
):
    """Test auto-stop when flow is consistently above the cutoff threshold."""
    mock_bookoo_config.enable_auto_stop_flow_cutoff = True
    mock_bookoo_config.auto_stop_flow_cutoff_threshold = 0.2  # Actual attribute name
    mock_bookoo_config.auto_stop_pre_infusion_ignore_duration = (
        5.0  # Actual attribute name, ensure float
    )

    # Set other potentially relevant configs for auto-stop logic to defaults or sensible test values
    mock_bookoo_config.auto_stop_min_flow_for_stability = 0.1  # Default-like value
    mock_bookoo_config.auto_stop_min_duration_for_stability = 3.0  # Default-like value
    mock_bookoo_config.auto_stop_max_flow_variance_for_stability = (
        25.0  # Default-like value (%CV)
    )
    mock_bookoo_config.auto_stop_min_duration_for_cutoff = 2.0  # Default-like value

    mock_hass.async_create_task = MagicMock()

    # Simulate flow stable period met
    session_manager._auto_stop_flow_stable_start_time = (
        mock_utcnow.return_value - timedelta(seconds=6)
    )
    # Simulate previously being below cutoff, now recovered
    session_manager._auto_stop_flow_below_cutoff_start_time = (
        mock_utcnow.return_value - timedelta(seconds=1)
    )

    session_manager._check_auto_stop_flow_cutoff(
        10.0, 0.3
    )  # Flow (0.3) > threshold (0.2)

    mock_hass.async_create_task.assert_not_called()
    assert (
        session_manager._auto_stop_flow_below_cutoff_start_time is None
    )  # Should be reset


@pytest.mark.asyncio
async def test_check_auto_stop_initial_stable_period_not_met(
    session_manager: SessionManager,
    mock_bookoo_config: MagicMock,
    mock_hass: MagicMock,
    mock_utcnow: MagicMock,
):
    """Test auto-stop when flow drops but initial stable period is not yet met."""
    mock_bookoo_config.enable_auto_stop_flow_cutoff = True
    mock_bookoo_config.auto_stop_flow_cutoff_threshold = 0.2
    mock_bookoo_config.auto_stop_pre_infusion_ignore_duration = (
        3.0  # e.g., ignore first 3s
    )
    mock_bookoo_config.auto_stop_min_flow_for_stability = (
        0.3  # Stability requires flow >= 0.3
    )
    mock_bookoo_config.auto_stop_min_duration_for_stability = (
        5.0  # Stability requires 5s of stable flow
    )
    mock_bookoo_config.auto_stop_max_flow_variance_for_stability = 25.0  # %CV
    mock_hass.async_create_task = MagicMock()

    session_manager.is_shot_active = True  # Shot must be active
    session_manager._auto_stop_flow_stable_start_time = (
        None  # Ensure stability not yet met
    )
    session_manager._auto_stop_flow_below_cutoff_start_time = (
        None  # Ensure this starts as None
    )

    # Simulate being past ignore phase (e.g., at 4s), but stability criteria not met yet
    # Flow is low (0.1), below cutoff_threshold (0.2) and below min_flow_for_stability (0.3)
    # Because stability isn't met, the cutoff timer should not start.
    session_manager._check_auto_stop_flow_cutoff(
        current_elapsed_time=4.0, current_flow_rate=0.1
    )

    mock_hass.async_create_task.assert_not_called()
    assert (
        session_manager._auto_stop_flow_stable_start_time is None
    )  # Stability should not have been achieved
    assert (
        session_manager._auto_stop_flow_below_cutoff_start_time is None
    )  # This is the key assertion

    # Current elapsed time is less than initial_stable_flow_seconds
    session_manager._check_auto_stop_flow_cutoff(
        3.0, 0.1
    )  # Flow (0.1) < threshold (0.2)

    mock_hass.async_create_task.assert_not_called()
    assert session_manager._auto_stop_flow_stable_start_time is None  # Not set yet
    assert (
        session_manager._auto_stop_flow_below_cutoff_start_time
        == mock_utcnow.return_value
    )


@pytest.mark.asyncio
async def test_check_auto_stop_below_cutoff_duration_not_met(
    session_manager: SessionManager,
    mock_bookoo_config: MagicMock,
    mock_hass: MagicMock,
    mock_utcnow: MagicMock,
):
    """Test auto-stop when flow is below cutoff, stable period met, but cutoff duration not yet met."""
    mock_bookoo_config.auto_stop_flow_cutoff_enabled = True
    mock_bookoo_config.auto_stop_flow_cutoff_threshold_gps = 0.2
    mock_bookoo_config.auto_stop_initial_stable_flow_seconds = 5
    mock_bookoo_config.auto_stop_flow_cutoff_duration_seconds = 3
    mock_hass.async_create_task = MagicMock()

    # Simulate flow stable period met
    session_manager._auto_stop_flow_stable_start_time = (
        mock_utcnow.return_value - timedelta(seconds=6)
    )
    # Simulate flow just dropped below cutoff
    session_manager._auto_stop_flow_below_cutoff_start_time = (
        mock_utcnow.return_value - timedelta(seconds=1)
    )

    session_manager._check_auto_stop_flow_cutoff(
        10.0, 0.1
    )  # Flow (0.1) < threshold (0.2)

    mock_hass.async_create_task.assert_not_called()
    # _auto_stop_flow_below_cutoff_start_time should remain as it was (1 second ago)
    assert (
        session_manager._auto_stop_flow_below_cutoff_start_time
        == mock_utcnow.return_value - timedelta(seconds=1)
    )


@pytest.mark.asyncio
async def test_check_auto_stop_triggers_stop_session(
    session_manager: SessionManager,
    mock_bookoo_config: MagicMock,
    mock_hass: MagicMock,
    mock_utcnow: MagicMock,
):
    """Test auto-stop successfully triggers stop_session."""
    mock_bookoo_config.enable_auto_stop_flow_cutoff = True
    mock_bookoo_config.auto_stop_flow_cutoff_threshold = 0.2
    mock_bookoo_config.auto_stop_pre_infusion_ignore_duration = 3.0
    mock_bookoo_config.auto_stop_min_duration_for_cutoff = (
        2.0  # e.g., stop if below cutoff for 2s
    )
    # Ensure stability config is also set, as it's a prerequisite
    mock_bookoo_config.auto_stop_min_flow_for_stability = 0.1
    mock_bookoo_config.auto_stop_min_duration_for_stability = 3.0
    mock_bookoo_config.auto_stop_max_flow_variance_for_stability = 25.0

    mock_hass.async_create_task = MagicMock()
    # It's important to mock the stop_session method itself if we are only testing that it's *called*,
    # otherwise, the real stop_session will run and might have side effects or require more mocks.
    # For this test, we want to ensure async_create_task is called with the correct coroutine.
    # We can't directly assert the coroutine object itself easily without executing it.
    # So, we'll mock stop_session on the instance to check its call args if needed,
    # or just verify async_create_task is called.

    session_manager.is_shot_active = True
    # Simulate flow stable period met
    session_manager._auto_stop_flow_stable_start_time = (
        mock_utcnow.return_value - timedelta(seconds=10)
    )
    # Simulate flow has been below cutoff for longer than min_duration_for_cutoff
    time_flow_dropped = mock_utcnow.return_value - timedelta(
        seconds=mock_bookoo_config.auto_stop_min_duration_for_cutoff + 1
    )
    session_manager._auto_stop_flow_below_cutoff_start_time = time_flow_dropped

    # Call with current_elapsed_time > pre_infusion_ignore_duration, and low flow
    session_manager._check_auto_stop_flow_cutoff(
        current_elapsed_time=10.0, current_flow_rate=0.1
    )

    # Check that hass.async_create_task was called.
    # Verifying the argument (the coroutine) is tricky without more advanced mocking or letting it run.
    # For now, ensuring it's called is a good first step.
    mock_hass.async_create_task.assert_called_once()
    # To be more precise, you could patch session_manager.stop_session and check its call
    # args = mock_hass.async_create_task.call_args[0][0]
    # assert isinstance(args, asyncio.Task) # This is not directly testable for the coroutine itself
    # For now, we assume if async_create_task is called, it's with the right coroutine from the code path.
    mock_hass.async_create_task.assert_called_once()
    # To verify stop_session was the target, we'd ideally patch stop_session and check it was called.
    # For now, checking async_create_task is a good first step.
    # Example of patching stop_session for more direct assertion:
    # with patch.object(session_manager, "stop_session", new_callable=AsyncMock) as mock_stop_session_method:
    #     session_manager._check_auto_stop_flow_cutoff(15.0, 0.1)
    #     mock_stop_session_method.assert_called_once_with(stop_reason="auto_flow_cutoff")


@pytest.mark.asyncio
async def test_check_auto_stop_flow_recovers_before_cutoff(
    session_manager: SessionManager,
    mock_bookoo_config: MagicMock,
    mock_hass: MagicMock,
    mock_utcnow: MagicMock,
):
    """Test auto-stop when flow drops then recovers before cutoff duration is met."""
    mock_bookoo_config.enable_auto_stop_flow_cutoff = True
    mock_bookoo_config.auto_stop_flow_cutoff_threshold = 0.2
    mock_bookoo_config.auto_stop_pre_infusion_ignore_duration = 3.0
    mock_bookoo_config.auto_stop_min_duration_for_cutoff = (
        3.0  # Stop if below cutoff for 3s
    )

    # Ensure stability config is also set, as it's a prerequisite for cutoff check
    mock_bookoo_config.auto_stop_min_flow_for_stability = 0.1
    mock_bookoo_config.auto_stop_min_duration_for_stability = 3.0
    mock_bookoo_config.auto_stop_max_flow_variance_for_stability = 25.0

    mock_hass.async_create_task = MagicMock()
    session_manager.is_shot_active = True

    # Simulate flow stable period met
    session_manager._auto_stop_flow_stable_start_time = (
        mock_utcnow.return_value - timedelta(seconds=10)
    )
    # Simulate flow was below cutoff, but for less than min_duration_for_cutoff (e.g., 1 second ago)
    session_manager._auto_stop_flow_below_cutoff_start_time = (
        mock_utcnow.return_value - timedelta(seconds=1)
    )

    # Now, flow recovers to be above the cutoff threshold
    # current_elapsed_time is past pre_infusion_ignore_duration
    session_manager._check_auto_stop_flow_cutoff(
        current_elapsed_time=10.0, current_flow_rate=0.3
    )  # Flow 0.3 > threshold 0.2

    mock_hass.async_create_task.assert_not_called()  # Stop session should not be called
    assert (
        session_manager._auto_stop_flow_below_cutoff_start_time is None
    )  # Timer should be reset
    session_manager._auto_stop_flow_below_cutoff_start_time = (
        mock_utcnow.return_value - timedelta(seconds=1)
    )

    # Now flow recovers (0.3 > 0.2)
    session_manager._check_auto_stop_flow_cutoff(10.0, 0.3)

    mock_hass.async_create_task.assert_not_called()
    assert (
        session_manager._auto_stop_flow_below_cutoff_start_time is None
    )  # Should be reset


# --- Tests for _determine_shot_status_and_duration ---


@pytest.mark.parametrize(
    "stop_reason, session_duration_seconds, min_config_duration, max_config_duration, expected_status, expected_duration_calc",
    [
        # Normal completed shot
        ("ha_service", 25, 10, 45, "completed", 25),
        # Disconnected
        ("disconnected", 15, 10, 45, "aborted_disconnected", 15),
        # Too long
        ("ha_service", 50, 10, 45, "aborted_too_long", 50),
        # Too short
        ("ha_service", 5, 10, 45, "aborted_too_short", 5),
        # Max duration disabled (0)
        ("ha_service", 100, 10, 0, "completed", 100),
        # Forced stop, but still too short (current code makes it aborted_too_short)
        # If logic changes to allow forced short shots as 'completed', this test needs update
        (
            "ha_service_stop_forced",
            5,
            10,
            45,
            "completed",
            5,
        ),  # Expect 'completed' due to forced stop overriding 'too_short'
        # Forced stop, normal duration
        ("ha_service_stop_forced", 25, 10, 45, "completed", 25),
        # Edge case: duration equals min_duration
        ("ha_service", 10, 10, 45, "completed", 10),
        # Edge case: duration equals max_duration (and max_duration > 0)
        ("ha_service", 45, 10, 45, "completed", 45),
    ],
)
def test_determine_shot_status_and_duration(
    session_manager: SessionManager,
    mock_bookoo_config: MagicMock,
    mock_utcnow: MagicMock,
    stop_reason: str,
    session_duration_seconds: int,
    min_config_duration: int,
    max_config_duration: int,
    expected_status: str,
    expected_duration_calc: int,
):
    """Test various scenarios for _determine_shot_status_and_duration."""
    mock_bookoo_config.min_shot_duration = min_config_duration
    mock_bookoo_config.max_shot_duration = max_config_duration

    current_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    mock_utcnow.return_value = current_time  # Ensure current_time is consistent if _determine... uses utcnow directly
    session_start_time = current_time - timedelta(seconds=session_duration_seconds)

    # session_manager.session_start_time_utc would be set by start_session,
    # but this method receives it as an argument 'current_session_start_time_utc', so we pass it directly.

    status, duration = session_manager._determine_shot_status_and_duration(
        stop_reason, session_start_time, current_time
    )

    assert status == expected_status
    assert duration == pytest.approx(expected_duration_calc)


# --- Tests for _calculate_final_shot_metrics ---


@pytest.fixture
def sample_flow_profile() -> collections.deque[FlowDataPoint]:
    """Provides a sample flow profile for testing metrics."""
    return collections.deque(
        [
            FlowDataPoint(elapsed_time=0.0, flow_rate=0.0),
            FlowDataPoint(
                elapsed_time=1.0, flow_rate=0.1
            ),  # Below first flow threshold
            FlowDataPoint(elapsed_time=2.0, flow_rate=0.3),  # First flow
            FlowDataPoint(elapsed_time=3.0, flow_rate=1.5),
            FlowDataPoint(elapsed_time=4.0, flow_rate=2.5),  # Peak flow
            FlowDataPoint(elapsed_time=5.0, flow_rate=1.8),
            FlowDataPoint(elapsed_time=6.0, flow_rate=0.0),  # End of flow
        ],
        maxlen=MAX_PROFILE_POINTS,
    )


def test_calculate_final_shot_metrics_normal_shot(
    session_manager: SessionManager,
    sample_flow_profile: collections.deque[FlowDataPoint],
):
    """Test metrics calculation for a typical shot."""
    duration_seconds = 6.0
    final_weight_grams = 36.0

    metrics = session_manager._calculate_final_shot_metrics(
        duration_seconds, final_weight_grams, sample_flow_profile
    )

    assert metrics["average_flow_rate_gps"] == pytest.approx(36.0 / 6.0)
    assert metrics["peak_flow_rate_gps"] == pytest.approx(2.5)
    assert metrics["time_to_peak_flow_seconds"] == pytest.approx(4.0)
    assert metrics["time_to_first_flow_seconds"] == pytest.approx(2.0)


def test_calculate_final_shot_metrics_no_flow(session_manager: SessionManager):
    """Test metrics calculation when there is no significant flow."""
    duration_seconds = 10.0
    final_weight_grams = 0.5  # Minimal weight, maybe just drips
    flow_profile = collections.deque(
        [
            FlowDataPoint(elapsed_time=0.0, flow_rate=0.0),
            FlowDataPoint(elapsed_time=1.0, flow_rate=0.01),
            FlowDataPoint(elapsed_time=2.0, flow_rate=0.005),
        ],
        maxlen=MAX_PROFILE_POINTS,
    )

    metrics = session_manager._calculate_final_shot_metrics(
        duration_seconds, final_weight_grams, flow_profile
    )

    assert metrics["average_flow_rate_gps"] == pytest.approx(0.5 / 10.0)
    assert metrics["peak_flow_rate_gps"] == 0.0  # No flow above 0.01 threshold for peak
    assert metrics["time_to_peak_flow_seconds"] is None
    assert metrics["time_to_first_flow_seconds"] is None  # No flow above 0.2 threshold


def test_calculate_final_shot_metrics_zero_duration(
    session_manager: SessionManager,
    sample_flow_profile: collections.deque[FlowDataPoint],
):
    """Test metrics calculation with zero duration."""
    duration_seconds = 0.0
    final_weight_grams = 0.0
    # Flow profile might still exist if data was added before stop with 0 duration

    metrics = session_manager._calculate_final_shot_metrics(
        duration_seconds, final_weight_grams, sample_flow_profile
    )

    assert metrics["average_flow_rate_gps"] == 0.0
    # Other metrics depend on profile, should still be calculated if profile is valid
    assert metrics["peak_flow_rate_gps"] == pytest.approx(2.5)
    assert metrics["time_to_peak_flow_seconds"] == pytest.approx(4.0)
    assert metrics["time_to_first_flow_seconds"] == pytest.approx(2.0)


def test_calculate_final_shot_metrics_empty_profile(session_manager: SessionManager):
    """Test metrics calculation with an empty flow profile."""
    duration_seconds = 10.0
    final_weight_grams = 5.0
    flow_profile: collections.deque[FlowDataPoint] = collections.deque(
        maxlen=MAX_PROFILE_POINTS
    )

    metrics = session_manager._calculate_final_shot_metrics(
        duration_seconds, final_weight_grams, flow_profile
    )

    assert metrics["average_flow_rate_gps"] == pytest.approx(5.0 / 10.0)
    assert metrics["peak_flow_rate_gps"] == 0.0
    assert metrics["time_to_peak_flow_seconds"] is None
    assert metrics["time_to_first_flow_seconds"] is None


# --- Tests for _finalize_and_store_shot ---


@pytest.fixture
def valid_raw_event_data(mock_coordinator: MagicMock, mock_utcnow: MagicMock) -> dict:
    """Provides a sample of valid raw event data for testing _finalize_and_store_shot."""
    start_time = mock_utcnow.return_value - timedelta(seconds=30)
    return {
        "device_id": mock_coordinator.config_entry.unique_id,
        "unique_shot_id": f"{start_time.isoformat()}_{mock_coordinator.config_entry.unique_id}",
        "start_time_utc": start_time.isoformat(),
        "end_time_utc": mock_utcnow.return_value.isoformat(),
        "duration_seconds": 30.0,
        "final_weight_grams": 36.0,
        "status": "completed",
        "start_trigger": "manual",
        "stop_reason": "manual_stop",
        "input_parameters": {"grind": "fine"},
        "flow_profile": [FlowDataPoint(elapsed_time=1.0, flow_rate=1.0)],
        "weight_profile": [WeightDataPoint(elapsed_time=1.0, weight=5.0)],
        "scale_timer_profile": [ScaleTimerDataPoint(elapsed_time=1.0, timer_value=1)],
        # Analytics fields that would be populated
        "channeling_status": "None",
        "pre_infusion_detected": False,
        "pre_infusion_duration_seconds": None,
        "extraction_uniformity_metric": 0.8,
        "average_flow_rate_gps": 1.2,
        "peak_flow_rate_gps": 1.5,
        "time_to_first_flow_seconds": 5.0,
        "time_to_peak_flow_seconds": 15.0,
        "shot_quality_score": 90.0,
    }


@pytest.mark.asyncio
async def test_finalize_and_store_shot_success(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_coordinator: MagicMock,
    mock_async_add_shot_record: AsyncMock,
    valid_raw_event_data: dict,
    caplog: pytest.LogCaptureFixture,
):
    """Test successful validation, storage, and event firing."""
    shot_status = "completed"
    expected_event_name = f"{mock_coordinator.config_entry.domain}_shot_completed"

    await session_manager._finalize_and_store_shot(valid_raw_event_data, shot_status)

    assert isinstance(session_manager.last_shot_data, BookooShotCompletedEventDataModel)
    mock_async_add_shot_record.assert_called_once_with(
        mock_hass, session_manager.last_shot_data
    )
    mock_hass.bus.async_fire.assert_called_once_with(
        expected_event_name, session_manager.last_shot_data.model_dump(mode="json")
    )
    assert f"Shot (status: {shot_status}) data stored successfully." in caplog.text
    assert f"Fired {expected_event_name} event" in caplog.text


@pytest.mark.asyncio
async def test_finalize_and_store_shot_aborted_too_short(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_coordinator: MagicMock,
    mock_async_add_shot_record: AsyncMock,
    valid_raw_event_data: dict,
    caplog: pytest.LogCaptureFixture,
):
    """Test behavior when shot_status is 'aborted_too_short'."""
    shot_status = "aborted_too_short"
    valid_raw_event_data["status"] = shot_status  # Update status in data
    expected_event_name = f"{mock_coordinator.config_entry.domain}_shot_completed"

    await session_manager._finalize_and_store_shot(valid_raw_event_data, shot_status)

    assert isinstance(session_manager.last_shot_data, BookooShotCompletedEventDataModel)
    mock_async_add_shot_record.assert_not_called()
    mock_hass.bus.async_fire.assert_called_once_with(
        expected_event_name, session_manager.last_shot_data.model_dump(mode="json")
    )
    assert (
        f"Shot was aborted too short (status: {shot_status}), not saving to history."
        in caplog.text
    )
    assert f"Fired {expected_event_name} event" in caplog.text


@pytest.mark.asyncio
async def test_finalize_and_store_shot_storage_failure(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_coordinator: MagicMock,
    mock_async_add_shot_record: AsyncMock,
    valid_raw_event_data: dict,
    caplog: pytest.LogCaptureFixture,
):
    """Test behavior when async_add_shot_record raises an exception."""
    shot_status = "completed"
    mock_async_add_shot_record.side_effect = Exception("DB write error")

    await session_manager._finalize_and_store_shot(valid_raw_event_data, shot_status)

    assert isinstance(session_manager.last_shot_data, BookooShotCompletedEventDataModel)
    mock_async_add_shot_record.assert_called_once()
    mock_hass.bus.async_fire.assert_not_called()  # Event should not be fired on storage failure
    assert (
        f"Failed to store shot record (status: {shot_status}): DB write error"
        in caplog.text
    )
    assert "Shot event NOT fired due to storage failure" in caplog.text


@pytest.mark.asyncio
async def test_finalize_and_store_shot_validation_error(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_async_add_shot_record: AsyncMock,
    valid_raw_event_data: dict,
    caplog: pytest.LogCaptureFixture,
):
    """Test behavior when BookooShotCompletedEventDataModel validation fails."""
    shot_status = "completed"
    invalid_data = valid_raw_event_data.copy()
    del invalid_data["device_id"]  # Remove a required field

    await session_manager._finalize_and_store_shot(invalid_data, shot_status)

    assert (
        session_manager.last_shot_data is None
    )  # Should not be set on validation error
    mock_async_add_shot_record.assert_not_called()
    mock_hass.bus.async_fire.assert_not_called()
    assert "Validation error preparing shot data" in caplog.text


@pytest.mark.asyncio
async def test_finalize_and_store_shot_unexpected_error(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_async_add_shot_record: AsyncMock,
    valid_raw_event_data: dict,
    caplog: pytest.LogCaptureFixture,
):
    """Test behavior with an unexpected error during finalization (e.g., model_dump)."""
    shot_status = "completed"

    # To simulate error during model_dump, we can patch the model instance if it's created
    # Or, more simply, cause an error after validation but before event firing/storage if possible.
    # Let's assume the Pydantic model itself is fine, but something else goes wrong.
    # For instance, if hass.bus.async_fire was not a mock and had an issue.
    # Here, we'll mock a part of the process after validation to fail.

    # This test is a bit tricky to set up perfectly without complex patching of Pydantic internals.
    # A simpler approach: if async_add_shot_record is fine, but async_fire fails.
    # However, the current logic fires event only if storage is successful.
    # Let's simulate an error after successful validation but before storage is attempted, if possible.
    # The current structure makes this hard. Let's assume an error in logging or similar.

    # For this example, let's assume a non-critical error occurs after validation.
    # The primary goal is to see if the broad `except Exception` is hit.
    # We can achieve this by making `last_shot_data.model_dump()` fail after it's set.

    with patch.object(
        BookooShotCompletedEventDataModel,
        "model_dump",
        side_effect=Exception("Dump error"),
    ):
        # This patch will affect the model_dump call within _finalize_and_store_shot
        # when it tries to log the validated_shot_data or prepare it for the event.
        await session_manager._finalize_and_store_shot(
            valid_raw_event_data, shot_status
        )

    # `last_shot_data` might be set if validation passed before dump error
    # assert session_manager.last_shot_data is not None
    mock_async_add_shot_record.assert_not_called()  # Because model_dump for logging/event would fail first
    mock_hass.bus.async_fire.assert_not_called()
    assert "Unexpected error finalizing shot: Dump error" in caplog.text


# --- Tests for stop_session ---


@pytest.mark.asyncio
async def test_stop_session_successful_normal_shot(
    session_manager: SessionManager,
    mock_hass: MagicMock,
    mock_coordinator: MagicMock,
    mock_utcnow: MagicMock,
    mock_async_add_shot_record: AsyncMock,  # Used by _finalize_and_store_shot
):
    """Test stop_session for a normally completed shot."""
    # Start a session first
    session_manager.is_shot_active = True
    session_manager.session_start_time_utc = mock_utcnow.return_value - timedelta(
        seconds=30
    )
    session_manager.session_start_trigger = "manual_test"
    session_manager.session_input_parameters = {"test_param": "test_value"}
    # Add some dummy profile data
    session_manager.session_flow_profile.append(FlowDataPoint(1.0, 1.0))
    session_manager.session_weight_profile.append(WeightDataPoint(1.0, 5.0))
    session_manager.session_scale_timer_profile.append(ScaleTimerDataPoint(1.0, 1))

    # Mock helper methods called by stop_session
    mock_determine_status = MagicMock(return_value=("completed", 30.0))
    mock_calc_analytics = MagicMock(return_value={"analytics_key": "analytics_value"})
    mock_calc_metrics = MagicMock(return_value={"metrics_key": "metrics_value"})
    mock_finalize_store = AsyncMock()
    mock_reset_state = MagicMock()

    with patch.object(
        session_manager, "_determine_shot_status_and_duration", mock_determine_status
    ):
        with patch.object(
            session_manager, "_calculate_shot_analytics", mock_calc_analytics
        ):
            with patch.object(
                session_manager, "_calculate_final_shot_metrics", mock_calc_metrics
            ):
                with patch.object(
                    session_manager, "_finalize_and_store_shot", mock_finalize_store
                ):
                    with patch.object(
                        session_manager,
                        "_reset_internal_session_state",
                        mock_reset_state,
                    ):
                        await session_manager.stop_session(
                            stop_reason="ha_service_test"
                        )

    mock_determine_status.assert_called_once_with(
        "ha_service_test",
        session_manager.session_start_time_utc,
        mock_utcnow.return_value,
    )
    mock_calc_analytics.assert_called_once()
    mock_calc_metrics.assert_called_once_with(
        30.0, mock_coordinator.scale.weight, session_manager.session_flow_profile
    )

    # Check arguments passed to _finalize_and_store_shot
    finalize_call_args = mock_finalize_store.call_args[0]
    raw_event_data_arg = finalize_call_args[0]
    shot_status_arg = finalize_call_args[1]

    assert shot_status_arg == "completed"
    assert raw_event_data_arg["status"] == "completed"
    assert raw_event_data_arg["duration_seconds"] == 30.0
    assert raw_event_data_arg["stop_reason"] == "ha_service_test"
    assert raw_event_data_arg["start_trigger"] == "manual_test"
    assert raw_event_data_arg["input_parameters"] == {"test_param": "test_value"}
    assert raw_event_data_arg["analytics_key"] == "analytics_value"
    assert raw_event_data_arg["metrics_key"] == "metrics_value"
    assert len(raw_event_data_arg["flow_profile"]) == 1

    mock_reset_state.assert_called_once()
    mock_coordinator.async_update_listeners.assert_called_once()


@pytest.mark.asyncio
async def test_stop_session_no_active_session(
    session_manager: SessionManager, caplog: pytest.LogCaptureFixture
):
    """Test stop_session when no session is active."""
    session_manager.is_shot_active = False
    session_manager.session_start_time_utc = None

    # Spy on methods that should NOT be called
    with patch.object(
        session_manager, "_determine_shot_status_and_duration"
    ) as mock_determine:
        with patch.object(session_manager, "_finalize_and_store_shot") as mock_finalize:
            with patch.object(
                session_manager, "_reset_internal_session_state"
            ) as mock_reset:
                await session_manager.stop_session(stop_reason="irrelevant")

        mock_determine.assert_not_called()
        mock_finalize.assert_not_called()
        mock_reset.assert_not_called()  # Reset is only called after a successful stop processing
        assert (
            "Stop session called but no active session or start time found."
            in caplog.text
        )


@pytest.mark.asyncio
async def test_stop_session_aborted_too_short(
    session_manager: SessionManager,
    mock_coordinator: MagicMock,
    mock_utcnow: MagicMock,
    mock_async_add_shot_record: AsyncMock,  # Used by _finalize_and_store_shot
):
    """Test stop_session for a shot determined to be 'aborted_too_short'."""
    session_manager.is_shot_active = True
    session_manager.session_start_time_utc = mock_utcnow.return_value - timedelta(
        seconds=5
    )
    session_manager.session_start_trigger = "short_test"

    # Mock _determine_shot_status_and_duration to return 'aborted_too_short'
    mock_determine_status = MagicMock(return_value=("aborted_too_short", 5.0))
    mock_calc_analytics = MagicMock()
    mock_calc_metrics = MagicMock()
    mock_finalize_store = (
        AsyncMock()
    )  # Mock the method that handles storage and event firing
    mock_reset_state = MagicMock()

    with patch.object(
        session_manager, "_determine_shot_status_and_duration", mock_determine_status
    ):
        with patch.object(
            session_manager, "_calculate_shot_analytics", mock_calc_analytics
        ):
            with patch.object(
                session_manager, "_calculate_final_shot_metrics", mock_calc_metrics
            ):
                with patch.object(
                    session_manager, "_finalize_and_store_shot", mock_finalize_store
                ):
                    with patch.object(
                        session_manager,
                        "_reset_internal_session_state",
                        mock_reset_state,
                    ):
                        await session_manager.stop_session(
                            stop_reason="test_abort_reason"
                        )

    mock_determine_status.assert_called_once_with(
        "test_abort_reason",
        session_manager.session_start_time_utc,
        mock_utcnow.return_value,
    )
    mock_calc_analytics.assert_not_called()  # Not called for aborted_too_short
    mock_calc_metrics.assert_not_called()  # Not called for aborted_too_short

    # Assert _finalize_and_store_shot was called correctly
    mock_finalize_store.assert_called_once()
    finalize_call_args = mock_finalize_store.call_args[0]
    raw_event_data_arg = finalize_call_args[0]
    shot_status_arg = finalize_call_args[1]

    assert shot_status_arg == "aborted_too_short"
    assert raw_event_data_arg["status"] == "aborted_too_short"
    assert raw_event_data_arg["duration_seconds"] == 5.0
    assert raw_event_data_arg["stop_reason"] == "test_abort_reason"
    assert raw_event_data_arg["start_trigger"] == "short_test"
    # Ensure analytics and metrics keys are NOT in raw_event_data for aborted_too_short
    assert "analytics_key" not in raw_event_data_arg
    assert "metrics_key" not in raw_event_data_arg
    assert (
        "flow_profile" not in raw_event_data_arg
    )  # Profiles not added for aborted_too_short

    mock_reset_state.assert_called_once()
    mock_coordinator.async_update_listeners.assert_called_once()
    # We will test _finalize_and_store_shot's actual behavior here, not mock it fully
    # but we need to ensure async_add_shot_record is available for it.

    with patch.object(
        session_manager, "_determine_shot_status_and_duration", mock_determine_status
    ):
        with patch.object(
            session_manager, "_calculate_shot_analytics", mock_calc_analytics
        ):
            with patch.object(
                session_manager, "_calculate_final_shot_metrics", mock_calc_metrics
            ):
                with patch.object(
                    session_manager.hass.bus, "async_fire"
                ) as mock_event_fire:  # Spy on event firing
                    await session_manager.stop_session(stop_reason="too_short_reason")

    mock_determine_status.assert_called_once()
    mock_calc_analytics.assert_not_called()  # Should not be called for aborted_too_short
    mock_calc_metrics.assert_not_called()  # Should not be called for aborted_too_short

    # _finalize_and_store_shot IS called. Check its effects:
    # 1. async_add_shot_record should NOT be called for aborted_too_short
    mock_async_add_shot_record.assert_not_called()
    # 2. Event should be fired
    mock_event_fire.assert_called_once()
    fired_event_data = mock_event_fire.call_args[0][1]
    assert fired_event_data["status"] == "aborted_too_short"
    assert fired_event_data["duration_seconds"] == 5.0
    assert fired_event_data["stop_reason"] == "too_short_reason"
    # Ensure profiles are empty or minimal as analytics/metrics weren't run to populate them in raw_event_data
    assert (
        "flow_profile" not in fired_event_data
    )  # Or it's empty, depending on exact _finalize logic for minimal

    assert session_manager.is_shot_active is False  # Reset should have occurred
    mock_coordinator.async_update_listeners.assert_called_once()
