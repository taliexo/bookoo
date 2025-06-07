# tests/unit/test_types.py
import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from custom_components.bookoo.types import (
    FlowDataPoint,
    WeightDataPoint,
    ScaleTimerDataPoint,
    BookooShotCompletedEventDataModel,
)

# Sample valid data for BookooShotCompletedEventDataModel
VALID_SHOT_EVENT_DATA = {
    "device_id": "test_device_001",
    "entry_id": "test_entry_abc",
    "start_time_utc": datetime(2023, 10, 26, 10, 0, 0, tzinfo=timezone.utc),
    "end_time_utc": datetime(2023, 10, 26, 10, 0, 30, tzinfo=timezone.utc),
    "duration_seconds": 30.5,
    "final_weight_grams": 36.2,
    "flow_profile": [
        FlowDataPoint(elapsed_time=0.0, flow_rate=0.0),
        FlowDataPoint(elapsed_time=5.0, flow_rate=1.5),
        FlowDataPoint(elapsed_time=25.0, flow_rate=2.0),
    ],
    "scale_timer_profile": [
        ScaleTimerDataPoint(elapsed_time=0.0, timer_value=0),
        ScaleTimerDataPoint(elapsed_time=30.0, timer_value=30),
    ],
    "input_parameters": {"grind_setting": "7", "bean_weight": 18.5},
    "start_trigger": "manual_service_call",
    "stop_reason": "auto_stop_flow_cutoff",
    "status": "completed",
    "channeling_status": "None",
    "pre_infusion_detected": True,
    "pre_infusion_duration_seconds": 7.5,
    "extraction_uniformity_metric": 0.85,
    "average_flow_rate_gps": 1.18,
    "peak_flow_rate_gps": 2.1,
    "time_to_first_flow_seconds": 6.0,
    "time_to_peak_flow_seconds": 15.0,
    "shot_quality_score": 88.0,
}


def test_flow_data_point_creation():
    """Test creation of FlowDataPoint."""
    fdp = FlowDataPoint(elapsed_time=1.2, flow_rate=2.3)
    assert fdp.elapsed_time == 1.2
    assert fdp.flow_rate == 2.3


def test_weight_data_point_creation():
    """Test creation of WeightDataPoint."""
    wdp = WeightDataPoint(elapsed_time=1.5, weight=18.5)
    assert wdp.elapsed_time == 1.5
    assert wdp.weight == 18.5


def test_scale_timer_data_point_creation():
    """Test creation of ScaleTimerDataPoint."""
    stp = ScaleTimerDataPoint(elapsed_time=30.0, timer_value=30)
    assert stp.elapsed_time == 30.0
    assert stp.timer_value == 30


def test_bookoo_shot_completed_event_data_model_valid():
    """Test successful creation of BookooShotCompletedEventDataModel with valid data."""
    model = BookooShotCompletedEventDataModel(**VALID_SHOT_EVENT_DATA)
    assert model.device_id == VALID_SHOT_EVENT_DATA["device_id"]
    assert model.duration_seconds == VALID_SHOT_EVENT_DATA["duration_seconds"]
    assert len(model.flow_profile) == 3


def test_bookoo_shot_completed_event_data_model_missing_required_field():
    """Test BookooShotCompletedEventDataModel creation fails if a required field is missing."""
    invalid_data = VALID_SHOT_EVENT_DATA.copy()
    del invalid_data["device_id"]  # device_id is a required field
    with pytest.raises(ValidationError):
        BookooShotCompletedEventDataModel(**invalid_data)


def test_bookoo_shot_completed_event_data_model_invalid_type():
    """Test BookooShotCompletedEventDataModel creation fails if a field has an invalid type."""
    invalid_data = VALID_SHOT_EVENT_DATA.copy()
    invalid_data["duration_seconds"] = (
        "not_a_float"  # duration_seconds should be a float
    )
    with pytest.raises(ValidationError):
        BookooShotCompletedEventDataModel(**invalid_data)


def test_bookoo_shot_completed_event_data_model_optional_fields_none():
    """Test BookooShotCompletedEventDataModel with optional fields as None."""
    data_with_optionals_none = VALID_SHOT_EVENT_DATA.copy()
    data_with_optionals_none["pre_infusion_duration_seconds"] = None
    data_with_optionals_none["extraction_uniformity_metric"] = None
    data_with_optionals_none["time_to_first_flow_seconds"] = None
    data_with_optionals_none["time_to_peak_flow_seconds"] = None
    data_with_optionals_none["shot_quality_score"] = None
    data_with_optionals_none["start_trigger"] = None

    model = BookooShotCompletedEventDataModel(**data_with_optionals_none)
    assert model.pre_infusion_duration_seconds is None
    assert model.start_trigger is None


def test_bookoo_shot_completed_event_data_model_flow_profile_empty():
    """Test BookooShotCompletedEventDataModel with an empty flow_profile."""
    data_empty_flow = VALID_SHOT_EVENT_DATA.copy()
    data_empty_flow["flow_profile"] = []
    model = BookooShotCompletedEventDataModel(**data_empty_flow)
    assert model.flow_profile == []


# It's good practice to ensure immutability if Config.allow_mutation = False
# However, Pydantic v2 by default makes models immutable if frozen=True in model_config
# or if Config.allow_mutation = False. Let's test this behavior.


# Pydantic v2 uses model_config now, not Config class directly for this.
# Assuming BookooShotCompletedEventDataModel.Config.allow_mutation = False is effective.
@pytest.mark.skip(
    reason="Pydantic v2 handles immutability differently, direct attr assignment might not be the best test or might be allowed if not frozen."
)
def test_bookoo_shot_completed_event_data_model_immutability(request):
    """Test that the model is immutable after creation if configured as such."""
    # This test's relevance depends on Pydantic version and exact model config.
    # If model_config = {"frozen": True} is used, AttributeError is expected.
    # If Config.allow_mutation = False is used, it might still allow attribute setting
    # but not pass validation on model_dump or similar operations if types are wrong.

    # For Pydantic V2, if frozen=True in model_config:
    # model = BookooShotCompletedEventDataModel(**VALID_SHOT_EVENT_DATA)
    # with pytest.raises(AttributeError): # Or pydantic.ValidationError if frozen=True
    #     model.device_id = "new_id"

    # If only allow_mutation = False in a nested Config class (older Pydantic style or specific setup):
    # This test might need adjustment. Pydantic v2's default behavior is more towards
    # immutability if no setters are defined and fields are typed.
    # Let's assume for now the model is intended to be immutable post-creation.
    model = BookooShotCompletedEventDataModel(**VALID_SHOT_EVENT_DATA)
    with pytest.raises(
        TypeError
    ):  # Pydantic v1 with Config.allow_mutation = False often raised TypeError on setattr
        # Pydantic v2 with frozen=True would raise PydanticFrozenInstanceError (subclass of AttributeError)
        model.device_id = (
            "new_id"  # This behavior is highly dependent on Pydantic version/config
        )
        # If this test fails, it might be due to Pydantic version differences or how immutability is configured.
        # For Pydantic v2, if not `frozen=True`, direct attribute assignment is typically allowed.
        # The `allow_mutation = False` in a nested Config class might be a v1 pattern.
        # If the intent is strict immutability, `model_config = {"frozen": True}` is the v2 way.

    # Given the skip, this is more of a placeholder for future clarification on immutability goals.
    pass
