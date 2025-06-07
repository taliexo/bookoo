import datetime
from typing import (
    Any,
    NamedTuple,
)

from pydantic import BaseModel


class FlowDataPoint(NamedTuple):
    elapsed_time: float
    flow_rate: float  # g/s


class WeightDataPoint(NamedTuple):
    elapsed_time: float
    weight: float  # grams


class ScaleTimerDataPoint(NamedTuple):
    elapsed_time: float
    timer_value: int  # seconds


FlowProfile = list[FlowDataPoint]
WeightProfile = list[WeightDataPoint]
ScaleTimerProfile = list[ScaleTimerDataPoint]


class BookooShotCompletedEventDataModel(BaseModel):
    device_id: str
    entry_id: str
    start_time_utc: datetime.datetime
    end_time_utc: datetime.datetime
    duration_seconds: float
    final_weight_grams: float
    flow_profile: FlowProfile
    scale_timer_profile: ScaleTimerProfile
    input_parameters: dict[str, Any]
    start_trigger: str | None = None
    stop_reason: str
    status: str
    channeling_status: str
    pre_infusion_detected: bool
    pre_infusion_duration_seconds: float | None = None
    extraction_uniformity_metric: float | None = None
    average_flow_rate_gps: float
    peak_flow_rate_gps: float
    time_to_first_flow_seconds: float | None = None
    time_to_peak_flow_seconds: float | None = None
    shot_quality_score: float | None = None
    next_shot_recommendation: str | None = None

    model_config = {
        "frozen": True  # Make instances immutable after creation
    }
