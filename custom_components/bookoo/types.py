from typing import (
    NamedTuple,
    List,
    Optional,
    Any,
)  # TypedDict removed for this specific type
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


FlowProfile = List[FlowDataPoint]
WeightProfile = List[WeightDataPoint]
ScaleTimerProfile = List[ScaleTimerDataPoint]


class BookooShotCompletedEventDataModel(BaseModel):
    device_id: str
    entry_id: str
    start_time_utc: str  # ISO format, consider datetime for future parsing
    end_time_utc: str  # ISO format, consider datetime for future parsing
    duration_seconds: float
    final_weight_grams: float
    flow_profile: FlowProfile
    scale_timer_profile: ScaleTimerProfile
    input_parameters: dict[str, Any]
    start_trigger: Optional[str] = None
    stop_reason: str
    status: str
    channeling_status: str
    pre_infusion_detected: bool
    pre_infusion_duration_seconds: Optional[float] = None
    extraction_uniformity_metric: Optional[float] = None
    average_flow_rate_gps: float
    peak_flow_rate_gps: float
    time_to_first_flow_seconds: Optional[float] = None
    time_to_peak_flow_seconds: Optional[float] = None
    shot_quality_score: Optional[float] = None

    class Config:
        allow_mutation = False  # Make instances immutable after creation
        # orm_mode = True # If you were creating this from an ORM model
