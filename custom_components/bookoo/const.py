"""Constants for component."""

from homeassistant.const import Platform
from dataclasses import dataclass
from typing import (
    Optional,
)  # Required for ConfigEntry type hint if used directly, but we'll use TYPE_CHECKING
from homeassistant.config_entries import ConfigEntry

DOMAIN = "bookoo"
CONF_IS_VALID_SCALE = "is_valid_scale"

# Event names
EVENT_BOOKOO_SHOT_COMPLETED = f"{DOMAIN}_shot_completed"

# Service names
SERVICE_START_SHOT = "start_shot"
SERVICE_STOP_SHOT = "stop_shot"

# Option Keys
OPTION_MIN_SHOT_DURATION = "minimum_shot_duration_seconds"
OPTION_LINKED_BEAN_WEIGHT_ENTITY = "linked_bean_weight_entity"
OPTION_LINKED_COFFEE_NAME_ENTITY = "linked_coffee_name_entity"

# Auto-Stop Feature Option Keys
OPTION_ENABLE_AUTO_STOP_FLOW_CUTOFF = "enable_auto_stop_flow_cutoff"
OPTION_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION = (
    "auto_stop_pre_infusion_ignore_duration_seconds"
)
OPTION_AUTO_STOP_MIN_FLOW_FOR_STABILITY = "auto_stop_min_flow_for_stability_gps"
OPTION_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY = (
    "auto_stop_max_flow_variance_for_stability_percent"
)
OPTION_AUTO_STOP_MIN_DURATION_FOR_STABILITY = (
    "auto_stop_min_duration_for_stability_seconds"
)
OPTION_AUTO_STOP_FLOW_CUTOFF_THRESHOLD = "auto_stop_flow_cutoff_threshold_gps"
OPTION_AUTO_STOP_MIN_DURATION_FOR_CUTOFF = "auto_stop_min_duration_for_cutoff_seconds"


# Default values for auto-stop, if not configured
DEFAULT_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION = 5.0  # seconds
DEFAULT_AUTO_STOP_MIN_FLOW_FOR_STABILITY = 0.5  # g/s
DEFAULT_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY = 20.0  # percent (e.g., 0.20 for 20%)
DEFAULT_AUTO_STOP_MIN_DURATION_FOR_STABILITY = 3.0  # seconds
DEFAULT_AUTO_STOP_FLOW_CUTOFF_THRESHOLD = 0.2  # g/s
DEFAULT_AUTO_STOP_MIN_DURATION_FOR_CUTOFF = 1.0  # seconds


@dataclass(frozen=True)
class BookooConfig:
    """Typed configuration for the Bookoo integration."""

    min_shot_duration: int
    linked_bean_weight_entity: Optional[str]
    linked_coffee_name_entity: Optional[str]

    enable_auto_stop_flow_cutoff: bool
    auto_stop_pre_infusion_ignore_duration: float
    auto_stop_min_flow_for_stability: float
    auto_stop_max_flow_variance_for_stability: (
        float  # Stored as percent, e.g., 20.0 for 20%
    )
    auto_stop_min_duration_for_stability: float
    auto_stop_flow_cutoff_threshold: float
    auto_stop_min_duration_for_cutoff: float

    @classmethod
    def from_config_entry(cls, entry: ConfigEntry) -> "BookooConfig":
        """Create a BookooConfig instance from a ConfigEntry."""
        options = entry.options
        return cls(
            min_shot_duration=options.get(OPTION_MIN_SHOT_DURATION, 10),
            linked_bean_weight_entity=options.get(OPTION_LINKED_BEAN_WEIGHT_ENTITY),
            linked_coffee_name_entity=options.get(OPTION_LINKED_COFFEE_NAME_ENTITY),
            enable_auto_stop_flow_cutoff=options.get(
                OPTION_ENABLE_AUTO_STOP_FLOW_CUTOFF, False
            ),
            auto_stop_pre_infusion_ignore_duration=options.get(
                OPTION_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
                DEFAULT_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION,
            ),
            auto_stop_min_flow_for_stability=options.get(
                OPTION_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
                DEFAULT_AUTO_STOP_MIN_FLOW_FOR_STABILITY,
            ),
            auto_stop_max_flow_variance_for_stability=options.get(
                OPTION_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
                DEFAULT_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY,
            ),
            auto_stop_min_duration_for_stability=options.get(
                OPTION_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
                DEFAULT_AUTO_STOP_MIN_DURATION_FOR_STABILITY,
            ),
            auto_stop_flow_cutoff_threshold=options.get(
                OPTION_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
                DEFAULT_AUTO_STOP_FLOW_CUTOFF_THRESHOLD,
            ),
            auto_stop_min_duration_for_cutoff=options.get(
                OPTION_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
                DEFAULT_AUTO_STOP_MIN_DURATION_FOR_CUTOFF,
            ),
        )


PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.BUTTON,  # Buttons for start/stop shot services
]
