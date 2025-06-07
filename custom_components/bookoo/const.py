"""Constants for component."""

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform

DOMAIN = "bookoo"
CONF_IS_VALID_SCALE = "is_valid_scale"

# Event names
EVENT_BOOKOO_SHOT_COMPLETED = f"{DOMAIN}_shot_completed"

# Service names
SERVICE_START_SHOT = "start_shot"
SERVICE_STOP_SHOT = "stop_shot"
SERVICE_CONNECT_SCALE = "connect_scale"
SERVICE_DISCONNECT_SCALE = "disconnect_scale"

# Option Keys
OPTION_MIN_SHOT_DURATION = "minimum_shot_duration_seconds"
OPTION_MAX_SHOT_DURATION = "maximum_shot_duration_seconds"
OPTION_LINKED_BEAN_WEIGHT_ENTITY = "linked_bean_weight_entity"
OPTION_LINKED_COFFEE_NAME_ENTITY = "linked_coffee_name_entity"
OPTION_LINKED_GRIND_SETTING_ENTITY = "linked_grind_setting_entity"
OPTION_LINKED_BREW_TEMPERATURE_ENTITY = "linked_brew_temperature_entity"

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

# Bluetooth Timeout Option Keys
OPTION_CONNECT_TIMEOUT = "bluetooth_connect_timeout_seconds"
OPTION_COMMAND_TIMEOUT = "bluetooth_command_timeout_seconds"


# Default values for auto-stop, if not configured
DEFAULT_AUTO_STOP_PRE_INFUSION_IGNORE_DURATION = 5.0  # seconds
DEFAULT_AUTO_STOP_MIN_FLOW_FOR_STABILITY = 0.5  # g/s
DEFAULT_AUTO_STOP_MAX_FLOW_VARIANCE_FOR_STABILITY = 20.0  # percent (e.g., 0.20 for 20%)
DEFAULT_AUTO_STOP_MIN_DURATION_FOR_STABILITY = 3.0  # seconds
DEFAULT_AUTO_STOP_FLOW_CUTOFF_THRESHOLD = 0.2  # g/s
DEFAULT_AUTO_STOP_MIN_DURATION_FOR_CUTOFF = 1.0  # seconds

# Default values for Bluetooth timeouts
DEFAULT_MAX_SHOT_DURATION = 45.0  # seconds
DEFAULT_CONNECT_TIMEOUT = 15.0  # seconds
DEFAULT_COMMAND_TIMEOUT = 10.0  # seconds


@dataclass(frozen=True)
class BookooConfig:
    """Typed configuration for the Bookoo integration."""

    min_shot_duration: int
    max_shot_duration: int
    linked_bean_weight_entity: str | None
    linked_coffee_name_entity: str | None
    linked_grind_setting_entity: str | None
    linked_brew_temperature_entity: str | None

    enable_auto_stop_flow_cutoff: bool
    auto_stop_pre_infusion_ignore_duration: float
    auto_stop_min_flow_for_stability: float
    auto_stop_max_flow_variance_for_stability: (
        float  # Stored as percent, e.g., 20.0 for 20%
    )
    auto_stop_min_duration_for_stability: float
    auto_stop_flow_cutoff_threshold: float
    auto_stop_min_duration_for_cutoff: float

    # Bluetooth Timeouts
    connect_timeout: float
    command_timeout: float

    @classmethod
    def from_config_entry(cls, entry: ConfigEntry) -> "BookooConfig":
        """Create a BookooConfig instance from a ConfigEntry."""
        options = entry.options
        return cls(
            min_shot_duration=options.get(OPTION_MIN_SHOT_DURATION, 10),
            max_shot_duration=options.get(
                OPTION_MAX_SHOT_DURATION, DEFAULT_MAX_SHOT_DURATION
            ),
            linked_bean_weight_entity=options.get(OPTION_LINKED_BEAN_WEIGHT_ENTITY),
            linked_coffee_name_entity=options.get(OPTION_LINKED_COFFEE_NAME_ENTITY),
            linked_grind_setting_entity=options.get(OPTION_LINKED_GRIND_SETTING_ENTITY),
            linked_brew_temperature_entity=options.get(
                OPTION_LINKED_BREW_TEMPERATURE_ENTITY
            ),
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
            connect_timeout=options.get(
                OPTION_CONNECT_TIMEOUT,
                DEFAULT_CONNECT_TIMEOUT,
            ),
            command_timeout=options.get(
                OPTION_COMMAND_TIMEOUT,
                DEFAULT_COMMAND_TIMEOUT,
            ),
        )


PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.BUTTON,  # Buttons for start/stop shot services
]
