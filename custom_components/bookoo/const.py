"""Constants for component."""

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
