"""Coordinator for Bookoo integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any  # For session_input_parameters

from aiobookoov2.const import UPDATE_SOURCE_COMMAND_CHAR, UPDATE_SOURCE_WEIGHT_CHAR
from aiobookoov2.decode import (
    decode as aiobookoo_decode,
)  # Alias to avoid name clash if local decode exists
import logging

from aiobookoov2.bookooscale import BookooScale
from aiobookoov2.exceptions import BookooDeviceNotFound, BookooError

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ADDRESS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    CONF_IS_VALID_SCALE,
    EVENT_BOOKOO_SHOT_COMPLETED,
    OPTION_MIN_SHOT_DURATION,
    OPTION_LINKED_BEAN_WEIGHT_ENTITY,
    OPTION_LINKED_COFFEE_NAME_ENTITY,
)  # Assuming DOMAIN is needed for event firing context
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE # For reading entity states

SCAN_INTERVAL = timedelta(seconds=5)

_LOGGER = logging.getLogger(__name__)

type BookooConfigEntry = ConfigEntry[BookooCoordinator]


class BookooCoordinator(DataUpdateCoordinator[None]):
    """Class to handle fetching data from the scale."""

    config_entry: BookooConfigEntry

    def __init__(self, hass: HomeAssistant, entry: BookooConfigEntry) -> None:
        """Initialize coordinator."""
        self._scale = BookooScale(
            address_or_ble_device=entry.data[CONF_ADDRESS],
            name=entry.title,
            is_valid_scale=entry.data.get(CONF_IS_VALID_SCALE, False),
            notify_callback=self.async_update_listeners,  # General state update
            characteristic_update_callback=self._handle_characteristic_update,  # Detailed char data
        )

        super().__init__(
            hass,
            _LOGGER,
            name=f"Bookoo {self._scale.mac}",
            update_interval=SCAN_INTERVAL,
            config_entry=entry,
        )
        
        # Initialize other coordinator-specific attributes
        self.is_shot_active: bool = False
        self.session_start_time_utc: datetime | None = None
        self.session_flow_profile: list[tuple[float, float]] = []
        self.session_scale_timer_profile: list[tuple[float, int]] = []
        self.session_input_parameters: dict[str, Any] = {} # Initialized here
        self.session_start_trigger: str | None = None
        self.last_shot_data: dict[str, Any] = {} # Initialized here

        # Load options
        self.min_shot_duration: int = 10 # Default
        self.linked_bean_weight_entity_id: str | None = None
        self.linked_coffee_name_entity_id: str | None = None
        self._load_options() # Load initial options

        # Listener for options updates
        self._options_update_listener = entry.add_update_listener(self._options_update_callback)

    @property
    def scale(self) -> BookooScale:
        """Return the scale object."""
        return self._scale

    def _load_options(self) -> None:
        """Load options from the config entry."""
        self.min_shot_duration = self.config_entry.options.get(OPTION_MIN_SHOT_DURATION, 10)
        self.linked_bean_weight_entity_id = self.config_entry.options.get(OPTION_LINKED_BEAN_WEIGHT_ENTITY)
        self.linked_coffee_name_entity_id = self.config_entry.options.get(OPTION_LINKED_COFFEE_NAME_ENTITY)
        _LOGGER.debug(
            "Loaded options: Min Duration=%s, Bean Weight Entity=%s, Coffee Name Entity=%s",
            self.min_shot_duration,
            self.linked_bean_weight_entity_id,
            self.linked_coffee_name_entity_id
        )

    async def _options_update_callback(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle options update."""
        _LOGGER.debug("Bookoo options updated, reloading.")
        self._load_options()
        # If options affect sensors directly, you might trigger self.async_update_listeners() here

    def _handle_characteristic_update(self, source: str, data: bytes | None) -> None:
        """Handle updates from aiobookoo's characteristic_update_callback."""
        if not data:
            _LOGGER.debug(
                "Received characteristic update with no data from source: %s", source
            )
            return

        is_command_char = source == UPDATE_SOURCE_COMMAND_CHAR
        _LOGGER.debug("[HANDLE_CHAR_DEBUG] Comparison: (source == UPDATE_SOURCE_COMMAND_CHAR) is %s", is_command_char)
        if is_command_char:
            _LOGGER.debug("Command Char Update (raw): %s", data.hex())
            decoded_cmd_data = aiobookoo_decode(data) if data else None
            _LOGGER.debug("[HANDLE_CHAR_DEBUG] decoded_cmd_data: %s (type: %s)", decoded_cmd_data, type(decoded_cmd_data).__name__)

            if isinstance(decoded_cmd_data, dict):
                msg_type = decoded_cmd_data.get("type")
                event = decoded_cmd_data.get("event")

                if msg_type == "auto_timer" and event == "start":
                    if not self.is_shot_active:
                        _LOGGER.info("Scale auto-timer (decoded via type/event) started shot.")
                        self.hass.async_create_task(
                            self._start_session(trigger="scale_auto_decoded")
                        )
                    else:
                        _LOGGER.debug(
                            "Scale auto-timer start event (decoded via type/event) received, but shot already active."
                        )
                elif msg_type == "auto_timer" and event == "stop":
                    if self.is_shot_active:
                        _LOGGER.info("Scale auto-timer (decoded via type/event) stopped shot.")
                        self.hass.async_create_task(
                            self._stop_session(stop_reason="scale_auto_stop_decoded")
                        )
                    _LOGGER.error(
                        "Error processing updated scale properties during active shot: %s", e, exc_info=True
                    )
            
            # Always update listeners for weight changes, so all sensors (weight, flow, timer, battery) get refreshed
            self.async_update_listeners()
            return

        elif source == UPDATE_SOURCE_COMMAND_CHAR:
            # For command characteristic, 'data' should be a pre-decoded dict for known commands (like auto-timer)
            # from BookooScale, or raw bytes if BookooScale couldn't decode it into a dict.
            
            if isinstance(data, dict): # Pre-decoded by BookooScale
                event_type = data.get("type")
                event_action = data.get("event")
                
                if event_type == "auto_timer":
                    if event_action == "start" and not self.is_shot_active:
                        _LOGGER.info("Auto-timer start event detected by scale (decoded by library).")
                        self._start_session(trigger="scale_auto_timer")
                    elif event_action == "stop" and self.is_shot_active:
                        _LOGGER.info("Auto-timer stop event detected by scale (decoded by library).")
                        self._stop_session(stop_reason="scale_auto_timer")
                    else:
                        _LOGGER.debug("Auto-timer event (%s) received but state condition not met (active: %s).", event_action, self.is_shot_active)
                else:
                    _LOGGER.debug("Received known dict from command char, but not 'auto_timer' type: %s", data)
            
            elif isinstance(data, bytes): # Raw bytes, BookooScale didn't decode to dict
                _LOGGER.debug("Received raw bytes from command char: %s. Attempting coordinator-level parse for legacy auto-timer.", data.hex())
                # Legacy raw byte parsing for auto-timer as a fallback or for other commands
                if len(data) >= 3 and data[0] == 0x03 and data[1] == 0x0D: # Check for 0x030D auto-timer prefix
                    if data[2] == 0x01: # CMD_BYTE3_AUTO_TIMER_EVENT_START
                        if not self.is_shot_active:
                            _LOGGER.info("Auto-timer start event detected by scale (parsed raw by coordinator).")
                            self._start_session(trigger="scale_auto_timer_raw")
                        else:
                            _LOGGER.debug("Scale auto-timer start event (raw) received, but shot already active.")
                    elif data[2] == 0x00: # CMD_BYTE3_AUTO_TIMER_EVENT_STOP
                        if self.is_shot_active:
                            _LOGGER.info("Scale auto-timer stop event detected by scale (parsed raw by coordinator).")
                            self._stop_session(stop_reason="scale_auto_timer_raw")
                        else:
                            _LOGGER.debug("Scale auto-timer stop event (raw) received, but no shot active.")
                    else:
                        _LOGGER.debug("Known auto-timer prefix (0x030D) but unknown event byte %02x.", data[2])
                else:
                    _LOGGER.debug("Raw command char data not recognized as 0x030D auto-timer: %s", data.hex())
            
            else:
                _LOGGER.debug("Received unexpected data type from command char: %s (%s)", data, type(data).__name__)
            
            self.async_update_listeners() # Update listeners after processing any command char data
            return

        # For unknown_char_update or if logic falls through without returning
        _LOGGER.debug("Unhandled characteristic update source or data type: %s", source)
        self.async_update_listeners()

    async def _async_update_data(self) -> None:
        """Fetch data."""

        # scale is already connected, return
        if self._scale.connected:
            return

        # scale is not connected, try to connect
        try:
            await self._scale.connect(setup_tasks=False)
        except (BookooDeviceNotFound, BookooError, TimeoutError) as ex:
            _LOGGER.debug(
                "Could not connect to scale: %s, Error: %s",
                self.config_entry.data[CONF_ADDRESS],
                ex,
            )
            self._scale.device_disconnected_handler(notify=False)
            if self.is_shot_active:
                _LOGGER.warning(
                    "Scale disconnected during an active shot. Ending session."
                )
                self.hass.async_create_task(
                    self._stop_session(stop_reason="disconnected")
                )
            return

        # connected, set up background tasks

        if not self._scale.process_queue_task or self._scale.process_queue_task.done():
            self._scale.process_queue_task = (
                self.config_entry.async_create_background_task(
                    hass=self.hass,
                    target=self._scale.process_queue(),
                    name="bookoo_process_queue_task",
                )
            )

    # Service call handlers
    async def async_start_shot_service(self) -> None:
        """Service call to start a new shot session via HA."""
        if self.is_shot_active:
            _LOGGER.warning("Start shot service called, but a shot is already active.")
            return
        _LOGGER.info("HA service starting shot.")
        await self._start_session(trigger="ha_service")
        try:
            # Assuming aiobookoo has an async_tare_and_start_timer method
            await self.scale.async_send_command("tareAndStartTime")
            _LOGGER.debug("Sent Tare & Start Timer command to scale.")
        except BookooError as e:
            _LOGGER.error("Error sending Tare & Start Timer command to scale: %s", e)
            # Optionally, stop the session if command fails, or let it run without scale sync
            # await self._stop_session(stop_reason="command_failed_start")

    async def async_stop_shot_service(self) -> None:
        """Service call to stop the current shot session via HA."""
        if not self.is_shot_active:
            _LOGGER.warning("Stop shot service called, but no shot is active.")
            return
        _LOGGER.info("HA service stopping shot.")
        # Scale command is sent first, then session is stopped locally.
        # This ensures that if the command fails, the HA session still stops.
        try:
            # Assuming aiobookoo has an async_stop_timer method
            await self.scale.async_send_command("stopTimer")
            _LOGGER.debug("Sent Stop Timer command to scale.")
        except BookooError as e:
            _LOGGER.error("Error sending Stop Timer command to scale: %s", e)

        await self._stop_session(stop_reason="ha_service")

    async def _start_session(self, trigger: str) -> None:
        """Internal method to start a new shot session."""
        if self.is_shot_active:
            _LOGGER.warning(
                "Attempted to start a new shot session (trigger: %s) but one is already active.", trigger
            )
            return

        _LOGGER.info("Starting new shot session, triggered by: %s", trigger)
        self.is_shot_active = True
        self.session_start_time_utc = dt_util.utcnow()
        self.session_flow_profile = []
        self.session_scale_timer_profile = []
        self.session_start_trigger = trigger
        self.session_input_parameters = {} # Clear/initialize for the new session

        # Read linked input_number/input_text entities
        if self.linked_bean_weight_entity_id:
            bean_weight_state = self.hass.states.get(self.linked_bean_weight_entity_id)
            if bean_weight_state and bean_weight_state.state not in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                self.session_input_parameters["bean_weight"] = bean_weight_state.state
                _LOGGER.debug("Logged bean_weight: %s from %s", bean_weight_state.state, self.linked_bean_weight_entity_id)
            else:
                _LOGGER.warning("Could not read state for linked bean weight entity: %s", self.linked_bean_weight_entity_id)
        
        if self.linked_coffee_name_entity_id:
            coffee_name_state = self.hass.states.get(self.linked_coffee_name_entity_id)
            if coffee_name_state and coffee_name_state.state not in [STATE_UNKNOWN, STATE_UNAVAILABLE]:
                self.session_input_parameters["coffee_name"] = coffee_name_state.state
                _LOGGER.debug("Logged coffee_name: %s from %s", coffee_name_state.state, self.linked_coffee_name_entity_id)
            else:
                _LOGGER.warning("Could not read state for linked coffee name entity: %s", self.linked_coffee_name_entity_id)

        self.async_update_listeners()  # Notify HA about state change (e.g., binary_sensor)

    async def _stop_session(self, stop_reason: str) -> None:
        """Internal method to stop the current shot session."""
        if not self.is_shot_active or not self.session_start_time_utc:
            _LOGGER.debug(
                "Stop session called but no active session or start time found."
            )
            return

        current_time = dt_util.utcnow()
        shot_duration = (current_time - self.session_start_time_utc).total_seconds()
        _LOGGER.info(
            "Stopping shot session (reason: %s). Duration: %.2f seconds.",
            stop_reason,
            shot_duration,
        )

        shot_status = "completed"
        # Preserve start time and inputs for potential 'aborted_too_short' last_shot_data
        # Ensure self.session_start_time_utc is not None before calling isoformat
        original_start_time_utc_iso = self.session_start_time_utc.isoformat() if self.session_start_time_utc else None
        original_start_trigger = self.session_start_trigger
        original_input_params = dict(self.session_input_parameters) # Make a copy before it's cleared

        if stop_reason == "disconnected":
            shot_status = "aborted_disconnected"
        elif stop_reason not in ["ha_service_stop_forced"] and shot_duration < self.min_shot_duration:
            _LOGGER.info(
                "Shot duration (%.2f s) is less than minimum configured (%s s). Aborting full log.",
                shot_duration, self.min_shot_duration
            )
            shot_status = "aborted_too_short"
            
            self.last_shot_data = {
                "device_id": self.config_entry.unique_id or self.config_entry.entry_id,
                "entry_id": self.config_entry.entry_id,
                "start_time_utc": original_start_time_utc_iso,
                "end_time_utc": current_time.isoformat(),
                "duration_seconds": round(shot_duration, 2),
                "status": shot_status,
                "start_trigger": original_start_trigger,
                "stop_reason": stop_reason,
                "input_parameters": original_input_params,
                "final_weight_grams": 0.0, # No reliable final weight for aborted short shot
                "flow_profile": [], # No profile for aborted short shot
                "scale_timer_profile": [], # No profile for aborted short shot
            }
            # Reset session variables
            self.is_shot_active = False
            self.session_start_time_utc = None
            self.session_flow_profile = []
            self.session_scale_timer_profile = []
            self.session_start_trigger = None
            self.session_input_parameters = {}
            self.async_update_listeners() # Notify HA about state change (shot ended)
            return # Do not fire full event for aborted short shot

        # For completed shots:
        final_weight_grams = self.scale.weight if self.scale.weight is not None else 0.0
        if not self.session_flow_profile and final_weight_grams == 0.0:
             _LOGGER.warning("Session flow profile is empty and scale weight is 0, final_weight_grams might be inaccurate.")
        # Note: self.session_flow_profile might store flow rate, not absolute weight. 
        # Using self.scale.weight is a placeholder for a more robust way to get final yield if needed.

        event_data = {
            "device_id": self.config_entry.unique_id or self.config_entry.entry_id,
            "entry_id": self.config_entry.entry_id,
            "start_time_utc": original_start_time_utc_iso,
            "end_time_utc": current_time.isoformat(),
            "duration_seconds": round(shot_duration, 2),
            "final_weight_grams": round(final_weight_grams, 2), 
            "flow_profile": self.session_flow_profile,
            "scale_timer_profile": self.session_scale_timer_profile,
            "start_trigger": original_start_trigger,
            "stop_reason": stop_reason,
            "status": shot_status, 
            "input_parameters": original_input_params,
        }
        
        self.last_shot_data = event_data # Store for 'Last Shot' sensors
        self.hass.bus.async_fire(EVENT_BOOKOO_SHOT_COMPLETED, event_data)
        # Log event_data without the potentially very long profile lists
        logged_event_data = {k: v for k, v in event_data.items() if k not in ['flow_profile', 'scale_timer_profile']}
        _LOGGER.info("Fired EVENT_BOOKOO_SHOT_COMPLETED with data: %s", logged_event_data)

        # Reset session variables
        self.is_shot_active = False
        self.session_start_time_utc = None
        self.session_flow_profile = []
        self.session_scale_timer_profile = []
        self.session_start_trigger = None
        self.session_input_parameters = {} 

        self.async_update_listeners()  # Notify HA about state change
