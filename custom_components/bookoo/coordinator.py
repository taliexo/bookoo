"""Coordinator for Bookoo integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any  # For session_input_parameters

from aiobookoo.const import UPDATE_SOURCE_COMMAND_CHAR, UPDATE_SOURCE_WEIGHT_CHAR
from aiobookoo.decode import (
    decode as aiobookoo_decode,
)  # Alias to avoid name clash if local decode exists
import logging

from aiobookoo.bookooscale import BookooScale
from aiobookoo.exceptions import BookooDeviceNotFound, BookooError

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ADDRESS
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .const import (
    CONF_IS_VALID_SCALE,
    EVENT_BOOKOO_SHOT_COMPLETED,
)  # Assuming DOMAIN is needed for event firing context

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
            name=f"Bookoo {self._scale.address}",
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

    @property
    def scale(self) -> BookooScale:
        """Return the scale object."""
        return self._scale

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
                    else:
                        _LOGGER.debug(
                            "Scale auto-timer stop event (decoded via type/event) received, but no shot active."
                        )
                else:
                    _LOGGER.debug("Received other or unrecognized decoded command data structure: %s", decoded_cmd_data)
                    # TODO: Handle other types of command characteristic responses if necessary
            else:
                _LOGGER.debug("Command char data not decoded into a dict or was None: %s. Raw byte checks might be needed if this is unexpected.", decoded_cmd_data)
                # Fallback to raw byte checks if aiobookoo_decode doesn't handle it (for now, keeping original logic as fallback)
                # This section can be removed if aiobookoo_decode is confirmed to handle all command char events.
                if (
                    len(data) >= 3 # Corrected length check for data[2]
                    and data[0] == 0x03 # CMD_BYTE1_PRODUCT_NUMBER
                    and data[1] == 0x0D # CMD_BYTE2_MESSAGE_TYPE_AUTO_TIMER
                    and data[2] == 0x01 # CMD_BYTE3_AUTO_TIMER_EVENT_START
                ):
                    if not self.is_shot_active:
                        _LOGGER.info("Scale auto-timer (raw) started shot.")
                        self.hass.async_create_task(
                            self._start_session(trigger="scale_auto_raw")
                        )
                    else:
                        _LOGGER.debug(
                            "Scale auto-timer start event (raw) received, but shot already active."
                        )
                elif (
                    len(data) >= 3 # Corrected length check for data[2]
                    and data[0] == 0x03
                    and data[1] == 0x0D
                    and data[2] == 0x00 # CMD_BYTE3_AUTO_TIMER_EVENT_STOP
                ):
                    if self.is_shot_active:
                        _LOGGER.info("Scale auto-timer (raw) stopped shot.")
                        self.hass.async_create_task(
                            self._stop_session(stop_reason="scale_auto_stop_raw")
                        )
                    else:
                        _LOGGER.debug(
                            "Scale auto-timer stop event (raw) received, but no shot active."
                        )

        elif source == UPDATE_SOURCE_WEIGHT_CHAR and self.is_shot_active:
            _LOGGER.debug(
                "[HANDLE_CHAR_DEBUG] Entered WEIGHT_CHAR block. Source: '%s', Active: %s", 
                source, self.is_shot_active
            )
            if not self.session_start_time_utc: # Safety check
                _LOGGER.warning("WEIGHT_CHAR: Shot active but session_start_time_utc is None. Aborting update.")
                return

            try:
                decoded_data = aiobookoo_decode(data) if data else None
                _LOGGER.debug("[HANDLE_CHAR_DEBUG] decoded_data (weight): %s (type: %s)", decoded_data, type(decoded_data).__name__)

                if not decoded_data:
                    _LOGGER.error("Received None from aiobookoo_decode for weight char update.")
                    return
                if not isinstance(decoded_data, dict):
                    _LOGGER.error(
                        "aiobookoo_decode did not return a dict for weight char. Got: %s (type: %s)",
                        decoded_data, type(decoded_data).__name__
                    )
                    return

                current_time = dt_util.utcnow()
                elapsed_shot_time = round(
                    (current_time - self.session_start_time_utc).total_seconds(), 2
                )

                timer_milliseconds = decoded_data.get("timer_milliseconds") # Expecting int or None
                flow_rate = decoded_data.get("flow_rate", 0.0) # Expecting float or None, default to 0.0

                self.session_flow_profile.append((elapsed_shot_time, flow_rate))
                self.session_scale_timer_profile.append(
                    (elapsed_shot_time, timer_milliseconds if timer_milliseconds is not None else 0)
                )
                
                # Update live sensors via listeners for this specific update
                self.async_update_listeners()
            except Exception as e:
                _LOGGER.error(
                    "Error processing weight data during active shot: %s", e, exc_info=True
                )
            return # Finished processing weight char update

        self.async_update_listeners()  # General update for any other listeners/sensors

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
            # This case should ideally be caught by callers
            _LOGGER.warning(
                "Attempted to start a session when one is already active. Trigger: %s",
                trigger,
            )
            return

        self.is_shot_active = True
        self.session_start_time_utc = dt_util.utcnow()
        self.session_flow_profile = []
        self.session_scale_timer_profile = []
        self.session_start_trigger = trigger

        # Capture input parameters from linked entities
        self.session_input_parameters = {}
        options = self.config_entry.options

        bean_weight_entity_id = options.get("linked_bean_weight_entity")
        if bean_weight_entity_id and (
            bean_weight_state := self.hass.states.get(bean_weight_entity_id)
        ):
            self.session_input_parameters["bean_weight_grams"] = bean_weight_state.state

        coffee_name_entity_id = options.get("linked_coffee_name_entity")
        if coffee_name_entity_id and (
            coffee_name_state := self.hass.states.get(coffee_name_entity_id)
        ):
            self.session_input_parameters["coffee_name"] = coffee_name_state.state

        # Add logic for other linked entities here, e.g.:
        # target_weight_entity_id = options.get("linked_target_weight_entity")
        # if target_weight_entity_id and (target_weight_state := self.hass.states.get(target_weight_entity_id)):
        #     self.session_input_parameters["target_weight_grams"] = target_weight_state.state
        _LOGGER.info(
            "Espresso shot session started. Trigger: %s, Start time: %s, Inputs: %s",
            trigger,
            self.session_start_time_utc,
            self.session_input_parameters,
        )
        self.async_update_listeners()  # Update binary sensor, etc.

    async def _stop_session(self, stop_reason: str) -> None:
        """Internal method to stop the current shot session."""
        if not self.is_shot_active or not self.session_start_time_utc:
            _LOGGER.debug(
                "Stop session called but no active session or start time found."
            )
            return

        session_end_time_utc = dt_util.utcnow()
        duration = (session_end_time_utc - self.session_start_time_utc).total_seconds()

        # Retrieve minimum shot duration from config entry options
        min_duration = self.config_entry.options.get("minimum_shot_duration_seconds", 5)
        _LOGGER.debug("Using minimum shot duration: %s seconds", min_duration)

        final_weight = self.scale.weight  # Get final weight from scale
        shot_status = "completed"

        if stop_reason == "disconnected":
            shot_status = "aborted_disconnected"
            _LOGGER.warning(
                "Shot session aborted due to disconnection. Duration: %.2f s", duration
            )
        elif duration < min_duration:
            shot_status = "aborted_too_short"
            _LOGGER.info(
                "Shot session ended but was too short (%.2f s) to be recorded. Min duration: %s s. Reason: %s",
                duration,
                min_duration,
                stop_reason,
            )
            # Populate last_shot_data for aborted_too_short shots
            self.last_shot_data = {
                "device_id": self.config_entry.unique_id,
                "entry_id": self.config_entry.entry_id,
                "start_time_utc": self.session_start_time_utc.isoformat() if self.session_start_time_utc else None,
                "end_time_utc": session_end_time_utc.isoformat(),
                "duration_seconds": round(duration, 2),
                "final_weight_grams": final_weight if final_weight is not None else 0.0,
                "flow_profile_gps": [
                    (round(t, 2), round(f, 2)) for t, f in self.session_flow_profile
                ],
                "scale_timer_profile_ms": [
                    (round(t, 2), ms) for t, ms in self.session_scale_timer_profile
                ],
                "input_parameters": dict(self.session_input_parameters), # Use a copy before it's reset
                "start_trigger": self.session_start_trigger,
                "stop_reason": stop_reason,
                "status": shot_status, # This will be "aborted_too_short"
            }
            
            # Reset session state
            self.is_shot_active = False
            self.session_start_time_utc = None
            self.session_flow_profile = []
            self.session_scale_timer_profile = []
            self.session_input_parameters = {}
            self.session_start_trigger = None
            self.async_update_listeners()
            return  # Do not fire event for too-short shots

        _LOGGER.debug("[STOP_SESSION_DEBUG] self.session_input_parameters before event_data: %s", self.session_input_parameters)
        event_data = {
            "device_id": self.config_entry.unique_id,  # or other device identifier
            "entry_id": self.config_entry.entry_id,
            "start_time_utc": self.session_start_time_utc.isoformat(),
            "end_time_utc": session_end_time_utc.isoformat(),
            "duration_seconds": round(duration, 2),
            "final_weight_grams": final_weight if final_weight is not None else 0.0,
            "flow_profile_gps": [
                (round(t, 2), round(f, 2)) for t, f in self.session_flow_profile
            ],
            "scale_timer_profile_ms": [
                (round(t, 2), ms) for t, ms in self.session_scale_timer_profile
            ],
            "input_parameters": dict(self.session_input_parameters), # Use a copy
            "start_trigger": self.session_start_trigger,
            "stop_reason": stop_reason,
            "status": shot_status,
            # Potentially add brew ratio, average flow rate if calculated here
        }

        self.hass.bus.async_fire(EVENT_BOOKOO_SHOT_COMPLETED, event_data)
        _LOGGER.info("Fired %s event: %s", EVENT_BOOKOO_SHOT_COMPLETED, event_data)

        self.last_shot_data = event_data  # Store for 'Last Shot' sensors

        # Reset session state variables
        self.is_shot_active = False
        self.session_start_time_utc = None
        self.session_flow_profile = []
        self.session_scale_timer_profile = []
        self.session_input_parameters = {} # Correctly placed reset
        self.session_start_trigger = None

        self.async_update_listeners()  # Update binary sensor, last shot sensors, etc.
