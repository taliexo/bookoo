"""Data update coordinator and central hub for the Bookoo Home Assistant integration.

Manages the connection to the Bookoo scale, handles data updates,
coordinates shot sessions, and provides data to entities.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TypeAlias  # For Optional type hint and TypeAlias

from aiobookoov2.bookooscale import BookooScale
from aiobookoov2.const import (
    UPDATE_SOURCE_COMMAND_CHAR,
    UPDATE_SOURCE_WEIGHT_CHAR,
)
from aiobookoov2.exceptions import BookooDeviceNotFound, BookooError
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_ADDRESS
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .analytics import ShotAnalyzer
from .const import CONF_IS_VALID_SCALE, BookooConfig  # Assuming DOMAIN might be needed
from .session_manager import SessionManager
from .types import BookooShotCompletedEventDataModel

_LOGGER = logging.getLogger(__name__)

# Type alias for the config entry specific to this integration
BookooConfigEntry: TypeAlias = ConfigEntry["BookooCoordinator"]

SCAN_INTERVAL = timedelta(seconds=5)  # Example, adjust as needed
ANALYTICS_UPDATE_INTERVAL = timedelta(
    seconds=0.5
)  # Update analytics at most every 0.5 seconds


class BookooCoordinator(
    DataUpdateCoordinator[None]
):  # Specify None if not pushing polled data
    """Manages all interactions with the Bookoo scale and integration data.

    This coordinator handles:
    - Bluetooth connection and data parsing from the BookooScale.
    - Shot session management via the SessionManager.
    - Real-time analytics updates.
    - Service call handling for starting/stopping shots.
    - Providing data updates to registered listeners (entities).
    """

    config_entry: BookooConfigEntry  # Type hint for the config entry

    def __init__(self, hass: HomeAssistant, entry: BookooConfigEntry) -> None:
        """Initialize coordinator."""
        self._scale = BookooScale(
            address_or_ble_device=entry.data[CONF_ADDRESS],
            name=entry.title,
            is_valid_scale=entry.data.get(CONF_IS_VALID_SCALE, False),
            notify_callback=self.async_update_listeners,
            characteristic_update_callback=self._handle_characteristic_update,
        )

        super().__init__(
            hass,
            _LOGGER,
            name=f"Bookoo {self._scale.mac or entry.title}",  # Use MAC for a more unique name
            update_interval=SCAN_INTERVAL,
            config_entry=entry,
        )

        self.session_manager = SessionManager(hass, self)
        self.last_shot_data: BookooShotCompletedEventDataModel | None = None
        self.shot_analyzer = ShotAnalyzer()

        # Real-time analytics attributes
        self.realtime_channeling_status: str = "Undetermined"
        self.realtime_pre_infusion_active: bool = False
        self.realtime_pre_infusion_duration: float | None = None
        self.realtime_extraction_uniformity: float | None = 0.0
        self.realtime_shot_quality_score: float | None = 0.0

        # Load options using the BookooConfig dataclass
        self.bookoo_config: BookooConfig = BookooConfig.from_config_entry(entry)

        # Listener for options updates
        self._options_update_listener = entry.add_update_listener(
            self._options_update_callback
        )
        self._last_analytics_update_time: datetime | None = None

    @property
    def scale(self) -> BookooScale:
        """Return the scale object."""
        return self._scale

    def _load_options(self) -> None:
        """Reload options from the config entry into the BookooConfig dataclass."""
        self.bookoo_config = BookooConfig.from_config_entry(self.config_entry)
        _LOGGER.debug(
            "%s: Reloaded options into self.bookoo_config: %s",
            self.name,
            self.bookoo_config,
        )

    async def _options_update_callback(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        _LOGGER.debug("%s: Options update callback triggered.", self.name)
        self._load_options()
        await self.hass.config_entries.async_reload(self.config_entry.entry_id)

    def _reset_realtime_analytics(self) -> None:
        """Resets the real-time analytics attributes."""
        self.realtime_channeling_status = "Undetermined"
        self.realtime_pre_infusion_active = False
        self.realtime_pre_infusion_duration = None
        self.realtime_extraction_uniformity = 0.0
        self.realtime_shot_quality_score = 0.0
        _LOGGER.debug("%s: Coordinator real-time analytics reset.", self.name)
        self._last_analytics_update_time = (
            None  # Reset for immediate calculation on new shot
        )

    def _handle_command_char_update(self, data: bytes | dict | None) -> None:
        """Handle updates from the command characteristic."""
        if isinstance(data, bytes):
            _LOGGER.debug("Received raw bytes from command char: %s.", data.hex())
            self.async_update_listeners()  # Update listeners as scale state might be implied
            return

        if not isinstance(data, dict):
            # If data is None or an unexpected type, still update listeners
            # as a characteristic update occurred.
            _LOGGER.debug(
                "Command char update with no data or unexpected type: %s", type(data)
            )
            self.async_update_listeners()
            return

        # At this point, data is a dict
        msg_type = data.get("type")
        event = data.get("event")
        _LOGGER.debug(
            "Processing decoded command data: type='%s', event='%s'", msg_type, event
        )

        if msg_type == "auto_timer":
            if event == "start" and not self.session_manager.is_shot_active:
                _LOGGER.info(
                    "%s: Scale auto-timer (decoded dict) started shot.", self.name
                )
                self._reset_realtime_analytics()
                self.hass.async_create_task(
                    self.session_manager.start_session(trigger="scale_auto_dict")
                )
            elif event == "stop" and self.session_manager.is_shot_active:
                _LOGGER.info(
                    "%s: Scale auto-timer (decoded dict) stopped shot.", self.name
                )
                self.hass.async_create_task(
                    self.session_manager.stop_session(stop_reason="scale_auto_dict")
                )

        # Always update listeners after processing command char data,
        # as actions taken (or not taken) might be relevant.
        self.async_update_listeners()

    def _update_realtime_analytics_if_needed(self) -> None:
        """Update real-time analytics if a shot is active and throttling interval has passed."""
        if not (
            self.session_manager.is_shot_active  # Ensure shot is still active
            and self.session_manager.session_start_time_utc  # And start time is set
        ):
            return

        now = datetime.now(timezone.utc)
        if not (
            self._last_analytics_update_time is None
            or (now - self._last_analytics_update_time) > ANALYTICS_UPDATE_INTERVAL
        ):
            return  # Throttled

        if not self.session_manager.session_flow_profile:  # Ensure there's data
            _LOGGER.debug("%s: No flow profile data to update analytics.", self.name)
            return

        _LOGGER.debug("%s: Updating real-time analytics.", self.name)
        self.realtime_channeling_status = self.shot_analyzer.detect_channeling(
            list(self.session_manager.session_flow_profile)
        )
        (
            self.realtime_pre_infusion_active,
            self.realtime_pre_infusion_duration,
        ) = self.shot_analyzer.identify_pre_infusion(
            list(self.session_manager.session_flow_profile),
            list(self.session_manager.session_scale_timer_profile),
        )
        self.realtime_extraction_uniformity = (
            self.shot_analyzer.calculate_extraction_uniformity(
                list(self.session_manager.session_flow_profile)
            )
        )
        self._update_shot_quality_score()  # This method calculates and sets realtime_shot_quality_score
        self._last_analytics_update_time = now
        self.async_update_listeners()  # Update listeners after analytics change

    def _handle_weight_char_update(self) -> None:
        """Handle updates from the weight characteristic."""
        # Note: `data` argument is not used here as aiobookoov2's callback for weight
        # char provides None for `data`, and we rely on self._scale attributes.
        if (
            self.session_manager.is_shot_active
            and self.session_manager.session_start_time_utc
        ):
            current_time_elapsed = (
                datetime.now(timezone.utc) - self.session_manager.session_start_time_utc
            ).total_seconds()

            if self._scale.weight is not None:
                self.session_manager.add_weight_data(
                    current_time_elapsed, self._scale.weight
                )
            if self._scale.flow_rate is not None:
                self.session_manager.add_flow_data(
                    current_time_elapsed, self._scale.flow_rate
                )
            if self._scale.timer is not None:
                self.session_manager.add_scale_timer_data(
                    current_time_elapsed, int(self._scale.timer)
                )

            self._update_realtime_analytics_if_needed()

        # Always update listeners for weight char as it drives sensor updates.
        # The _update_realtime_analytics_if_needed method also calls this, but
        # we want to ensure an update even if analytics weren't re-calculated
        # (e.g. due to throttling or no flow data yet) because basic scale
        # weight/timer sensors should still refresh.
        self.async_update_listeners()

    def _handle_characteristic_update(
        self, source: str, data: bytes | dict | None
    ) -> None:
        """Handle updates from aiobookoo's characteristic_update_callback."""
        _LOGGER.debug(
            "%s: [HANDLE_CHAR_DEBUG] Update from source: %s, type: %s, data: %s",
            self.name,
            source,
            type(data).__name__,
            data.hex() if isinstance(data, bytes) else data,
        )

        if source == UPDATE_SOURCE_COMMAND_CHAR:
            self._handle_command_char_update(data)
        elif source == UPDATE_SOURCE_WEIGHT_CHAR:
            # The `data` param for weight char is None from aiobookoov2, handled in helper.
            if data is not None:
                _LOGGER.warning(
                    "Unexpected data with UPDATE_SOURCE_WEIGHT_CHAR: %s. Expected None.",
                    data,
                )
            self._handle_weight_char_update()
        else:
            _LOGGER.warning("Unknown characteristic update source: %s", source)

    def _update_shot_quality_score(self) -> None:
        """Calculate the real-time shot quality score."""
        if self.realtime_extraction_uniformity is None:
            self.realtime_shot_quality_score = 0.0
            return

        quality_score = self.realtime_extraction_uniformity * 100.0
        channeling_penalty = 0
        if self.realtime_channeling_status == "Mild Channeling (High Variation)":
            channeling_penalty = 15
        elif self.realtime_channeling_status == "Suspected Channeling (Spike)":
            channeling_penalty = 20
        elif (
            self.realtime_channeling_status
            == "Mild-Moderate Channeling (Variation & Notable Peak)"
        ):
            channeling_penalty = 25
        elif (
            self.realtime_channeling_status
            == "Moderate Channeling (High Variation & Spike)"
        ):
            channeling_penalty = 30

        quality_score -= channeling_penalty
        self.realtime_shot_quality_score = max(0, min(100, quality_score))
        _LOGGER.debug(
            "Updated shot quality score: %.1f", self.realtime_shot_quality_score
        )

    def _handle_active_shot_disconnection(self, stop_reason_suffix: str) -> None:
        """Handles stopping an active shot due to a disconnection event."""
        _LOGGER.warning(
            "%s: Scale disconnected/failed during an active shot. Ending session (reason suffix: %s).",
            self.name,
            stop_reason_suffix,
        )
        self.hass.async_create_task(
            self.session_manager.stop_session(
                stop_reason=f"disconnected_{stop_reason_suffix}"
            )
        )

    def _ensure_queue_processor_running(self) -> None:
        """Ensures the BookooScale's process_queue task is running."""
        if not self._scale.connected:
            _LOGGER.warning(
                "%s: Cannot ensure queue processor running, scale not connected.",
                self.name,
            )
            return

        if (
            self._scale.process_queue_task is None
            or self._scale.process_queue_task.done()
        ):
            _LOGGER.info(
                "%s: process_queue task not running or completed. Restarting.",
                self.name,
            )
            if self._scale.process_queue_task and self._scale.process_queue_task.done():
                try:
                    if ex := self._scale.process_queue_task.exception():
                        _LOGGER.warning(
                            "%s: Previous process_queue task ended with exception: %s",
                            self.name,
                            ex,
                        )
                except asyncio.CancelledError:
                    _LOGGER.debug(
                        "%s: Previous process_queue task was cancelled.", self.name
                    )

            self._scale.process_queue_task = (
                self.config_entry.async_create_background_task(
                    hass=self.hass,
                    target=self._scale.process_queue(),  # type: ignore[no-untyped-call]
                    name="bookoo_process_queue_task",
                )
            )
            _LOGGER.debug("%s: process_queue background task (re)started.", self.name)

    async def _attempt_bookoo_connection(self) -> None:
        """Attempts to connect to the Bookoo scale if not already connected.
        Raises exceptions on failure.
        """
        if self._scale.connected:
            return

        _LOGGER.info("%s: Scale not connected, attempting to connect.", self.name)
        try:
            # Use a timeout for the connection attempt itself
            async with asyncio.timeout(self.bookoo_config.connect_timeout):
                connected_successfully = await self._scale.async_connect()  # type: ignore[attr-defined] # MyPy error: async_connect

            if not connected_successfully:
                _LOGGER.warning(
                    "%s: Failed to connect to scale (async_connect returned False).",
                    self.name,
                )
                # This specific condition is treated as a timeout for consistency in error handling
                raise asyncio.TimeoutError(
                    "Connection attempt failed (async_connect returned False)"
                )

            _LOGGER.info("%s: Successfully connected to scale.", self.name)
            # Defensive check, though async_connect returning True should mean connected.
            if not self._scale.connected:
                _LOGGER.error(
                    "%s: async_connect reported success but scale.connected is False.",
                    self.name,
                )
                # This indicates an inconsistency, treat as a BookooError.
                raise BookooError(
                    "Scale connection state inconsistent after async_connect."
                )

        except asyncio.TimeoutError as err:
            _LOGGER.warning(
                "%s: Timeout connecting to Bookoo scale: %s", self.name, err
            )
            # Ensure scale is marked as not connected on timeout
            if hasattr(self._scale, "connected"):  # Check if mock has 'connected'
                self._scale.connected = False  # type: ignore[assignment] # For mocks
            raise  # Re-raise to be caught by _async_update_data
        except BookooError as err:
            _LOGGER.warning("%s: Error connecting to Bookoo scale: %s", self.name, err)
            if hasattr(self._scale, "connected"):
                self._scale.connected = False  # type: ignore[assignment]
            raise  # Re-raise
        except Exception as err:
            _LOGGER.exception(
                "%s: Unexpected error during Bookoo scale connection: %s",
                self.name,
                err,
            )
            if hasattr(self._scale, "connected"):
                self._scale.connected = False  # type: ignore[assignment]
            raise BookooError(f"Unexpected error connecting to scale: {err}") from err

    async def async_close(self) -> None:
        """Close resources and disconnect the scale."""
        _LOGGER.debug("Closing BookooCoordinator resources for %s", self.name)
        if self._scale:
            if self._scale.connected:
                try:
                    _LOGGER.debug(
                        "Disconnecting scale %s during async_close", self.name
                    )
                    await self._scale.disconnect()
                    _LOGGER.debug(
                        "Scale %s disconnected successfully during async_close",
                        self.name,
                    )
                except Exception as e:
                    _LOGGER.error(
                        "Error disconnecting scale %s during async_close: %s",
                        self.name,
                        e,
                    )
            else:
                _LOGGER.debug(
                    "Scale %s already disconnected, skipping disconnect.", self.name
                )
        else:
            _LOGGER.debug("No scale object found to disconnect for %s.", self.name)

        # Call super().async_shutdown() to ensure DataUpdateCoordinator cleans up its resources
        await super().async_shutdown()

    async def _ensure_scale_connected_and_processing(self) -> None:
        """Ensure the scale is connected and its data processing queue is active.

        Raises:
            UpdateFailed: If connection or queue setup fails.
            BookooDeviceNotFound: If the device is not found.
            BookooError: For other Bookoo specific errors during connection.
            asyncio.TimeoutError: If connection times out.
        """
        # Step 1: Attempt connection (will raise on failure)
        # The _attempt_bookoo_connection method handles its own logging for various states.
        await self._attempt_bookoo_connection()

        # Step 2: If connection successful (i.e., no exception raised by step 1),
        # ensure queue processor is running.
        # _attempt_bookoo_connection should ensure self._scale.is_connected is True if it returns.
        if not self._scale.connected:
            # This path indicates an unexpected state if _attempt_bookoo_connection is supposed to always raise on failure.
            _LOGGER.error(
                "%s: Scale not connected after _attempt_bookoo_connection, but no exception was raised. This is unexpected.",
                self.name,
            )
            raise UpdateFailed(
                f"Bookoo scale {self.name} connection state inconsistent."
            )

        self._ensure_queue_processor_running()

    def _handle_specific_update_exception(
        self, err: Exception, error_message_prefix: str, disconnect_reason_suffix: str
    ) -> None:
        """Handles common logic for specific known exceptions during _async_update_data."""
        _LOGGER.warning("%s: %s: %s", self.name, error_message_prefix, err)
        if self.session_manager.is_shot_active:
            self._handle_active_shot_disconnection(
                f"{disconnect_reason_suffix}_during_update"
            )
        raise UpdateFailed(f"{error_message_prefix}: {err}") from err

    async def _async_update_data(self) -> None:
        """Fetch the latest data from the Bookoo scale.

        This method is called by the DataUpdateCoordinator base class to refresh data.
        It ensures the scale is connected and processes any queued updates.
        Returns None as entities subscribe to updates rather than fetching data directly
        from this method's return value.
        Raises UpdateFailed on critical errors to signal HA.
        """
        _LOGGER.debug("%s: Attempting to update data from Bookoo scale.", self.name)
        try:
            await self._ensure_scale_connected_and_processing()
            _LOGGER.debug(
                "%s: Data update check complete. Scale connected and processing.",
                self.name,
            )
        except BookooDeviceNotFound as err:
            self._handle_specific_update_exception(
                err, "Bookoo scale device not found", "device_not_found"
            )
        except (
            BookooError
        ) as err:  # Catches specific Bookoo errors like connection issues
            self._handle_specific_update_exception(
                err, "Error communicating with Bookoo scale", "bookoo_error"
            )
        except asyncio.TimeoutError as err:  # Specifically for asyncio.TimeoutError
            self._handle_specific_update_exception(
                err, "Timeout connecting to Bookoo scale", "timeout"
            )
        except Exception as err:  # Catch-all for other unexpected errors
            _LOGGER.exception(  # Keep full exception log for truly unexpected errors
                "%s: Unexpected error updating Bookoo scale data: %s", self.name, err
            )
            if self.session_manager.is_shot_active:
                self._handle_active_shot_disconnection("unexpected_error_during_update")
            # Ensure we always raise UpdateFailed for the coordinator's error handling
            if isinstance(err, UpdateFailed):  # If it's already UpdateFailed, re-raise
                raise
            raise UpdateFailed(f"Unexpected error updating Bookoo data: {err}") from err
        # This return is only reached if the try block completes successfully.

    # Service call handlers
    async def async_start_shot_service(self, call: ServiceCall) -> None:
        """Service call to start a new shot session via HA."""
        if self.session_manager.is_shot_active:
            _LOGGER.warning(
                "%s: Start shot service called, but a shot is already active.",
                self.name,
            )
            return
        _LOGGER.info("%s: HA service starting shot.", self.name)
        self._reset_realtime_analytics()
        # Create task for starting session to not block service call return
        self.hass.async_create_task(
            self.session_manager.start_session(trigger="ha_service")
        )
        try:
            await self._scale.tare_and_start_timer()
            _LOGGER.debug("Sent Tare & Start Timer command to scale.")
        except BookooError as e:
            _LOGGER.error(
                "%s: Error sending Tare & Start Timer command to scale: %s",
                self.name,
                e,
            )
            # Optionally, stop session if scale command failed after session started
            # self.hass.async_create_task(self.session_manager.stop_session(stop_reason="command_failed_start"))

    async def async_stop_shot_service(self, call: ServiceCall) -> None:
        """Service call to stop the current shot session via HA."""
        if not self.session_manager.is_shot_active:
            _LOGGER.warning(
                "%s: Stop shot service called, but no shot is active.", self.name
            )
            return
        _LOGGER.info("%s: HA service stopping shot.", self.name)
        stop_reason_for_session = "ha_service"
        try:
            await self._scale.stop_timer()
            _LOGGER.debug("Sent Stop Timer command to scale.")
        except BookooError as e:
            _LOGGER.error(
                "%s: Error sending Stop Timer command to scale: %s", self.name, e
            )
            stop_reason_for_session = "ha_service_scale_error"

        # Create task for stopping session to not block service call return
        self.hass.async_create_task(
            self.session_manager.stop_session(stop_reason=stop_reason_for_session)
        )
