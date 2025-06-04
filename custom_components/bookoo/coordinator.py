"""Coordinator for Bookoo integration."""

from __future__ import annotations

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
BookooConfigEntry: TypeAlias = ConfigEntry  # runtime_data will store BookooCoordinator

SCAN_INTERVAL = timedelta(seconds=5)  # Example, adjust as needed


class BookooCoordinator(
    DataUpdateCoordinator[None]
):  # Specify None if not pushing polled data
    """Class to handle fetching data from the scale and coordinating updates."""

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
            if isinstance(data, dict):
                msg_type = data.get("type")
                event = data.get("event")
                _LOGGER.debug(
                    "Processing decoded command data: type='%s', event='%s'",
                    msg_type,
                    event,
                )
                if msg_type == "auto_timer" and event == "start":
                    if not self.session_manager.is_shot_active:
                        _LOGGER.info(
                            "%s: Scale auto-timer (decoded dict) started shot.",
                            self.name,
                        )
                        self._reset_realtime_analytics()
                        self.hass.async_create_task(
                            self.session_manager.start_session(
                                trigger="scale_auto_dict"
                            )
                        )
                elif msg_type == "auto_timer" and event == "stop":
                    if self.session_manager.is_shot_active:
                        _LOGGER.info(
                            "%s: Scale auto-timer (decoded dict) stopped shot.",
                            self.name,
                        )
                        self.hass.async_create_task(
                            self.session_manager.stop_session(
                                stop_reason="scale_auto_dict"
                            )
                        )
            elif isinstance(data, bytes):
                _LOGGER.debug("Received raw bytes from command char: %s.", data.hex())
            # Potentially update listeners if command char changes state relevant to HA
            self.async_update_listeners()

        elif source == UPDATE_SOURCE_WEIGHT_CHAR:
            if data is not None:  # Should be None from aiobookoov2
                _LOGGER.warning(
                    "Unexpected data with UPDATE_SOURCE_WEIGHT_CHAR: %s. Expected None.",
                    data,
                )

            if (
                self.session_manager.is_shot_active
                and self.session_manager.session_start_time_utc
            ):
                current_time_elapsed = (
                    datetime.now(timezone.utc)
                    - self.session_manager.session_start_time_utc
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

                if self.session_manager.session_flow_profile:  # Ensure there's data
                    self.realtime_channeling_status = (
                        self.shot_analyzer.detect_channeling(
                            self.session_manager.session_flow_profile
                        )
                    )
                    (
                        self.realtime_pre_infusion_active,
                        self.realtime_pre_infusion_duration,
                    ) = self.shot_analyzer.identify_pre_infusion(
                        self.session_manager.session_flow_profile,
                        self.session_manager.session_scale_timer_profile,
                    )
                    self.realtime_extraction_uniformity = (
                        self.shot_analyzer.calculate_extraction_uniformity(
                            self.session_manager.session_flow_profile
                        )
                    )
                    self._update_shot_quality_score()
            # Always update listeners for weight char as it drives sensor updates
            self.async_update_listeners()
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

        # Call super().async_close() to ensure DataUpdateCoordinator cleans up its resources
        await super().async_close()

    async def _async_update_data(self) -> None:
        """Fetch data from the Bookoo scale, ensuring connection."""
        try:
            if not self._scale.is_connected:  # type: ignore[attr-defined]
                _LOGGER.info(
                    "%s: Scale not connected. Attempting to connect.", self.name
                )
                connected = await self._scale.async_connect()  # type: ignore[attr-defined]
                if not connected:
                    _LOGGER.warning("%s: Failed to connect to scale.", self.name)
                    raise UpdateFailed(f"Failed to connect to Bookoo scale {self.name}")
                _LOGGER.info("%s: Successfully connected to scale.", self.name)

            if hasattr(self._scale, "process_queue") and (
                not hasattr(self._scale, "process_queue_task")
                or self._scale.process_queue_task is None  # type: ignore[attr-defined]
                or self._scale.process_queue_task.done()  # type: ignore[attr-defined]
            ):
                _LOGGER.debug(
                    "%s: Starting/restarting scale data processing queue task.",
                    self.name,
                )
                self._scale.process_queue_task = (
                    self.config_entry.async_create_background_task(
                        hass=self.hass,
                        target=self._scale.process_queue(),  # type: ignore[attr-defined]
                        name="bookoo_process_queue_task",
                    )
                )
        except BookooDeviceNotFound as ex:
            _LOGGER.info("%s: Scale device not found during update: %s", self.name, ex)
            if self._scale.is_connected:  # type: ignore[attr-defined] # Should ideally be false if device not found
                try:
                    await self._scale.async_disconnect()  # type: ignore[attr-defined]
                except BookooError:
                    pass
            self._scale.is_connected = False  # type: ignore[attr-defined] # Ensure state is updated
            if self.session_manager.is_shot_active:
                self._handle_active_shot_disconnection("device_not_found_during_update")
            raise UpdateFailed(f"Bookoo scale device not found: {ex}") from ex
        except BookooError as ex:
            _LOGGER.warning(
                "%s: A Bookoo specific error occurred during update: %s", self.name, ex
            )
            if self.session_manager.is_shot_active:
                self._handle_active_shot_disconnection("bookoo_error_during_update")
            # Consider if this error implies disconnection
            # if implies_disconnection(ex): self._scale.is_connected = False # type: ignore[attr-defined]
            raise UpdateFailed(f"Bookoo error during update: {ex}") from ex
        except TimeoutError as ex:
            _LOGGER.warning("%s: Timeout during scale communication: %s", self.name, ex)
            if self._scale.is_connected:  # type: ignore[attr-defined]
                try:
                    await self._scale.async_disconnect()  # type: ignore[attr-defined]
                except BookooError:
                    pass
            self._scale.is_connected = False  # type: ignore[attr-defined]
            if self.session_manager.is_shot_active:
                self._handle_active_shot_disconnection("timeout_during_update")
            raise UpdateFailed(f"Timeout communicating with Bookoo scale: {ex}") from ex
        except Exception as ex:
            if isinstance(
                ex, UpdateFailed
            ):  # If it's already an UpdateFailed, re-raise it directly
                raise
            _LOGGER.exception(
                "%s: Unexpected error during data update: %s", self.name, ex
            )
            if self._scale.is_connected:  # type: ignore[attr-defined] # Try to clean up connection
                try:
                    await self._scale.async_disconnect()  # type: ignore[attr-defined]
                except BookooError:
                    pass
            self._scale.is_connected = False  # type: ignore[attr-defined]
            if self.session_manager.is_shot_active:
                self._handle_active_shot_disconnection("unexpected_error_during_update")
            raise UpdateFailed(f"Unexpected error updating Bookoo data: {ex}") from ex

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
