"""Manages shot session logic for the Bookoo integration."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TYPE_CHECKING, Optional
import logging

from homeassistant.util import dt as dt_util
from homeassistant.const import STATE_UNKNOWN, STATE_UNAVAILABLE
from .types import (
    FlowProfile,
    WeightProfile,
    ScaleTimerProfile,
    FlowDataPoint,
    WeightDataPoint,
    ScaleTimerDataPoint,
    BookooShotCompletedEventDataModel,  # Changed from TypedDict to Pydantic Model
)
from pydantic import ValidationError
from .storage import async_add_shot_record

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    # To avoid circular import with BookooCoordinator, we type hint it.
    # Actual instance is passed in __init__.
    from .coordinator import BookooCoordinator

_LOGGER = logging.getLogger(__name__)


class SessionManager:
    """Handles the state and logic for an active shot session."""

    def __init__(self, hass: HomeAssistant, coordinator: BookooCoordinator) -> None:
        """Initialize the session manager."""
        self.hass = hass
        self.coordinator = coordinator

        self.is_shot_active: bool = False
        self.session_start_time_utc: datetime | None = None
        self.session_flow_profile: FlowProfile = []
        self.session_weight_profile: WeightProfile = []
        self.session_scale_timer_profile: ScaleTimerProfile = []
        self.session_input_parameters: dict[str, Any] = {}
        self.session_start_trigger: str | None = None
        self.last_shot_data: Optional[BookooShotCompletedEventDataModel] = (
            None  # This will be assigned in stop_session
        )

    def _reset_internal_session_state(self) -> None:
        """Internal helper to reset session-specific state variables."""
        self.is_shot_active = False
        self.session_start_time_utc = None
        self.session_flow_profile = []
        self.session_weight_profile = []
        self.session_scale_timer_profile = []
        self.session_input_parameters = {}
        self.session_start_trigger = None
        # self.last_shot_data is intentionally not cleared here

    async def start_session(self, trigger: str) -> None:
        """Starts a new shot session."""
        # This block was incorrectly modified by the previous tool call and is being reverted/ignored.
        # The actual target for the first model_dump is later in the stop_session method.

        if self.is_shot_active:
            _LOGGER.warning(
                "Attempted to start new shot (trigger: %s) but one is active.",
                trigger,
            )
            return

        _LOGGER.info(
            "%s: Starting new shot session, triggered by: %s",
            self.coordinator.name,
            trigger,
        )
        self._reset_internal_session_state()  # Clear previous session data before starting new
        self.is_shot_active = True
        self.session_start_time_utc = dt_util.utcnow()
        self.session_start_trigger = trigger

        # Read linked input_number/input_text entities
        if self.coordinator.bookoo_config.linked_bean_weight_entity:
            bean_weight_state = self.hass.states.get(
                self.coordinator.bookoo_config.linked_bean_weight_entity
            )
            if bean_weight_state and bean_weight_state.state not in [
                STATE_UNKNOWN,
                STATE_UNAVAILABLE,
            ]:
                self.session_input_parameters["bean_weight"] = bean_weight_state.state
                _LOGGER.debug(
                    "%s: Logged bean_weight: %s from %s",
                    self.coordinator.name,
                    bean_weight_state.state,
                    self.coordinator.bookoo_config.linked_bean_weight_entity,
                )
            else:
                _LOGGER.warning(
                    "%s: Could not read state for linked bean weight entity: %s",
                    self.coordinator.name,
                    self.coordinator.bookoo_config.linked_bean_weight_entity,
                )

        if self.coordinator.bookoo_config.linked_coffee_name_entity:
            coffee_name_state = self.hass.states.get(
                self.coordinator.bookoo_config.linked_coffee_name_entity
            )
            if coffee_name_state and coffee_name_state.state not in [
                STATE_UNKNOWN,
                STATE_UNAVAILABLE,
            ]:
                self.session_input_parameters["coffee_name"] = coffee_name_state.state
                _LOGGER.debug(
                    "%s: Logged coffee_name: %s from %s",
                    self.coordinator.name,
                    coffee_name_state.state,
                    self.coordinator.bookoo_config.linked_coffee_name_entity,
                )
            else:
                _LOGGER.warning(
                    "%s: Could not read state for linked coffee name entity: %s",
                    self.coordinator.name,
                    self.coordinator.bookoo_config.linked_coffee_name_entity,
                )
        # Coordinator will handle resetting its own realtime analytics and updating listeners
        self.coordinator.async_update_listeners()  # Notify HA of state change (shot active)

    async def stop_session(self, stop_reason: str) -> None:
        """Stops the current shot session, calculates metrics, and fires event."""
        if not self.is_shot_active or not self.session_start_time_utc:
            _LOGGER.debug(
                "%s: Stop session called but no active session or start time found.",
                self.coordinator.name,
            )
            return

        current_session_start_time_utc = self.session_start_time_utc
        current_time = dt_util.utcnow()
        duration_seconds = (
            current_time - current_session_start_time_utc
        ).total_seconds()
        _LOGGER.info(
            "%s: Stopping shot session (reason: %s). Duration: %.2f seconds.",
            self.coordinator.name,
            stop_reason,
            duration_seconds,
        )

        shot_status = "completed"
        original_start_trigger = self.session_start_trigger
        original_input_params = dict(self.session_input_parameters)
        original_start_time_utc_iso = current_session_start_time_utc.isoformat()

        # Access min_shot_duration from coordinator's BookooConfig
        min_duration = self.coordinator.bookoo_config.min_shot_duration

        if stop_reason == "disconnected":
            shot_status = "aborted_disconnected"
        elif (
            stop_reason not in ["ha_service_stop_forced"]
            and duration_seconds < min_duration
        ):
            _LOGGER.info(
                "%s: Shot duration (%.2f s) < min configured (%s s). Aborting, saving minimal.",
                self.coordinator.name,
                duration_seconds,
                min_duration,
            )
            shot_status = "aborted_too_short"

            raw_event_data: dict[str, Any] = {
                "device_id": self.coordinator.config_entry.unique_id
                or self.coordinator.config_entry.entry_id,
                "entry_id": self.coordinator.config_entry.entry_id,
                "start_time_utc": original_start_time_utc_iso,
                "end_time_utc": current_time.isoformat(),
                "duration_seconds": round(duration_seconds, 2),
                "status": shot_status,
                "start_trigger": original_start_trigger,
                "stop_reason": stop_reason,
                "input_parameters": original_input_params,
                "final_weight_grams": 0.0,
                "flow_profile": [],  # Minimal data for aborted short shot
                "scale_timer_profile": [],  # Minimal data
                "average_flow_rate_gps": 0.0,
                "peak_flow_rate_gps": 0.0,
                "time_to_first_flow_seconds": None,
                "time_to_peak_flow_seconds": None,
                "channeling_status": self.coordinator.realtime_channeling_status,
                "pre_infusion_detected": self.coordinator.realtime_pre_infusion_active,
                "pre_infusion_duration_seconds": self.coordinator.realtime_pre_infusion_duration,
                "extraction_uniformity_metric": self.coordinator.realtime_extraction_uniformity,
                "shot_quality_score": round(
                    self.coordinator.realtime_shot_quality_score, 1
                )
                if self.coordinator.realtime_shot_quality_score is not None
                else None,
            }
            try:
                validated_event_data = BookooShotCompletedEventDataModel(
                    **raw_event_data
                )
                self.last_shot_data = validated_event_data
                self.coordinator.last_shot_data = validated_event_data.model_copy(
                    deep=True
                )
            except ValidationError as e:
                _LOGGER.error(
                    "%s: Shot data validation failed for aborted_too_short: %s. Data: %s",
                    self.coordinator.name,
                    e,
                    raw_event_data,
                )
                # Decide how to handle: maybe return, or fire a minimal error event
                self._reset_internal_session_state()
                self.coordinator.async_update_listeners()
                return

            _LOGGER.info(
                "%s: Attempting to save 'aborted_too_short' shot record to SQLite.",
                self.coordinator.name,
            )
            assert self.last_shot_data is not None  # Help mypy with type narrowing
            await async_add_shot_record(self.hass, self.last_shot_data.model_dump())

            self._reset_internal_session_state()
            self.coordinator._reset_realtime_analytics()  # Reset analytics on coordinator
            self.coordinator.async_update_listeners()
            return

        final_weight_grams = (
            self.coordinator.scale.weight
            if self.coordinator.scale.weight is not None
            else 0.0
        )

        average_flow_rate_gps = 0.0
        if duration_seconds > 0 and final_weight_grams > 0:
            average_flow_rate_gps = round(final_weight_grams / duration_seconds, 2)

        peak_flow_rate_gps = 0.0
        time_to_peak_flow_seconds = None
        if self.session_flow_profile:
            valid_flow_points = [
                dp for dp in self.session_flow_profile if dp.flow_rate > 0.01
            ]
            if valid_flow_points:
                peak_flow_dp = max(
                    valid_flow_points,
                    key=lambda item: item.flow_rate,
                )
                time_to_peak_flow_seconds = round(peak_flow_dp.elapsed_time, 2)
                peak_flow_rate_gps = round(peak_flow_dp.flow_rate, 2)
            elif (
                self.session_flow_profile
            ):  # if no valid_flow_points but profile exists
                peak_flow_rate_gps = 0.0

        time_to_first_flow_seconds = None
        FIRST_FLOW_THRESHOLD_GPS = 0.2
        if self.session_flow_profile:
            for dp in self.session_flow_profile:
                if dp.flow_rate > FIRST_FLOW_THRESHOLD_GPS:
                    time_to_first_flow_seconds = round(dp.elapsed_time, 2)
                    break

        # TODO: Add auto-stop logic here if applicable, it might change stop_reason and shot_status
        # This would involve checking flow rate stability and cutoff based on coordinator options.
        # Example:
        # if self.coordinator.config_entry.options.get(OPTION_ENABLE_AUTO_STOP_FLOW_CUTOFF):
        #     is_auto_stopped, auto_stop_details = self._check_auto_stop_conditions()
        #     if is_auto_stopped:
        #         stop_reason = "auto_flow_cutoff"
        #         # Potentially update other metrics based on auto-stop

        raw_event_data = {
            "device_id": self.coordinator.config_entry.unique_id
            or self.coordinator.config_entry.entry_id,
            "entry_id": self.coordinator.config_entry.entry_id,
            "start_time_utc": original_start_time_utc_iso,
            "end_time_utc": current_time.isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "status": shot_status,
            "start_trigger": original_start_trigger,
            "stop_reason": stop_reason,
            "final_weight_grams": round(final_weight_grams, 2),
            "flow_profile": self.session_flow_profile,
            "scale_timer_profile": self.session_scale_timer_profile,
            "input_parameters": original_input_params,
            "channeling_status": self.coordinator.realtime_channeling_status,
            "pre_infusion_detected": self.coordinator.realtime_pre_infusion_active,
            "pre_infusion_duration_seconds": self.coordinator.realtime_pre_infusion_duration,
            "extraction_uniformity_metric": self.coordinator.realtime_extraction_uniformity,
            "average_flow_rate_gps": average_flow_rate_gps,
            "peak_flow_rate_gps": peak_flow_rate_gps,
            "time_to_first_flow_seconds": time_to_first_flow_seconds,
            "time_to_peak_flow_seconds": time_to_peak_flow_seconds,
            "shot_quality_score": round(self.coordinator.realtime_shot_quality_score, 1)
            if self.coordinator.realtime_shot_quality_score is not None
            else None,
        }
        try:
            validated_event_data = BookooShotCompletedEventDataModel(**raw_event_data)
            self.last_shot_data = validated_event_data
            self.coordinator.last_shot_data = validated_event_data.model_copy(deep=True)
        except ValidationError as e:
            _LOGGER.error(
                "%s: Shot data validation failed for completed shot: %s. Data: %s",
                self.coordinator.name,
                e,
                raw_event_data,
            )
            # Decide how to handle: maybe return, or fire a minimal error event
            self._reset_internal_session_state()
            self.coordinator.async_update_listeners()
            return

        logged_event_data = {}
        if self.coordinator.last_shot_data:  # Ensure it's not None
            logged_event_data = {
                k: v
                for k, v in self.coordinator.last_shot_data.model_dump().items()
                if k not in ["flow_profile", "scale_timer_profile"]
            }
        _LOGGER.info(
            "%s: Fired EVENT_BOOKOO_SHOT_COMPLETED with (logged) data: %s",
            self.coordinator.name,
            logged_event_data,
        )

        _LOGGER.info(
            "%s: Attempting to save shot record to SQLite using validated event data.",
            self.coordinator.name,
        )
        assert self.coordinator.last_shot_data is not None  # Help mypy
        await async_add_shot_record(
            self.hass, self.coordinator.last_shot_data.model_dump()
        )

        self._reset_internal_session_state()
        self.coordinator._reset_realtime_analytics()  # Reset analytics on coordinator
        self.coordinator.async_update_listeners()

    def add_flow_data(self, elapsed_time: float, flow_rate: float) -> None:
        """Adds a flow rate data point to the current session."""
        if self.is_shot_active:
            self.session_flow_profile.append(
                FlowDataPoint(elapsed_time=elapsed_time, flow_rate=flow_rate)
            )

    def add_weight_data(self, elapsed_time: float, weight: float) -> None:
        """Adds a weight data point to the current session."""
        if self.is_shot_active:
            self.session_weight_profile.append(
                WeightDataPoint(elapsed_time=elapsed_time, weight=weight)
            )

    def add_scale_timer_data(self, elapsed_time: float, timer_value: int) -> None:
        """Adds a scale timer data point to the current session."""
        if self.is_shot_active:
            self.session_scale_timer_profile.append(
                ScaleTimerDataPoint(elapsed_time=elapsed_time, timer_value=timer_value)
            )
