"""Manages shot session logic for the Bookoo integration."""

from __future__ import annotations

import asyncio
import collections
import logging
import statistics
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from pydantic import ValidationError

from .storage import async_add_shot_record
from .types import (
    BookooShotCompletedEventDataModel,
    FlowDataPoint,
    ScaleTimerDataPoint,
    WeightDataPoint,
)

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    # To avoid circular import with BookooCoordinator, we type hint it.
    # Actual instance is passed in __init__.
    from .coordinator import BookooCoordinator

_LOGGER = logging.getLogger(__name__)

MAX_PROFILE_POINTS = 3000  # Max data points for session profiles (e.g., 5 mins at 10Hz)


class SessionManager:
    """Handles the state and logic for an active shot session."""

    def __init__(self, hass: HomeAssistant, coordinator: BookooCoordinator) -> None:
        """Initialize the session manager."""
        self.hass = hass
        self.coordinator = coordinator

        self.is_shot_active: bool = False
        self.session_start_time_utc: datetime | None = None
        self.session_flow_profile: collections.deque[FlowDataPoint] = collections.deque(
            maxlen=MAX_PROFILE_POINTS
        )
        self.session_weight_profile: collections.deque[WeightDataPoint] = (
            collections.deque(maxlen=MAX_PROFILE_POINTS)
        )
        self.session_scale_timer_profile: collections.deque[ScaleTimerDataPoint] = (
            collections.deque(maxlen=MAX_PROFILE_POINTS)
        )
        self.session_input_parameters: dict[str, Any] = {}
        self.session_start_trigger: str | None = None
        self.last_shot_data: BookooShotCompletedEventDataModel | None = (
            None  # This will be assigned in stop_session
        )

        # Auto-stop flow cutoff state
        self._auto_stop_flow_stable_start_time: datetime | None = None
        self._auto_stop_flow_below_cutoff_start_time: datetime | None = None

        self._session_lock = asyncio.Lock()

    def _reset_internal_session_state(self) -> None:
        """Internal helper to reset session-specific state variables."""
        self.is_shot_active = False
        self.session_start_time_utc = None
        self.session_flow_profile = collections.deque(maxlen=MAX_PROFILE_POINTS)
        self.session_weight_profile = collections.deque(maxlen=MAX_PROFILE_POINTS)
        self.session_scale_timer_profile = collections.deque(maxlen=MAX_PROFILE_POINTS)
        self.session_input_parameters = {}
        self.session_start_trigger = None
        self._auto_stop_flow_stable_start_time = None
        self._auto_stop_flow_below_cutoff_start_time = None
        # self.last_shot_data is intentionally not cleared here

    def _read_linked_input_to_params(
        self, entity_id: str | None, param_key: str, param_description: str
    ) -> None:
        """Reads a linked entity's state and adds it to session_input_parameters."""
        if not entity_id:
            return

        entity_state = self.hass.states.get(entity_id)
        if entity_state and entity_state.state not in (
            STATE_UNKNOWN,
            STATE_UNAVAILABLE,
        ):
            self.session_input_parameters[param_key] = entity_state.state
            _LOGGER.debug(
                "%s: Logged %s: %s from %s",
                self.coordinator.name,
                param_description,
                entity_state.state,
                entity_id,
            )
        else:
            _LOGGER.warning(
                "%s: Could not read state for linked %s entity: %s",
                self.coordinator.name,
                param_description,
                entity_id,
            )

    async def start_session(self, trigger: str) -> None:
        """Starts a new shot session.

        Args:
            trigger: A string describing what triggered the shot start (e.g., 'service', 'auto_timer').
        """
        async with self._session_lock:
            if self.is_shot_active:
                _LOGGER.warning(
                    "Attempted to start new shot (trigger: %s) but one is active. This will now raise an error to the user.",
                    trigger,
                )
                raise HomeAssistantError(translation_key="shot_already_active")

            _LOGGER.info(
                "%s: Starting new shot session, triggered by: %s",
                self.coordinator.name,
                trigger,
            )
            self._reset_internal_session_state()  # Clear previous session data before starting new
            self.is_shot_active = True
            self.session_start_time_utc = dt_util.utcnow()
            self.session_start_trigger = trigger
            self._auto_stop_flow_stable_start_time = None
            self._auto_stop_flow_below_cutoff_start_time = None

            self._read_linked_entities()

        # Coordinator will handle resetting its own realtime analytics and updating listeners
        self.coordinator.async_update_listeners()  # Notify HA of state change (shot active)

    def _read_linked_entities(self) -> None:
        """Reads linked entities and adds their states to session_input_parameters."""
        self._read_linked_input_to_params(
            self.coordinator.bookoo_config.linked_bean_weight_entity,
            "bean_weight",
            "bean weight",
        )
        self._read_linked_input_to_params(
            self.coordinator.bookoo_config.linked_coffee_name_entity,
            "coffee_name",
            "coffee name",
        )
        self._read_linked_input_to_params(
            self.coordinator.bookoo_config.linked_grind_setting_entity,
            "grind_setting",
            "grind setting",
        )
        self._read_linked_input_to_params(
            self.coordinator.bookoo_config.linked_brew_temperature_entity,
            "brew_temperature",
            "brew temperature",
        )

    def _determine_shot_status_and_duration(
        self,
        stop_reason: str,
        current_session_start_time_utc: datetime,
        current_time: datetime,
    ) -> tuple[str, float]:
        """Determines the shot status and duration."""
        duration_seconds = (
            current_time - current_session_start_time_utc
        ).total_seconds()

        config = self.coordinator.bookoo_config
        min_duration = config.min_shot_duration
        max_duration = config.max_shot_duration

        shot_status = "completed"  # Default

        if stop_reason == "disconnected":
            shot_status = "aborted_disconnected"
        elif max_duration > 0 and duration_seconds > max_duration:
            _LOGGER.info(
                "%s: Shot duration (%.2f s) > max configured (%s s). Aborting.",
                self.coordinator.name,
                duration_seconds,
                max_duration,
            )
            shot_status = "aborted_too_long"
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
        return shot_status, duration_seconds

    def _prepare_minimal_shot_data(self, context: dict[str, Any]) -> dict[str, Any]:
        """Prepares a minimal data dictionary for aborted shots."""
        return {
            "device_id": self.coordinator.config_entry.unique_id
            or self.coordinator.config_entry.entry_id,
            "entry_id": self.coordinator.config_entry.entry_id,
            "start_time_utc": context["original_start_time_utc_iso"],
            "end_time_utc": context["current_time_iso"],
            "duration_seconds": round(context["duration_seconds"], 2),
            "status": context["shot_status"],
            "start_trigger": context["original_start_trigger"],
            "stop_reason": context["stop_reason"],
            "input_parameters": context["original_input_params"],
            "final_weight_grams": 0.0,
            "flow_profile": [],
            "scale_timer_profile": [],
            "average_flow_rate_gps": 0.0,
            "peak_flow_rate_gps": 0.0,
            "time_to_first_flow_seconds": None,
            "time_to_peak_flow_seconds": None,
            "channeling_status": self.coordinator.realtime_channeling_status,
            "pre_infusion_detected": self.coordinator.realtime_pre_infusion_active,
            "pre_infusion_duration_seconds": self.coordinator.realtime_pre_infusion_duration,
            "extraction_uniformity_metric": self.coordinator.realtime_extraction_uniformity,
            "shot_quality_score": round(
                self.coordinator.realtime_shot_quality_score or 0.0, 1
            ),
            "weight_profile": [],  # Added for completeness, though minimal
        }

    def _calculate_shot_analytics(
        self,
    ) -> dict[str, Any]:
        """Calculates detailed analytics for a completed shot."""
        analyzer = self.coordinator.shot_analyzer
        final_weight = (
            self.session_weight_profile[-1].weight
            if self.session_weight_profile
            else 0.0
        )

        flow_profile_list = list(self.session_flow_profile)

        avg_flow = analyzer.calculate_average_flow_rate(flow_profile_list)
        peak_flow = analyzer.calculate_peak_flow_rate(flow_profile_list)
        time_to_first_flow = analyzer.calculate_time_to_first_flow(flow_profile_list)
        time_to_peak_flow = analyzer.calculate_time_to_peak_flow(flow_profile_list)
        return {
            "final_weight_grams": round(final_weight, 2),
            "average_flow_rate_gps": avg_flow,
            "peak_flow_rate_gps": peak_flow,
            "time_to_first_flow_seconds": time_to_first_flow,
            "time_to_peak_flow_seconds": time_to_peak_flow,
        }

    def _prepare_completed_shot_data(
        self, analytics: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepares a full data dictionary for completed shots."""
        return {
            "device_id": self.coordinator.config_entry.unique_id
            or self.coordinator.config_entry.entry_id,
            "entry_id": self.coordinator.config_entry.entry_id,
            "start_time_utc": context["original_start_time_utc_iso"],
            "end_time_utc": context["current_time_iso"],
            "duration_seconds": round(context["duration_seconds"], 2),
            "status": context["shot_status"],
            "start_trigger": context["original_start_trigger"],
            "stop_reason": context["stop_reason"],
            "input_parameters": context["original_input_params"],
            "final_weight_grams": analytics["final_weight_grams"],
            "flow_profile": [
                datapoint._asdict() for datapoint in self.session_flow_profile
            ],
            "scale_timer_profile": [
                datapoint._asdict() for datapoint in self.session_scale_timer_profile
            ],
            "weight_profile": [
                datapoint._asdict() for datapoint in self.session_weight_profile
            ],
            "average_flow_rate_gps": analytics.get(
                "average_flow_rate_gps", 0.0
            ),  # Use .get with default
            "peak_flow_rate_gps": analytics.get(
                "peak_flow_rate_gps", 0.0
            ),  # Use .get with default
            "time_to_first_flow_seconds": analytics.get(
                "time_to_first_flow_seconds"
            ),  # Use .get with default (None)
            "time_to_peak_flow_seconds": analytics.get(
                "time_to_peak_flow_seconds"
            ),  # Use .get with default (None)
            "channeling_status": self.coordinator.realtime_channeling_status,
            "pre_infusion_detected": self.coordinator.realtime_pre_infusion_active,
            "pre_infusion_duration_seconds": self.coordinator.realtime_pre_infusion_duration,
            "extraction_uniformity_metric": self.coordinator.realtime_extraction_uniformity,
            "shot_quality_score": round(
                self.coordinator.realtime_shot_quality_score or 0.0, 1
            ),
        }

    def _calculate_final_shot_metrics(
        self,
        duration_seconds: float,
        final_weight_grams: float,
        flow_profile: collections.deque[FlowDataPoint],
    ) -> dict[str, Any]:
        """Calculates final metrics for a completed shot."""
        metrics: dict[str, Any] = {
            "average_flow_rate_gps": 0.0,
            "peak_flow_rate_gps": 0.0,
            "time_to_peak_flow_seconds": None,
            "time_to_first_flow_seconds": None,
        }

        if duration_seconds > 0 and final_weight_grams > 0:
            metrics["average_flow_rate_gps"] = round(
                final_weight_grams / duration_seconds, 2
            )

        if flow_profile:
            # Peak flow
            # Ensure there are points with flow_rate > 0.01 before calling max()
            valid_flow_points = [dp for dp in flow_profile if dp.flow_rate > 0.01]
            if valid_flow_points:
                peak_flow_dp = max(valid_flow_points, key=lambda item: item.flow_rate)
                metrics["time_to_peak_flow_seconds"] = round(
                    peak_flow_dp.elapsed_time, 2
                )
                metrics["peak_flow_rate_gps"] = round(peak_flow_dp.flow_rate, 2)
            # If no valid_flow_points, peak_flow_rate_gps remains 0.0 as initialized

            # Time to first flow
            FIRST_FLOW_THRESHOLD_GPS = 0.2
            for dp in flow_profile:
                if dp.flow_rate > FIRST_FLOW_THRESHOLD_GPS:
                    metrics["time_to_first_flow_seconds"] = round(dp.elapsed_time, 2)
                    break
        return metrics

    async def _finalize_and_store_shot(
        self, raw_event_data: dict[str, Any], shot_status: str
    ) -> None:
        """Validates, stores, and dispatches the shot data."""
        try:
            # Add entry_id to the raw_event_data from the coordinator's config entry
            raw_event_data["entry_id"] = self.coordinator.config_entry.entry_id

            validated_shot_data = BookooShotCompletedEventDataModel(
                **dict(raw_event_data)
            )
            self.last_shot_data = validated_shot_data
            _LOGGER.debug(
                "%s: Shot data prepared for event and storage: %s",
                self.coordinator.name,
                validated_shot_data.model_dump(
                    mode="json"
                ),  # Use model_dump for Pydantic models
            )

            storage_attempted = False
            storage_successful = False

            if shot_status != "aborted_too_short":
                storage_attempted = True
                try:
                    await async_add_shot_record(self.hass, validated_shot_data)
                    storage_successful = True
                    _LOGGER.info(
                        "%s: Shot (status: %s) data stored successfully.",
                        self.coordinator.name,
                        shot_status,
                    )
                except Exception as e:
                    _LOGGER.error(
                        "%s: Failed to store shot record (status: %s): %s. Event will still be fired.",
                        self.coordinator.name,
                        shot_status,
                        e,
                        exc_info=True,
                    )
            else:
                _LOGGER.info(
                    "%s: Shot was aborted too short (status: %s), not saving to history.",
                    self.coordinator.name,
                    shot_status,
                )
                # For aborted_too_short, we didn't attempt storage, so consider it 'successful' for event firing logic
                storage_successful = True

            # Fire event with validated data if storage was successful or not attempted (for aborted_too_short)
            if storage_successful:
                self.hass.bus.async_fire(
                    f"{self.coordinator.config_entry.domain}_shot_completed",
                    validated_shot_data.model_dump(mode="json"),
                )
                _LOGGER.debug(
                    "%s: Fired %s event with data: %s (Storage attempted: %s, Storage successful: %s)",
                    self.coordinator.name,
                    f"{self.coordinator.config_entry.domain}_shot_completed",
                    validated_shot_data.model_dump(mode="json"),
                    storage_attempted,
                    storage_successful if storage_attempted else "N/A",
                )
            else:
                # This case means storage was attempted but failed
                _LOGGER.warning(
                    "%s: Shot event NOT fired due to storage failure for shot (status: %s) starting %s.",
                    self.coordinator.name,
                    shot_status,
                    validated_shot_data.start_time_utc,
                )

        except ValidationError as e:
            _LOGGER.error(
                "%s: Validation error preparing shot data: %s. Raw data: %s",
                self.coordinator.name,
                e,
                raw_event_data,
                exc_info=True,
            )
        except Exception as e:  # pylint: disable=broad-except
            _LOGGER.error(
                "%s: Unexpected error finalizing shot: %s",
                self.coordinator.name,
                e,
                exc_info=True,
            )

    async def stop_session(self, stop_reason: str) -> None:
        async with self._session_lock:
            """Stops the current shot session, calculates metrics, stores data, and fires an event.

            Args:
                stop_reason: A string describing why the shot was stopped (e.g., 'service', 'disconnected', 'auto_flow_cutoff').
            """
            if not self.is_shot_active or not self.session_start_time_utc:
                _LOGGER.debug(
                    "%s: Stop session called but no active session or start time found.",
                    self.coordinator.name,
                )
                return

            current_session_start_time_utc = self.session_start_time_utc
            current_time = dt_util.utcnow()

            shot_status, duration_seconds = self._determine_shot_status_and_duration(
                stop_reason, current_session_start_time_utc, current_time
            )

            _LOGGER.info(
                "%s: Stopping shot session (reason: %s). Status: %s. Duration: %.2f seconds.",
                self.coordinator.name,
                stop_reason,
                shot_status,
                duration_seconds,
            )

            # Prepare base data for the event
            start_time_iso = current_session_start_time_utc.isoformat()
            end_time_iso = current_time.isoformat()

            raw_event_data: dict[str, Any] = {
                "device_id": self.coordinator.config_entry.unique_id
                or self.coordinator.config_entry.entry_id,
                "unique_shot_id": f"{start_time_iso}_{self.coordinator.config_entry.unique_id or self.coordinator.config_entry.entry_id}",
                "start_time_utc": start_time_iso,
                "end_time_utc": end_time_iso,
                "duration_seconds": duration_seconds,
                "status": shot_status,
                "start_trigger": self.session_start_trigger,
                "stop_reason": stop_reason,
                "input_parameters": dict(self.session_input_parameters),
                "channeling_status": self.coordinator.realtime_channeling_status,
                "pre_infusion_detected": self.coordinator.realtime_pre_infusion_active,
                "pre_infusion_duration_seconds": self.coordinator.realtime_pre_infusion_duration,
                "extraction_uniformity_metric": self.coordinator.realtime_extraction_uniformity,
                "shot_quality_score": round(
                    self.coordinator.realtime_shot_quality_score or 0.0, 1
                )
                if self.coordinator.realtime_shot_quality_score is not None
                else None,
            }

        final_weight_grams = (
            self.coordinator.scale.weight
            if self.coordinator.scale.weight is not None
            else 0.0
        )
        raw_event_data["final_weight_grams"] = round(final_weight_grams, 2)

        if shot_status == "aborted_too_short":
            # For aborted shots, the base raw_event_data is likely sufficient.
            # If _prepare_minimal_shot_data was intended to add more specific keys for aborted shots,
            # it could be called here and its result merged into raw_event_data.
            # Example: raw_event_data.update(self._prepare_minimal_shot_data(raw_event_data.copy()))
            pass
        else:
            # For completed or other non-aborted_too_short statuses
            analytics = self._calculate_shot_analytics()
            raw_event_data.update(analytics)

            final_summary_metrics = self._calculate_final_shot_metrics(
                duration_seconds, final_weight_grams, self.session_flow_profile
            )
            raw_event_data.update(final_summary_metrics)

            raw_event_data["flow_profile"] = list(self.session_flow_profile)
            raw_event_data["weight_profile"] = list(self.session_weight_profile)
            raw_event_data["scale_timer_profile"] = list(
                self.session_scale_timer_profile
            )

        # Auto-stop logic (e.g., by flow cutoff) is now handled proactively in add_flow_data.
        # This ensures stop_session is called with the correct reason when conditions are met.

        await self._finalize_and_store_shot(raw_event_data, shot_status)

        self._reset_internal_session_state()
        self.coordinator._reset_realtime_analytics()
        self.coordinator.async_update_listeners()

    def _check_auto_stop_flow_cutoff(
        self, current_elapsed_time: float, current_flow_rate: float
    ) -> None:
        """Checks and triggers auto-stop based on flow cutoff conditions."""
        if (
            not self.is_shot_active
            or not self.coordinator.bookoo_config.enable_auto_stop_flow_cutoff
        ):
            return

        config = self.coordinator.bookoo_config
        now = dt_util.utcnow()

        # Ignore initial phase
        if current_elapsed_time < config.auto_stop_pre_infusion_ignore_duration:
            # Reset stability and cutoff timers if we are still in ignore phase
            self._auto_stop_flow_stable_start_time = None
            self._auto_stop_flow_below_cutoff_start_time = None
            return

        # --- Flow Stability Check ---
        if self._auto_stop_flow_stable_start_time is None:
            # Check if flow is currently above the minimum for stability
            if current_flow_rate >= config.auto_stop_min_flow_for_stability:
                # Potential start of stable phase, gather recent data points for CV calculation
                # We need enough data points that occurred *after* pre_infusion_ignore_duration
                relevant_flow_points = [
                    dp.flow_rate
                    for dp in self.session_flow_profile
                    if dp.elapsed_time >= config.auto_stop_pre_infusion_ignore_duration
                    and dp.flow_rate is not None
                    and dp.flow_rate >= config.auto_stop_min_flow_for_stability
                ]

                # Ensure we have enough data points and they span the min_duration_for_stability
                if (
                    relevant_flow_points and len(relevant_flow_points) >= 3
                ):  # Need at least 2 for stdev, 3 for better measure
                    first_relevant_dp_time = next(
                        (
                            dp.elapsed_time
                            for dp in self.session_flow_profile
                            if dp.elapsed_time
                            >= config.auto_stop_pre_infusion_ignore_duration
                            and dp.flow_rate is not None
                            and dp.flow_rate >= config.auto_stop_min_flow_for_stability
                        ),
                        current_elapsed_time,
                    )
                    duration_of_relevant_flow = (
                        current_elapsed_time - first_relevant_dp_time
                    )

                    if (
                        duration_of_relevant_flow
                        >= config.auto_stop_min_duration_for_stability
                    ):
                        mean_flow = statistics.mean(relevant_flow_points)
                        std_dev = (
                            statistics.stdev(relevant_flow_points)
                            if len(relevant_flow_points) > 1
                            else 0
                        )
                        cv = (
                            (std_dev / mean_flow) * 100
                            if mean_flow > 0
                            else float("inf")
                        )

                        if cv <= config.auto_stop_max_flow_variance_for_stability:
                            _LOGGER.debug(
                                "%s: Auto-stop flow considered stable. CV: %.2f%%, Mean: %.2fg/s, Duration: %.2fs",
                                self.coordinator.name,
                                cv,
                                mean_flow,
                                duration_of_relevant_flow,
                            )
                            self._auto_stop_flow_stable_start_time = now
                        # else: # Flow is not stable yet, wait for more data
                    # else: # Not enough duration of stable flow yet
                # else: # Not enough data points for stability check yet
            # else: # Flow is below stability threshold, reset stability timer
            #    self._auto_stop_flow_stable_start_time = None # This would reset on any dip, might be too sensitive. Let's only set it once stable.

        # --- Flow Cutoff Check (only if flow has been stable) ---
        if self._auto_stop_flow_stable_start_time is not None:
            if current_flow_rate < config.auto_stop_flow_cutoff_threshold:
                if self._auto_stop_flow_below_cutoff_start_time is None:
                    self._auto_stop_flow_below_cutoff_start_time = now

                if now - self._auto_stop_flow_below_cutoff_start_time >= timedelta(
                    seconds=config.auto_stop_min_duration_for_cutoff
                ):
                    _LOGGER.info(
                        "%s: Auto-stopping shot due to flow cutoff. Flow %.2fg/s < threshold %.2fg/s for %.2fs",
                        self.coordinator.name,
                        current_flow_rate,
                        config.auto_stop_flow_cutoff_threshold,
                        config.auto_stop_min_duration_for_cutoff,
                    )
                    self.hass.async_create_task(
                        self.stop_session(stop_reason="auto_stop_flow_cutoff")
                    )
            else:
                # Flow went back above cutoff, reset the cutoff timer
                self._auto_stop_flow_below_cutoff_start_time = None

    def add_flow_data(self, elapsed_time: float, flow_rate: float) -> None:
        """Adds a flow rate data point to the current session's profile.

        Args:
            elapsed_time: The time in seconds since the shot started.
            flow_rate: The calculated flow rate in g/s.
        """
        """Adds a flow rate data point to the current session."""
        if self.is_shot_active:
            self.session_flow_profile.append(
                FlowDataPoint(
                    elapsed_time=round(elapsed_time, 2), flow_rate=round(flow_rate, 2)
                )
            )
            # Only check auto-stop if the shot is active, as flow data is only relevant then
            self._check_auto_stop_flow_cutoff(elapsed_time, flow_rate)

    def add_weight_data(self, elapsed_time: float, weight: float) -> None:
        """Adds a weight data point to the current session's profile.

        Args:
            elapsed_time: The time in seconds since the shot started.
            weight: The current weight reading in grams.
        """
        """Adds a weight data point to the current session."""
        if self.is_shot_active:
            self.session_weight_profile.append(
                WeightDataPoint(elapsed_time=elapsed_time, weight=weight)
            )

    def add_scale_timer_data(self, elapsed_time: float, timer_value: int) -> None:
        """Adds a scale timer data point to the current session's profile.

        Args:
            elapsed_time: The time in seconds since the shot started.
            timer_value: The timer value from the scale in seconds.
        """
        """Adds a scale timer data point to the current session."""
        if self.is_shot_active:
            self.session_scale_timer_profile.append(
                ScaleTimerDataPoint(elapsed_time=elapsed_time, timer_value=timer_value)
            )
