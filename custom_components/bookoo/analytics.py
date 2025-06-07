"""Class-based analytics for real-time espresso shot analysis."""

import math
from dataclasses import dataclass

from .types import (
    FlowProfile,
    ScaleTimerProfile,
    BookooShotCompletedEventDataModel,
)


@dataclass
class AnalyticsConfig:
    """Configuration parameters for shot analytics."""

    # Channeling detection parameters
    channeling_initial_ignore_seconds: float = 7.0
    channeling_min_data_points: int = 10
    channeling_significant_flow_threshold: float = 0.05
    channeling_cv_threshold_mild: float = 0.38
    channeling_spike_factor_moderate: float = 1.8  # For mild-moderate spike check
    channeling_spike_factor_high: float = 2.2  # For more definitive spike check
    channeling_spike_mean_flow_threshold: float = (
        0.5  # Min mean flow to consider spike detection
    )
    channeling_spike_cv_threshold_moderate: float = (
        0.3  # CV threshold for mild-moderate spike
    )

    # Pre-infusion identification parameters
    pi_flow_threshold: float = 0.3  # g/s, flow rate below which might be PI
    pi_max_time: float = (
        15.0  # seconds, don't consider it PI if shot is longer than this
    )
    pi_min_shot_time_check: float = 1.0  # seconds, don't check too early in the shot
    pi_min_duration_meaningful: float = (
        1.0  # seconds, PI duration less than this is ignored
    )

    # Extraction uniformity parameters
    uniformity_initial_ignore_seconds: float = 7.0
    uniformity_min_data_points: int = 10
    uniformity_significant_flow_threshold: float = 0.1  # g/s
    uniformity_min_significant_flow_points_ratio: float = (
        0.5  # Ratio of min_data_points
    )


class ShotAnalyzer:
    """Analyzes espresso shot data using a given configuration."""

    def __init__(self, config: AnalyticsConfig | None = None) -> None:
        """Initialize the ShotAnalyzer with an AnalyticsConfig."""
        self.config = config or AnalyticsConfig()

    def detect_channeling(self, flow_profile: FlowProfile) -> str:
        """
        Detects channeling based on flow rate inconsistencies.
        Considers data after an initial ramp-up phase.
        """
        if not flow_profile:
            return "Undetermined"

        analysis_profile = [
            dp
            for dp in flow_profile
            if dp.elapsed_time >= self.config.channeling_initial_ignore_seconds
        ]

        if len(analysis_profile) < self.config.channeling_min_data_points:
            return "Undetermined (not enough data after initial phase)"

        flows = [
            dp.flow_rate
            for dp in analysis_profile
            if dp.flow_rate is not None
            and dp.flow_rate > self.config.channeling_significant_flow_threshold
        ]

        if len(flows) < self.config.channeling_min_data_points:
            return "Undetermined (not enough significant flow data)"

        mean_flow = sum(flows) / len(flows)
        if mean_flow == 0:
            return "None (no significant flow)"

        variance = sum([(flow - mean_flow) ** 2 for flow in flows]) / len(flows)
        std_dev = math.sqrt(variance)
        coeff_of_variation = std_dev / mean_flow if mean_flow > 0 else 0
        max_flow_in_analysis = max(flows) if flows else 0.0

        status = "None"

        if coeff_of_variation > self.config.channeling_cv_threshold_mild:
            status = "Mild Channeling (High Variation)"

        if (
            mean_flow > self.config.channeling_spike_mean_flow_threshold
            and max_flow_in_analysis
            > mean_flow * self.config.channeling_spike_factor_high
        ):
            if status == "Mild Channeling (High Variation)":
                status = "Moderate Channeling (High Variation & Spike)"
            else:
                status = "Suspected Channeling (Spike)"
        elif (
            mean_flow > self.config.channeling_spike_mean_flow_threshold
            and max_flow_in_analysis
            > mean_flow * self.config.channeling_spike_factor_moderate
            and coeff_of_variation > self.config.channeling_spike_cv_threshold_moderate
        ):
            # This condition implies a notable peak combined with existing variation
            if (
                status == "Mild Channeling (High Variation)"
            ):  # Enhances existing mild channeling
                status = "Mild-Moderate Channeling (Variation & Notable Peak)"
            # If status was "None", this specific combination might not warrant a direct jump
            # to "Mild-Moderate" without the base CV being high. The logic prioritizes CV first.

        return status

    def identify_pre_infusion(
        self,
        flow_profile: FlowProfile,
        scale_timer_profile: ScaleTimerProfile,  # Currently unused, but kept for signature consistency
    ) -> tuple[bool, float | None]:
        """
        Simplified real-time identification if the shot *might* currently be in a pre-infusion phase.
        """
        if not flow_profile or len(flow_profile) < 2:
            return False, None

        current_dp = flow_profile[-1]
        current_time = current_dp.elapsed_time
        current_flow = current_dp.flow_rate
        is_currently_pre_infusion = False
        estimated_duration: float | None = None

        if (
            self.config.pi_min_shot_time_check
            <= current_time
            <= self.config.pi_max_time
        ):
            if current_flow <= self.config.pi_flow_threshold:
                is_currently_pre_infusion = True
                # Try to find when this low-flow phase started
                pi_phase_start_time = current_time
                for dp_reversed in reversed(flow_profile):
                    if dp_reversed.flow_rate <= self.config.pi_flow_threshold:
                        pi_phase_start_time = dp_reversed.elapsed_time
                    else:
                        # Flow went above threshold, so PI ended before this point or at this point
                        break
                estimated_duration = current_time - pi_phase_start_time

        if (
            estimated_duration is not None
            and estimated_duration < self.config.pi_min_duration_meaningful
        ):
            estimated_duration = None  # Not a meaningful PI duration
            is_currently_pre_infusion = (
                False  # If duration too short, not considered PI
            )

        return is_currently_pre_infusion, estimated_duration

    def calculate_extraction_uniformity(self, flow_profile: FlowProfile) -> float:
        """
        Calculates a score for extraction uniformity based on flow profile.
        A higher score (towards 1.0) indicates better uniformity.
        """
        if not flow_profile:
            return 0.0

        analysis_profile = [
            dp
            for dp in flow_profile
            if dp.elapsed_time >= self.config.uniformity_initial_ignore_seconds
        ]

        if len(analysis_profile) < self.config.uniformity_min_data_points:
            return 0.0

        flows = [
            dp.flow_rate
            for dp in analysis_profile
            if dp.flow_rate is not None
            and dp.flow_rate > self.config.uniformity_significant_flow_threshold
        ]

        min_required_flow_points = int(
            self.config.uniformity_min_data_points
            * self.config.uniformity_min_significant_flow_points_ratio
        )
        if len(flows) < min_required_flow_points:
            return 0.0

        mean_flow = sum(flows) / len(flows)
        if mean_flow == 0:
            return 0.0  # Or handle as extremely non-uniform if preferred

        variance = sum([(f - mean_flow) ** 2 for f in flows]) / len(flows)
        std_dev = math.sqrt(variance)

        coeff_of_variation = std_dev / mean_flow if mean_flow > 0 else 1.0
        uniformity_score = max(0.0, 1.0 - coeff_of_variation)

        return round(uniformity_score, 2)

    def calculate_average_flow_rate(self, flow_profile: FlowProfile) -> float:
        """Calculates the average flow rate from the flow profile."""
        if not flow_profile:
            return 0.0

        flows = [
            dp.flow_rate
            for dp in flow_profile
            if dp.flow_rate is not None and dp.flow_rate > 0
        ]  # Consider only positive flow
        if not flows:
            return 0.0

        return round(sum(flows) / len(flows), 2)

    def calculate_peak_flow_rate(self, flow_profile: FlowProfile) -> float:
        """Calculates the peak flow rate from the flow profile."""
        if not flow_profile:
            return 0.0

        flows = [dp.flow_rate for dp in flow_profile if dp.flow_rate is not None]
        if not flows:
            return 0.0

        return round(max(flows), 2)

    def calculate_time_to_first_flow(self, flow_profile: FlowProfile) -> float | None:
        """Calculates the time to the first significant flow."""
        significant_flow_threshold = self.config.channeling_significant_flow_threshold

        for dp in flow_profile:
            if dp.flow_rate is not None and dp.flow_rate > significant_flow_threshold:
                return round(dp.elapsed_time, 2)
        return None

    def calculate_time_to_peak_flow(self, flow_profile: FlowProfile) -> float | None:
        """Calculates the time to reach the peak flow rate."""
        if not flow_profile:
            return None

        peak_flow = -1.0
        time_at_peak_flow: float | None = None

        valid_flow_points = [dp for dp in flow_profile if dp.flow_rate is not None]
        if not valid_flow_points:
            return None

        for dp in valid_flow_points:
            if dp.flow_rate > peak_flow:
                peak_flow = dp.flow_rate
                time_at_peak_flow = dp.elapsed_time

        return round(time_at_peak_flow, 2) if time_at_peak_flow is not None else None

    def analyze_shot_profile(
        self,
        flow_profile: FlowProfile,
        scale_timer_profile: ScaleTimerProfile,  # Kept for consistency, used by identify_pre_infusion
    ) -> dict[str, any]:
        """Analyzes the complete shot profile and returns a dictionary of metrics."""
        channeling_status = self.detect_channeling(flow_profile)
        pre_infusion_detected, pre_infusion_duration_seconds = (
            self.identify_pre_infusion(flow_profile, scale_timer_profile)
        )
        extraction_uniformity = self.calculate_extraction_uniformity(flow_profile)
        average_flow_rate = self.calculate_average_flow_rate(flow_profile)
        peak_flow_rate = self.calculate_peak_flow_rate(flow_profile)
        time_to_first_flow = self.calculate_time_to_first_flow(flow_profile)
        time_to_peak_flow = self.calculate_time_to_peak_flow(flow_profile)

        # Calculate shot_quality_score (0-100)
        # Base score from extraction uniformity (up to 70 points)
        quality_from_uniformity = (extraction_uniformity or 0.0) * 70.0

        # Penalty from channeling (deducted from a base of 30 points)
        base_score_for_no_channeling = 30.0
        channeling_penalty = 0.0
        if channeling_status == "Mild Channeling (High Variation)":
            channeling_penalty = 10.0
        elif channeling_status == "Suspected Channeling (Spike)":
            channeling_penalty = 15.0
        elif channeling_status in [
            "Moderate Channeling (High Variation & Spike)",
            "Mild-Moderate Channeling (Variation & Notable Peak)",
        ]:
            channeling_penalty = 30.0  # Max penalty for severe channeling
        elif (
            channeling_status == "Undetermined (not enough data after initial phase)"
            or channeling_status == "Undetermined (not enough significant flow data)"
        ):
            channeling_penalty = 5.0  # Small penalty for undetermined status
        # "None (no significant flow)" or "None" implies no channeling, so 0 penalty.

        score_after_channeling = base_score_for_no_channeling - channeling_penalty

        shot_quality_score = quality_from_uniformity + score_after_channeling
        # Clamp score between 0 and 100, round to one decimal place
        shot_quality_score = max(0.0, min(100.0, round(shot_quality_score, 1)))

        return {
            "channeling_status": channeling_status,
            "pre_infusion_detected": pre_infusion_detected,
            "pre_infusion_duration_seconds": pre_infusion_duration_seconds,
            "extraction_uniformity_metric": extraction_uniformity,
            "average_flow_rate_gps": average_flow_rate,
            "peak_flow_rate_gps": peak_flow_rate,
            "time_to_first_flow_seconds": time_to_first_flow,
            "time_to_peak_flow_seconds": time_to_peak_flow,
            "shot_quality_score": shot_quality_score,
        }

    def generate_next_shot_recommendation(
        self,
        completed_shot_data: "BookooShotCompletedEventDataModel | None",
    ) -> str | None:
        """Generates a textual recommendation for the next shot based on the last one.

        Args:
            completed_shot_data: The data model of the completed shot.

        Returns:
            A string recommendation, or None if no specific recommendation is generated.
        """
        if not completed_shot_data:
            return "No previous shot data to analyze."

        recommendations = []
        shot = completed_shot_data

        # Duration feedback
        if shot.duration_seconds < 18:
            recommendations.append(
                "Shot was very fast (under 18s). Aim for a longer extraction. Try a finer grind or increase dose."
            )
        elif shot.duration_seconds < 22:
            recommendations.append(
                "Shot was a bit fast (under 22s). Consider a slightly finer grind or a small dose increase."
            )
        elif shot.duration_seconds > 35:
            recommendations.append(
                "Shot was very long (over 35s). Aim for a shorter extraction. Try a coarser grind or decrease dose."
            )
        elif shot.duration_seconds > 30:
            recommendations.append(
                "Shot was a bit long (over 30s). Consider a slightly coarser grind or a small dose decrease."
            )
        else:
            recommendations.append(
                "Shot duration (%.1fs) was in a good range (22-30s)."
                % shot.duration_seconds
            )

        # Channeling feedback
        if (
            shot.channeling_status
            and "None" not in shot.channeling_status
            and "Undetermined" not in shot.channeling_status
        ):
            recommendations.append(
                f"Detected {shot.channeling_status.lower()}. Focus on puck preparation: ensure even distribution and tamping."
            )
        elif shot.channeling_status == "None (no significant flow)":
            recommendations.append(
                "No significant flow detected, check for a choked machine or very fine grind."
            )
        else:
            recommendations.append(
                "Good puck preparation, no significant channeling detected."
            )

        # Pre-infusion feedback
        if (
            shot.pre_infusion_detected
            and shot.pre_infusion_duration_seconds is not None
        ):
            if shot.pre_infusion_duration_seconds < 3:
                recommendations.append(
                    "Pre-infusion was short (%.1fs). If intended, this is fine."
                    % shot.pre_infusion_duration_seconds
                )
            elif shot.pre_infusion_duration_seconds > 10:
                recommendations.append(
                    "Pre-infusion was long (%.1fs). Ensure this is intended for your recipe."
                    % shot.pre_infusion_duration_seconds
                )
            else:
                recommendations.append(
                    "Pre-infusion time (%.1fs) seems reasonable."
                    % shot.pre_infusion_duration_seconds
                )

        # Final weight - this is highly dependent on target, so generic advice
        # Example: Assuming a 1:2 ratio and input dose might be around 18g, target ~36g
        if shot.input_parameters and shot.input_parameters.get("bean_weight"):
            try:
                dose = float(shot.input_parameters.get("bean_weight"))
                target_yield_min = dose * 1.8
                target_yield_max = dose * 2.2
                if shot.final_weight_grams < target_yield_min:
                    recommendations.append(
                        "Yield (%.1fg) was low for a %.1fg dose. Consider extending the shot or adjusting grind for more output."
                        % (shot.final_weight_grams, dose)
                    )
                elif shot.final_weight_grams > target_yield_max:
                    recommendations.append(
                        "Yield (%.1fg) was high for a %.1fg dose. Consider stopping the shot earlier or adjusting grind for less output."
                        % (shot.final_weight_grams, dose)
                    )
            except ValueError:
                pass  # Unable to parse bean_weight

        # Quality Score Feedback
        if shot.shot_quality_score is not None:
            if shot.shot_quality_score >= 85:
                recommendations.append(
                    f"Excellent shot quality score ({shot.shot_quality_score:.0f}/100)!"
                )
            elif shot.shot_quality_score >= 70:
                recommendations.append(
                    f"Good shot quality score ({shot.shot_quality_score:.0f}/100)."
                )
            elif shot.shot_quality_score >= 50:
                recommendations.append(
                    f"Decent shot quality score ({shot.shot_quality_score:.0f}/100), but room for improvement."
                )
            else:
                recommendations.append(
                    f"Low shot quality score ({shot.shot_quality_score:.0f}/100). Review other metrics for clues."
                )

        if not recommendations:
            return "Review all shot parameters and aim for consistency in your process."

        # Combine recommendations, prioritizing more critical ones or summarizing
        # For now, just join them. You might want more sophisticated joining/prioritization.
        final_recommendation = "Next shot tips: " + " ".join(recommendations)
        if len(final_recommendation) > 255:  # HA state character limit
            return final_recommendation[:252] + "..."
        return final_recommendation
