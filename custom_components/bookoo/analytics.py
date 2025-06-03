"""Functions for real-time espresso shot analytics."""

import math

# Placeholder for more sophisticated type hints if needed, e.g., for profile data points
FlowProfile = list[tuple[float, float]]  # (elapsed_seconds, flow_rate_gps)
ScaleTimerProfile = list[tuple[float, int]]  # (elapsed_seconds, scale_timer_seconds)


def detect_channeling(flow_profile: FlowProfile) -> str:
    """
    Detects channeling based on flow rate inconsistencies.
    Considers data after an initial ramp-up phase.

    Args:
        flow_profile: A list of (elapsed_seconds, flow_rate_gps) tuples.

    Returns:
        A string indicating channeling status (e.g., "None", "Mild Channeling", "Suspected Channeling (Spike)").
    """
    if not flow_profile:
        return "Undetermined"

    # Define parameters (these could be made configurable later)
    initial_ignore_seconds = 7.0  # Ignore data from the first X seconds
    min_data_points_for_analysis = (
        10  # Minimum number of data points after ignoring initial phase
    )

    # Filter out the initial phase
    analysis_profile = [dp for dp in flow_profile if dp[0] >= initial_ignore_seconds]

    if len(analysis_profile) < min_data_points_for_analysis:
        return "Undetermined (not enough data after initial phase)"

    flows = [
        dp[1] for dp in analysis_profile if dp[1] is not None and dp[1] > 0.05
    ]  # Consider flows > 0.05 g/s

    if (
        len(flows) < min_data_points_for_analysis
    ):  # Check again after filtering very low flows
        return "Undetermined (not enough significant flow data)"

    mean_flow = sum(flows) / len(flows)
    if mean_flow == 0:  # Should be caught by flows > 0.05, but as a safeguard
        return "None (no significant flow)"

    # Calculate Standard Deviation
    variance = sum([(flow - mean_flow) ** 2 for flow in flows]) / len(flows)
    std_dev = math.sqrt(variance)

    # Calculate Coefficient of Variation (CV)
    coeff_of_variation = std_dev / mean_flow if mean_flow > 0 else 0

    max_flow_in_analysis = 0.0
    if flows:  # Ensure flows is not empty
        max_flow_in_analysis = max(flows)

    status = "None"

    if coeff_of_variation > 0.38:  # Example threshold for high CV
        status = "Mild Channeling (High Variation)"

    if mean_flow > 0.5 and max_flow_in_analysis > mean_flow * 2.2:
        if status == "Mild Channeling (High Variation)":
            status = "Moderate Channeling (High Variation & Spike)"
        else:
            status = "Suspected Channeling (Spike)"
    elif (
        mean_flow > 0.5
        and max_flow_in_analysis > mean_flow * 1.8
        and coeff_of_variation > 0.3
    ):
        if status == "Mild Channeling (High Variation)":
            status = "Mild-Moderate Channeling (Variation & Notable Peak)"

    return status


def identify_pre_infusion(
    flow_profile: FlowProfile,
    scale_timer_profile: ScaleTimerProfile,  # scale_timer_profile unused for now
) -> tuple[bool, float | None]:
    """
    Simplified real-time identification if the shot *might* currently be in a pre-infusion phase.
    Duration is a placeholder for now, as robust duration detection is complex for real-time.
    Returns (is_currently_in_suspected_pi_phase, estimated_pi_duration_if_ended_or_ongoing)
    """
    if not flow_profile or len(flow_profile) < 2:
        return False, None

    current_time, current_flow = flow_profile[-1]

    # Parameters
    pi_flow_threshold = 0.3  # g/s
    max_time_for_pi = 15.0  # seconds (don't consider it PI if shot is longer than this)
    min_shot_time_for_pi_check = 1.0  # seconds (don't check too early)

    is_currently_pre_infusion = False
    estimated_duration: float | None = None

    if min_shot_time_for_pi_check <= current_time <= max_time_for_pi:
        if current_flow <= pi_flow_threshold:
            is_currently_pre_infusion = True
            pi_start_time = current_time
            for t, flow in reversed(flow_profile):
                if flow <= pi_flow_threshold:
                    pi_start_time = t
                else:
                    break
            estimated_duration = current_time - pi_start_time

    if estimated_duration is not None and estimated_duration < 1.0:
        estimated_duration = None

    return is_currently_pre_infusion, estimated_duration


def calculate_extraction_uniformity(flow_profile: FlowProfile) -> float:
    """
    Calculates a score for extraction uniformity based on flow profile.
    A higher score (towards 1.0) indicates better uniformity.
    This version focuses on the stability of flow after an initial period.
    """
    if not flow_profile:
        return 0.0

    initial_ignore_seconds = 7.0  # Corresponds to detect_channeling ignore period
    min_data_points_for_analysis = 10

    analysis_profile = [dp for dp in flow_profile if dp[0] >= initial_ignore_seconds]

    if len(analysis_profile) < min_data_points_for_analysis:
        return 0.0

    flows = [
        dp[1] for dp in analysis_profile if dp[1] is not None and dp[1] > 0.1
    ]  # Consider flows > 0.1 g/s for uniformity

    if (
        len(flows) < min_data_points_for_analysis // 2
    ):  # Need a decent number of significant flow points
        return 0.0

    mean_flow = sum(flows) / len(flows)
    if mean_flow == 0:
        return 0.0

    variance = sum([(f - mean_flow) ** 2 for f in flows]) / len(flows)
    std_dev = math.sqrt(variance)

    coeff_of_variation = (
        std_dev / mean_flow if mean_flow > 0 else 1.0
    )  # Assign high CV if mean_flow is 0

    uniformity_score = max(0.0, 1.0 - coeff_of_variation)

    return round(uniformity_score, 2)
