import unittest

from custom_components.bookoo.analytics import (
    detect_channeling,
    identify_pre_infusion,
    calculate_extraction_uniformity,
    FlowProfile,
    ScaleTimerProfile,
)


class TestDetectChanneling(unittest.TestCase):
    def test_no_profile_or_insufficient_data(self: "TestDetectChanneling") -> None:
        empty_profile: FlowProfile = []
        short_profile: FlowProfile = [(1.0, 0.1), (2.0, 0.2)]
        self.assertEqual(detect_channeling(empty_profile), "Undetermined")
        self.assertEqual(
            detect_channeling(short_profile),
            "Undetermined (not enough data after initial phase)",
        )
        profile_all_initial: FlowProfile = [
            (float(i), 0.5 + i * 0.1) for i in range(7)
        ]  # Assumes initial_ignore_seconds = 7.0
        self.assertEqual(
            detect_channeling(profile_all_initial),
            "Undetermined (not enough data after initial phase)",
        )

    def test_clear_non_channeling_shot(self: "TestDetectChanneling") -> None:
        stable_profile: FlowProfile = [(float(i), 0.2 * i) for i in range(7)] + [
            (float(i), 1.5) for i in range(7, 27)
        ]
        self.assertEqual(detect_channeling(stable_profile), "None")

    def test_high_coefficient_of_variation(self: "TestDetectChanneling") -> None:
        unstable_profile: FlowProfile = (
            [(float(i), 0.2 * i) for i in range(7)]
            + [
                (7.0, 1.0),
                (8.0, 2.5),
                (9.0, 0.8),
                (10.0, 2.2),
                (11.0, 0.5),
                (12.0, 3.0),
                (13.0, 0.7),
                (14.0, 2.8),
                (15.0, 1.0),
                (16.0, 2.5),
            ]
            + [(float(i), 1.5 + (0.5 * (i % 2))) for i in range(17, 27)]
        )
        self.assertEqual(
            detect_channeling(unstable_profile), "Mild Channeling (High Variation)"
        )

    def test_significant_spike(self: "TestDetectChanneling") -> None:
        spike_profile: FlowProfile = (
            [(float(i), 0.2 * i) for i in range(7)]
            + [(float(i), 1.5) for i in range(7, 15)]
            + [(15.0, 4.0)]
            + [(float(i), 1.6) for i in range(16, 27)]
        )
        self.assertEqual(
            detect_channeling(spike_profile), "Suspected Channeling (Spike)"
        )

    def test_high_cv_and_spike(self: "TestDetectChanneling") -> None:
        unstable_spike_profile: FlowProfile = (
            [(float(i), 0.2 * i) for i in range(7)]
            + [
                (7.0, 1.0),
                (8.0, 2.0),
                (9.0, 1.2),
                (10.0, 4.5),
                (11.0, 0.9),
                (12.0, 2.2),
                (13.0, 1.1),
                (14.0, 1.9),
                (15.0, 1.3),
                (16.0, 1.7),
            ]
            + [(float(i), 1.5 + (0.6 * (i % 2))) for i in range(17, 27)]
        )
        self.assertEqual(
            detect_channeling(unstable_spike_profile),
            "Moderate Channeling (High Variation & Spike)",
        )

    def test_low_flow_shot(self: "TestDetectChanneling") -> None:
        low_flow_profile: FlowProfile = [(float(i), 0.1 * i) for i in range(7)] + [
            (float(i), 0.2) for i in range(7, 27)
        ]
        self.assertEqual(detect_channeling(low_flow_profile), "None")

    def test_no_significant_flow_after_ignore(self: "TestDetectChanneling") -> None:
        no_sig_flow_profile: FlowProfile = [(float(i), 0.5 * i) for i in range(7)] + [
            (float(i), 0.03) for i in range(7, 27)
        ]
        self.assertEqual(
            detect_channeling(no_sig_flow_profile),
            "Undetermined (not enough significant flow data)",
        )


class TestIdentifyPreInfusion(unittest.TestCase):
    empty_stp: ScaleTimerProfile = []

    def test_no_profile(self: "TestIdentifyPreInfusion") -> None:
        self.assertEqual(identify_pre_infusion([], self.empty_stp), (False, None))

    def test_too_short_profile(self: "TestIdentifyPreInfusion") -> None:
        profile: FlowProfile = [(0.5, 0.1)]
        self.assertEqual(identify_pre_infusion(profile, self.empty_stp), (False, None))

    def test_in_pi_phase_early_shot(self: "TestIdentifyPreInfusion") -> None:
        profile: FlowProfile = [
            (1.0, 0.1),
            (2.0, 0.15),
            (3.0, 0.2),
        ]  # min_shot_time_for_pi_check = 1.0, current_time = 3.0
        is_pi, duration = identify_pre_infusion(profile, self.empty_stp)
        self.assertTrue(is_pi)
        assert duration is not None  # Added for mypy
        self.assertAlmostEqual(duration, 2.0)  # 3.0 - 1.0

    def test_in_pi_phase_sustained_low_flow(self: "TestIdentifyPreInfusion") -> None:
        profile: FlowProfile = [
            (float(i * 0.5) + 1.0, 0.15) for i in range(6)
        ]  # (1.0,0.15) to (3.5,0.15)
        is_pi, duration = identify_pre_infusion(profile, self.empty_stp)
        self.assertTrue(is_pi)
        assert duration is not None  # Added for mypy
        self.assertAlmostEqual(duration, 2.5)  # 3.5 - 1.0

    def test_exited_pi_phase(self: "TestIdentifyPreInfusion") -> None:
        # PI from 1s to 3s (duration 2s), then flow increases at 4s.
        profile: FlowProfile = [
            (1.0, 0.1),
            (2.0, 0.2),
            (3.0, 0.25),
            (4.0, 1.0),
            (5.0, 1.5),
        ]
        # When current_time is 5.0, current_flow is 1.5 (high).
        # The function should detect that PI *was* present.
        # The current simplified logic for identify_pre_infusion might not correctly return the *past* duration
        # if not currently in PI. Let's test its behavior.
        # Expected based on current simplified logic: (False, None) as it's not *currently* in PI.
        # A more robust version would return (False, 2.0) or similar.
        # For now, let's stick to what the current simplified version should do.
        is_pi, duration = identify_pre_infusion(profile, self.empty_stp)
        self.assertFalse(is_pi)  # Correct: not *currently* in PI
        # The duration part of the simplified placeholder is tricky for *past* PI.
        # It calculates duration based on current low flow. If current flow is high, it won't set duration.
        self.assertIsNone(duration)  # Current simplified logic will yield None here.

    def test_shot_too_long_for_pi(self: "TestIdentifyPreInfusion") -> None:
        profile: FlowProfile = [
            (float(i + 1), 0.2) for i in range(16)
        ]  # current_time = 16.0 > max_time_for_pi (15.0)
        is_pi, duration = identify_pre_infusion(profile, self.empty_stp)
        self.assertFalse(is_pi)
        self.assertIsNone(duration)

    def test_no_clear_pi_high_flow_early(self: "TestIdentifyPreInfusion") -> None:
        profile: FlowProfile = [(1.0, 1.0), (2.0, 1.5), (3.0, 1.2)]
        is_pi, duration = identify_pre_infusion(profile, self.empty_stp)
        self.assertFalse(is_pi)
        self.assertIsNone(duration)

    def test_pi_duration_too_short_is_nulled(self: "TestIdentifyPreInfusion") -> None:
        # current_time = 1.5, current_flow = 0.2. pi_start_time = 1.0. estimated_duration = 0.5 < 1.0
        profile_at_1_5: FlowProfile = [(1.0, 0.1), (1.5, 0.2)]
        is_pi_at_1_5, duration_at_1_5 = identify_pre_infusion(
            profile_at_1_5, self.empty_stp
        )
        self.assertTrue(is_pi_at_1_5)  # Still considered in PI phase if flow is low
        self.assertIsNone(duration_at_1_5)  # But duration is nulled because it's < 1.0


class TestCalculateExtractionUniformity(unittest.TestCase):
    def test_no_profile(self: "TestCalculateExtractionUniformity") -> None:
        self.assertEqual(calculate_extraction_uniformity([]), 0.0)

    def test_insufficient_data_after_ignore(
        self: "TestCalculateExtractionUniformity",
    ) -> None:
        profile: FlowProfile = [
            (float(i), 0.5) for i in range(8)
        ]  # initial_ignore_seconds = 7.0, so only 1 data point [7.0, 0.5]
        self.assertEqual(calculate_extraction_uniformity(profile), 0.0)

    def test_no_significant_flow_for_uniformity(
        self: "TestCalculateExtractionUniformity",
    ) -> None:
        profile: FlowProfile = (
            [(float(i), 0.2 * i) for i in range(7)]
            + [
                (float(i), 0.05) for i in range(7, 27)
            ]  # flow is <= 0.1, so 'flows' list will be empty
        )
        self.assertEqual(calculate_extraction_uniformity(profile), 0.0)

    def test_perfectly_uniform_flow(self: "TestCalculateExtractionUniformity") -> None:
        profile: FlowProfile = [(float(i), 0.2 * i) for i in range(7)] + [
            (float(i), 1.5) for i in range(7, 27)
        ]
        self.assertAlmostEqual(calculate_extraction_uniformity(profile), 1.0)

    def test_moderately_variable_flow(
        self: "TestCalculateExtractionUniformity",
    ) -> None:
        # Create a profile with moderate CoV. For 1 - CoV to be e.g. 0.8, CoV should be 0.2
        # mean = 1.5, std_dev = 0.3 => CoV = 0.2
        # Flows: e.g., 1.2, 1.8, 1.5, 1.2, 1.8, 1.5 ... (mean 1.5)
        # (1.2-1.5)^2 = 0.09, (1.8-1.5)^2 = 0.09. Variance = (0.09*N/2 + 0.09*N/2)/N = 0.09. std_dev = 0.3
        analysis_flows = [1.2, 1.8] * 10  # 20 points, mean 1.5, std_dev 0.3
        profile: FlowProfile = [(float(i), 0.2 * i) for i in range(7)] + [
            (float(i + 7), flow) for i, flow in enumerate(analysis_flows)
        ]
        # CoV = 0.3 / 1.5 = 0.2. Score = 1 - 0.2 = 0.8
        self.assertAlmostEqual(calculate_extraction_uniformity(profile), 0.80)

    def test_highly_variable_flow(self: "TestCalculateExtractionUniformity") -> None:
        # Create a profile with high CoV. For 1 - CoV to be e.g. 0.2, CoV should be 0.8
        # mean = 1.5, std_dev = 1.2 => CoV = 0.8
        # Flows: e.g., 0.3, 2.7 (mean 1.5)
        # (0.3-1.5)^2 = 1.44, (2.7-1.5)^2 = 1.44. Variance = 1.44. std_dev = 1.2
        analysis_flows = [0.3, 2.7] * 10
        profile: FlowProfile = [(float(i), 0.2 * i) for i in range(7)] + [
            (float(i + 7), flow) for i, flow in enumerate(analysis_flows)
        ]
        # CoV = 1.2 / 1.5 = 0.8. Score = 1 - 0.8 = 0.2
        self.assertAlmostEqual(calculate_extraction_uniformity(profile), 0.20)

    def test_uniformity_score_capped_at_zero(self):
        # CoV > 1, e.g. mean = 1.0, std_dev = 1.1 => CoV = 1.1. Score = max(0, 1 - 1.1) = 0
        # Let's use positive flows: e.g. [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.9] mean=0.38, std_dev large
        # Let's make it simpler: flows = [0.1] * 19 + [2.0] -> mean = (1.9+2.0)/20 = 0.195. std_dev will be large.
        # flows = [0.1, 0.1, ..., 0.1 (10 times), 2.0, ..., 2.0 (10 times)] -> mean = 1.05, CoV will be high
        # Let's use flows that are all > 0.1 for the 'flows' list filter in the function
        # To get CoV > 1: mean = 1.0, std_dev = 1.1. e.g. flows like [0.1, 0.1, ... 1.9, 1.9]
        # Let's use a simpler case where CoV is clearly > 1
        # mean = 0.5, flows = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 4.1] -> mean = 0.5. std_dev large.
        # (0.1-0.5)^2 * 9 = (-0.4)^2 * 9 = 0.16 * 9 = 1.44
        # (4.1-0.5)^2 * 1 = (3.6)^2 = 12.96
        # Variance = (1.44 + 12.96) / 10 = 14.4 / 10 = 1.44. std_dev = 1.2
        # CoV = 1.2 / 0.5 = 2.4. Score = max(0, 1 - 2.4) = 0.
        analysis_flows_very_high_cov = [0.1] * 9 + [4.1]
        profile: FlowProfile = [(float(i), 0.2 * i) for i in range(7)] + [
            (float(i + 7), flow) for i, flow in enumerate(analysis_flows_very_high_cov)
        ]
        self.assertAlmostEqual(calculate_extraction_uniformity(profile), 0.0)


if __name__ == "__main__":
    unittest.main()
