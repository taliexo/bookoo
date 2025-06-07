# tests/unit/test_analytics.py
import pytest
import math
from custom_components.bookoo.analytics import AnalyticsConfig, ShotAnalyzer
from custom_components.bookoo.types import (
    FlowDataPoint,
)  # ScaleTimerProfile might be needed later

# --- Fixtures for AnalyticsConfig ---


@pytest.fixture
def default_analytics_config() -> AnalyticsConfig:
    return AnalyticsConfig()


@pytest.fixture
def custom_analytics_config() -> AnalyticsConfig:
    return AnalyticsConfig(
        channeling_initial_ignore_seconds=5.0,
        channeling_cv_threshold_mild=0.30,
        channeling_spike_factor_high=2.0,
        pi_flow_threshold=0.25,
    )


# --- Fixtures for Flow Profiles (add more as needed) ---


@pytest.fixture
def empty_flow_profile() -> list[FlowDataPoint]:
    return []


@pytest.fixture
def insufficient_data_flow_profile(
    default_analytics_config: AnalyticsConfig,
) -> list[FlowDataPoint]:
    # Needs to be after initial_ignore_seconds but less than min_data_points
    return [
        FlowDataPoint(
            elapsed_time=default_analytics_config.channeling_initial_ignore_seconds
            + i * 0.1,
            flow_rate=1.0 + i * 0.1,
        )
        for i in range(default_analytics_config.channeling_min_data_points - 1)
    ]


@pytest.fixture
def no_significant_flow_profile(
    default_analytics_config: AnalyticsConfig,
) -> list[FlowDataPoint]:
    config = default_analytics_config
    return [
        FlowDataPoint(
            elapsed_time=config.channeling_initial_ignore_seconds + i * 0.1,
            flow_rate=config.channeling_significant_flow_threshold * 0.5,
        )
        for i in range(
            config.channeling_min_data_points + 5
        )  # Enough points, but all low flow
    ]


@pytest.fixture
def smooth_flow_profile(
    default_analytics_config: AnalyticsConfig,
) -> list[FlowDataPoint]:
    config = default_analytics_config
    return [
        FlowDataPoint(
            elapsed_time=config.channeling_initial_ignore_seconds + i * 0.1,
            flow_rate=1.0,
        )
        for i in range(config.channeling_min_data_points + 10)  # Stable flow
    ]


# --- Test Cases ---


# Test AnalyticsConfig
def test_analytics_config_defaults(default_analytics_config: AnalyticsConfig):
    """Test that AnalyticsConfig has expected default values."""
    assert default_analytics_config.channeling_initial_ignore_seconds == 7.0
    assert default_analytics_config.channeling_cv_threshold_mild == 0.38
    assert default_analytics_config.pi_flow_threshold == 0.3
    # Add more default checks as needed


def test_analytics_config_custom(custom_analytics_config: AnalyticsConfig):
    """Test creating AnalyticsConfig with custom values."""
    assert custom_analytics_config.channeling_initial_ignore_seconds == 5.0
    assert custom_analytics_config.channeling_cv_threshold_mild == 0.30
    assert custom_analytics_config.pi_flow_threshold == 0.25


# Test ShotAnalyzer initialization
def test_shot_analyzer_initialization(
    default_analytics_config: AnalyticsConfig, custom_analytics_config: AnalyticsConfig
):
    analyzer_default = ShotAnalyzer()
    assert (
        analyzer_default.config.channeling_initial_ignore_seconds == 7.0
    )  # Check against default

    analyzer_custom = ShotAnalyzer(config=custom_analytics_config)
    assert (
        analyzer_custom.config.channeling_initial_ignore_seconds == 5.0
    )  # Check against custom


# --- Tests for detect_channeling ---


def test_detect_channeling_empty_profile(
    default_analytics_config: AnalyticsConfig, empty_flow_profile: list[FlowDataPoint]
):
    analyzer = ShotAnalyzer(config=default_analytics_config)
    assert analyzer.detect_channeling(empty_flow_profile) == "Undetermined"


def test_detect_channeling_insufficient_data_after_ignore(
    default_analytics_config: AnalyticsConfig,
    insufficient_data_flow_profile: list[FlowDataPoint],
):
    """Test when data points after initial ignore are less than min_data_points."""
    analyzer = ShotAnalyzer(config=default_analytics_config)
    # This profile is constructed to have just enough points overall, but not after ignore for some configs
    # Let's make a profile that is explicitly too short *after* the ignore period.
    profile = [
        FlowDataPoint(
            elapsed_time=default_analytics_config.channeling_initial_ignore_seconds - 1,
            flow_rate=1.0,
        ),  # Before ignore
        FlowDataPoint(
            elapsed_time=default_analytics_config.channeling_initial_ignore_seconds
            + 0.1,
            flow_rate=1.0,
        ),  # After ignore, 1 point
    ]
    # Adjust min_data_points for this specific test scenario if needed, or ensure profile is truly insufficient
    config_min_points_test = AnalyticsConfig(channeling_min_data_points=2)
    analyzer_min_points = ShotAnalyzer(config=config_min_points_test)
    assert (
        analyzer_min_points.detect_channeling(profile)
        == "Undetermined (not enough data after initial phase)"
    )


def test_detect_channeling_no_significant_flow(
    default_analytics_config: AnalyticsConfig,
    no_significant_flow_profile: list[FlowDataPoint],
):
    analyzer = ShotAnalyzer(config=default_analytics_config)
    assert (
        analyzer.detect_channeling(no_significant_flow_profile)
        == "Undetermined (not enough significant flow data)"
    )


def test_detect_channeling_smooth_flow(
    default_analytics_config: AnalyticsConfig, smooth_flow_profile: list[FlowDataPoint]
):
    analyzer = ShotAnalyzer(config=default_analytics_config)
    assert analyzer.detect_channeling(smooth_flow_profile) == "None"


@pytest.mark.parametrize(
    "flow_values, expected_status_substring",
    [
        # Mild Channeling (High Variation)
        (
            [1.0, 1.1, 0.9, 1.5, 0.5, 1.2, 0.8],
            "None",
        ),  # CV is ~0.29, below mild threshold 0.38
        # Suspected Channeling (Spike) - high spike factor
        (
            [1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0],
            "Moderate Channeling (High Variation & Spike)",
        ),  # CV is also high for this data
        # Moderate Channeling (High Variation & Spike)
        (
            [1.0, 1.5, 0.5, 1.2, 0.8, 3.0, 0.9],
            "Moderate Channeling (High Variation & Spike)",
        ),
        # Mild-Moderate (Variation & Notable Peak) - moderate spike factor + moderate CV
        (
            [1.0, 1.3, 0.7, 1.2, 0.8, 2.0, 0.9],
            "None",
        ),  # CV ~0.36 (below mild), spike factor not met for moderate
    ],
)
def test_detect_channeling_various_scenarios(
    default_analytics_config: AnalyticsConfig,
    flow_values: list[float],
    expected_status_substring: str,
):
    config = default_analytics_config
    # Ensure enough data points for analysis after initial ignore
    # And ensure mean flow is high enough for spike detection if applicable
    if "Spike" in expected_status_substring or "Peak" in expected_status_substring:
        config.channeling_spike_mean_flow_threshold = (
            0.4  # Ensure mean flow is likely to be above this
        )

    # Construct profile ensuring it's long enough and values are applied after ignore period
    num_initial_padding_points = 5  # Points before ignore period starts
    profile = [
        FlowDataPoint(
            elapsed_time=i * 0.5, flow_rate=1.0
        )  # Some stable flow before ignore
        for i in range(num_initial_padding_points)
    ]
    profile.extend(
        [
            FlowDataPoint(
                elapsed_time=config.channeling_initial_ignore_seconds + i * 0.1,
                flow_rate=val,
            )
            for i, val in enumerate(flow_values)
        ]
    )
    # Ensure total points in analysis_profile meet min_data_points
    # Add more stable points if flow_values is too short for min_data_points
    min_total_analysis_points = config.channeling_min_data_points
    if len(flow_values) < min_total_analysis_points:
        needed_more_points = min_total_analysis_points - len(flow_values)
        last_time = profile[-1].elapsed_time
        profile.extend(
            [
                FlowDataPoint(
                    elapsed_time=last_time + (j + 1) * 0.1,
                    flow_rate=flow_values[-1] if flow_values else 1.0,
                )
                for j in range(needed_more_points)
            ]
        )

    analyzer = ShotAnalyzer(config=config)
    status = analyzer.detect_channeling(profile)
    assert expected_status_substring in status


# --- Tests for identify_pre_infusion ---


@pytest.mark.parametrize(
    "profile_data, expected_is_pi, expected_duration_substring",
    [
        ([], False, None),  # Empty profile
        ([FlowDataPoint(0.5, 0.1)], False, None),  # Insufficient profile (1 point)
    ],
)
def test_identify_pre_infusion_empty_or_insufficient(
    default_analytics_config: AnalyticsConfig,
    profile_data: list[FlowDataPoint],
    expected_is_pi: bool,
    expected_duration_substring: str
    | None,  # Not used here, but keeps signature consistent
):
    analyzer = ShotAnalyzer(config=default_analytics_config)
    is_pi, duration = analyzer.identify_pre_infusion(
        profile_data, []
    )  # Empty scale_timer_profile
    assert is_pi == expected_is_pi
    assert duration is None


def test_identify_pre_infusion_clear_case(default_analytics_config: AnalyticsConfig):
    config = default_analytics_config
    # Ensure pi_min_shot_time_check < start of PI < pi_max_time
    # Ensure duration > pi_min_duration_meaningful
    profile = [
        FlowDataPoint(
            config.pi_min_shot_time_check - 0.5, config.pi_flow_threshold + 0.1
        ),  # Before check time, high flow
        FlowDataPoint(
            config.pi_min_shot_time_check + 0.5, config.pi_flow_threshold - 0.1
        ),  # PI starts
        FlowDataPoint(
            config.pi_min_shot_time_check + 1.0, config.pi_flow_threshold - 0.1
        ),
        FlowDataPoint(
            config.pi_min_shot_time_check + 1.5, config.pi_flow_threshold - 0.1
        ),  # Current point, PI ongoing
    ]
    # Expected duration: (check_time + 1.5) - (check_time + 0.5) = 1.0. If meaningful, this is it.
    # If pi_min_duration_meaningful is, e.g., 0.5, then 1.0 is fine.
    config.pi_min_duration_meaningful = 0.5
    analyzer = ShotAnalyzer(config=config)
    is_pi, duration = analyzer.identify_pre_infusion(profile, [])
    assert is_pi is True
    assert duration == pytest.approx(1.0)


def test_identify_pre_infusion_no_pi_high_flow(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    profile = [
        FlowDataPoint(
            config.pi_min_shot_time_check + 1.0, config.pi_flow_threshold + 0.2
        ),
        FlowDataPoint(
            config.pi_min_shot_time_check + 2.0, config.pi_flow_threshold + 0.3
        ),
    ]
    analyzer = ShotAnalyzer(config=config)
    is_pi, duration = analyzer.identify_pre_infusion(profile, [])
    assert is_pi is False
    assert duration is None


def test_identify_pre_infusion_too_short_duration(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    config.pi_min_duration_meaningful = 2.0  # Make it hard to achieve meaningful PI
    profile = [
        FlowDataPoint(
            config.pi_min_shot_time_check + 0.5, config.pi_flow_threshold - 0.1
        ),  # PI starts
        FlowDataPoint(
            config.pi_min_shot_time_check + 1.0, config.pi_flow_threshold - 0.1
        ),  # PI ongoing, duration 0.5s
    ]
    analyzer = ShotAnalyzer(config=config)
    is_pi, duration = analyzer.identify_pre_infusion(profile, [])
    assert is_pi is False  # Because duration (0.5s) < pi_min_duration_meaningful (2.0s)
    assert duration is None


def test_identify_pre_infusion_exceeds_max_time(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    profile = [
        FlowDataPoint(config.pi_max_time - 1.0, config.pi_flow_threshold - 0.1),
        FlowDataPoint(
            config.pi_max_time + 1.0, config.pi_flow_threshold - 0.1
        ),  # Current time exceeds pi_max_time
    ]
    analyzer = ShotAnalyzer(config=config)
    is_pi, duration = analyzer.identify_pre_infusion(profile, [])
    assert is_pi is False
    assert duration is None


def test_identify_pre_infusion_before_min_check_time(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    profile = [
        FlowDataPoint(
            config.pi_min_shot_time_check - 0.5, config.pi_flow_threshold - 0.1
        ),  # Low flow, but too early
    ]
    analyzer = ShotAnalyzer(config=config)
    is_pi, duration = analyzer.identify_pre_infusion(profile, [])
    assert is_pi is False
    assert duration is None


def test_identify_pre_infusion_flow_recovers_then_dips(
    default_analytics_config: AnalyticsConfig,
):
    """Test PI detection when flow dips, recovers slightly above threshold, then dips again."""
    config = default_analytics_config
    config.pi_min_duration_meaningful = 0.5  # Ensure durations are meaningful
    profile = [
        FlowDataPoint(
            elapsed_time=2.0, flow_rate=config.pi_flow_threshold - 0.1
        ),  # PI starts (t=2)
        FlowDataPoint(
            elapsed_time=2.5, flow_rate=config.pi_flow_threshold - 0.1
        ),  # PI continues (t=2.5, dur=0.5)
        FlowDataPoint(
            elapsed_time=3.0, flow_rate=config.pi_flow_threshold + 0.05
        ),  # Flow recovers slightly (t=3)
        FlowDataPoint(
            elapsed_time=3.5, flow_rate=config.pi_flow_threshold - 0.1
        ),  # Dips again (t=3.5, new PI phase starts)
        FlowDataPoint(
            elapsed_time=4.0, flow_rate=config.pi_flow_threshold - 0.1
        ),  # Current point (t=4, dur=0.5 for this new phase)
    ]
    analyzer = ShotAnalyzer(config=config)
    is_pi, duration = analyzer.identify_pre_infusion(profile, [])
    assert is_pi is True
    assert duration == pytest.approx(0.5)  # Duration of the most recent low-flow phase


# --- Tests for calculate_extraction_uniformity ---


def test_calculate_extraction_uniformity_empty_profile(
    default_analytics_config: AnalyticsConfig, empty_flow_profile: list[FlowDataPoint]
):
    analyzer = ShotAnalyzer(config=default_analytics_config)
    assert analyzer.calculate_extraction_uniformity(empty_flow_profile) == 0.0


def test_calculate_extraction_uniformity_insufficient_data_after_ignore(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    # Profile with points only before or too few after ignore period
    profile = [
        FlowDataPoint(
            config.uniformity_initial_ignore_seconds - 1.0, 1.0
        ),  # Before ignore
        FlowDataPoint(
            config.uniformity_initial_ignore_seconds + 0.1, 1.0
        ),  # 1 point after
    ]
    # Ensure min_data_points is greater than points in analysis_profile
    config.uniformity_min_data_points = 2
    analyzer = ShotAnalyzer(config=config)
    assert analyzer.calculate_extraction_uniformity(profile) == 0.0


def test_calculate_extraction_uniformity_insufficient_significant_flow_points(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    # Enough points after ignore, but flow is too low for most of them
    profile = [
        FlowDataPoint(
            config.uniformity_initial_ignore_seconds + i * 0.1,
            config.uniformity_significant_flow_threshold * 0.5,
        )
        for i in range(
            config.uniformity_min_data_points
        )  # Total points meet min_data_points
    ]
    # Add one significant flow point, but not enough to meet ratio
    profile.append(
        FlowDataPoint(
            config.uniformity_initial_ignore_seconds
            + config.uniformity_min_data_points * 0.1,
            config.uniformity_significant_flow_threshold + 0.1,
        )
    )

    config.uniformity_min_significant_flow_points_ratio = (
        0.5  # Default, requires 5 if min_data_points is 10
    )
    # If min_data_points is 10, we need 5 significant points. We have 1.
    analyzer = ShotAnalyzer(config=config)
    assert analyzer.calculate_extraction_uniformity(profile) == 0.0


def test_calculate_extraction_uniformity_perfectly_stable_flow(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    flow_rate = (
        config.uniformity_significant_flow_threshold + 0.5
    )  # Ensure it's significant
    profile = [
        FlowDataPoint(config.uniformity_initial_ignore_seconds + i * 0.1, flow_rate)
        for i in range(
            config.uniformity_min_data_points + 5
        )  # Enough significant points
    ]
    analyzer = ShotAnalyzer(config=config)
    assert analyzer.calculate_extraction_uniformity(profile) == pytest.approx(1.0)


def test_calculate_extraction_uniformity_moderately_variable_flow(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    flows_in_analysis = [1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.3, 0.7, 1.0, 1.0]  # 10 points
    # Ensure these flows are above significant_flow_threshold
    config.uniformity_significant_flow_threshold = 0.5
    config.uniformity_min_data_points = len(flows_in_analysis)
    config.uniformity_min_significant_flow_points_ratio = (
        0.1  # Ensure all points are counted
    )

    profile = [
        FlowDataPoint(config.uniformity_initial_ignore_seconds + i * 0.1, flow_val)
        for i, flow_val in enumerate(flows_in_analysis)
    ]

    mean_flow = sum(flows_in_analysis) / len(flows_in_analysis)
    variance = sum([(f - mean_flow) ** 2 for f in flows_in_analysis]) / len(
        flows_in_analysis
    )
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_flow if mean_flow > 0 else float("inf")
    expected_score = max(0.0, 1.0 - cv)
    expected_score_rounded = round(
        expected_score, 2
    )  # Apply same rounding as the method

    analyzer = ShotAnalyzer(config=config)
    assert analyzer.calculate_extraction_uniformity(profile) == pytest.approx(
        expected_score_rounded
    )


def test_calculate_extraction_uniformity_highly_variable_flow(
    default_analytics_config: AnalyticsConfig,
):
    """Test with flow so variable that CV > 1.0, score should be 0.0."""
    config = default_analytics_config
    # This profile is designed to have a Coefficient of Variation (CV) > 1.0
    flows_in_analysis = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 5.0]
    config.uniformity_significant_flow_threshold = (
        0.05  # Ensure all points are considered significant
    )
    config.uniformity_min_data_points = len(flows_in_analysis)
    config.uniformity_min_significant_flow_points_ratio = (
        0.1  # Ensure enough significant points
    )
    config.uniformity_initial_ignore_seconds = 0  # Analyze from the start for this test

    profile = [
        FlowDataPoint(config.uniformity_initial_ignore_seconds + (i * 0.1), flow_val)
        for i, flow_val in enumerate(flows_in_analysis)
    ]
    analyzer = ShotAnalyzer(config=config)
    # With CV > 1, score should be max(0, 1 - CV) which results in 0.0
    # The method also rounds to 2 decimal places, so 0.0 is expected.
    assert analyzer.calculate_extraction_uniformity(profile) == pytest.approx(0.0)


def test_calculate_extraction_uniformity_all_flow_below_significant(
    default_analytics_config: AnalyticsConfig,
):
    config = default_analytics_config
    profile = [
        FlowDataPoint(
            config.uniformity_initial_ignore_seconds + i * 0.1,
            config.uniformity_significant_flow_threshold * 0.5,
        )
        for i in range(config.uniformity_min_data_points + 5)
    ]
    analyzer = ShotAnalyzer(config=config)
    assert analyzer.calculate_extraction_uniformity(profile) == 0.0


# --- Tests for calculate_peak_flow_rate ---


@pytest.mark.parametrize(
    "profile_data, expected_peak_flow",
    [
        ([], 0.0),  # Empty profile
        (
            [FlowDataPoint(1.0, None), FlowDataPoint(2.0, None)],
            0.0,
        ),  # All None flow rates
        (
            [FlowDataPoint(1.0, 0.5), FlowDataPoint(2.0, 1.5), FlowDataPoint(3.0, 1.0)],
            1.5,
        ),  # Clear peak
        ([FlowDataPoint(1.0, 0.0), FlowDataPoint(2.0, 0.0)], 0.0),  # All zero flow
        ([FlowDataPoint(1.0, 0.5)], 0.5),  # Single point
    ],
)
def test_calculate_peak_flow_rate(
    default_analytics_config: AnalyticsConfig,
    profile_data: list[FlowDataPoint],
    expected_peak_flow: float,
):
    analyzer = ShotAnalyzer(config=default_analytics_config)
    assert analyzer.calculate_peak_flow_rate(profile_data) == pytest.approx(
        expected_peak_flow
    )


# --- Tests for calculate_time_to_peak_flow ---


@pytest.mark.parametrize(
    "profile_data, expected_time_to_peak",
    [
        ([], None),  # Empty profile
        (
            [FlowDataPoint(1.0, None), FlowDataPoint(2.0, None)],
            None,
        ),  # All None flow rates
        (
            [FlowDataPoint(1.0, 0.5), FlowDataPoint(2.0, 1.5), FlowDataPoint(3.0, 1.0)],
            2.0,
        ),  # Clear peak
        (
            [FlowDataPoint(1.0, 0.0), FlowDataPoint(2.0, 0.0), FlowDataPoint(3.0, 0.0)],
            1.0,
        ),  # All zero flow, peak is first 0.0
        (
            [FlowDataPoint(1.0, 1.5), FlowDataPoint(2.0, 1.0), FlowDataPoint(3.0, 1.5)],
            3.0,
        ),  # Multiple equal peaks, last one taken
        ([FlowDataPoint(1.0, 0.5)], 1.0),  # Single point
    ],
)
def test_calculate_time_to_peak_flow(
    default_analytics_config: AnalyticsConfig,
    profile_data: list[FlowDataPoint],
    expected_time_to_peak: float | None,
):
    analyzer = ShotAnalyzer(config=default_analytics_config)
    actual_time = analyzer.calculate_time_to_peak_flow(profile_data)
    if expected_time_to_peak is None:
        assert actual_time is None
    else:
        assert actual_time == pytest.approx(expected_time_to_peak)
