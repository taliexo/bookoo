# tests/fixtures/shot_profiles.py
"""Sample shot profiles for testing."""

PERFECT_SHOT_PROFILE = {
    "duration": 28,
    "final_weight": 36.0,
    "flow_data": [
        (0, 0.0),
        (5, 0.5),
        (10, 2.0),
        (15, 2.2),
        (20, 2.1),
        (25, 1.8),
        (28, 0.2),
    ],
}

CHANNELING_SHOT_PROFILE = {
    "duration": 25,
    "final_weight": 32.0,
    "flow_data": [
        (0, 0.0),
        (5, 0.5),
        (10, 1.5),
        (12, 4.5),  # Spike
        (15, 1.2),
        (20, 3.8),
        (25, 0.1),  # Another spike
    ],
}
