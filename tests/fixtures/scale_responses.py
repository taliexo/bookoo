# tests/fixtures/scale_responses.py
"""Mock responses and data from the Bookoo scale.

This file can be used to store example byte arrays for characteristic
notifications or read responses from the scale. The exact byte structure
depends on the scale's Bluetooth protocol and how aiobookoov2 parses it.

For many unit/integration tests, it's often easier to mock the parsed data
objects (e.g., BookooWeightData) directly. However, these raw byte
fixtures can be useful for testing the parsing logic itself or for tests
that simulate BLE interactions at a lower level.
"""

# Example: Weight Data (Illustrative - actual format depends on scale protocol)
# Assuming a simple format for demonstration:
# - 2 bytes for weight in grams (e.g., 10ths of a gram, or grams directly)
# - 1 byte for status (e.g., 0 = stable, 1 = unstable)

WEIGHT_ZERO_STABLE = b"\x00\x00\x00"  # 0.0g, stable
WEIGHT_10_5G_STABLE = b"\x00\x69\x00"  # 10.5g (105 * 0.1g), stable (if 0.1g units)
# Or if direct grams, 0x0069 = 105g
# Let's assume direct grams for simplicity in this example
# So, 0x000A = 10g, 0x0005 = 0.5g - this is not how it usually works.
# A more common way: 105 means 10.5g. So 0x69 = 105.

WEIGHT_250_2G_STABLE = b"\x09\xc6\x00"  # 250.2g (2502 * 0.1g), stable. 0x09C6 = 2502
WEIGHT_500G_UNSTABLE = b"\x13\x88\x01"  # 500.0g (5000 * 0.1g), unstable. 0x1388 = 5000

# Example: Timer Data (Illustrative)
# Assuming 2 bytes for seconds
TIMER_30_SECONDS = b"\x00\x1e"  # 30 seconds
TIMER_65_SECONDS = b"\x00\x41"  # 65 seconds

# Example: Combined Data Packet (Illustrative - if scale sends one)
# e.g., <2 bytes weight> <1 byte weight_status> <2 bytes timer_seconds>
COMBINED_15_2G_STABLE_45S = (
    b"\x00\x98\x00\x00\x2d"  # 15.2g (152), stable, 45s (0x0098=152, 0x002D=45)
)

# Note: The actual parsing logic in aiobookoov2 would determine how these bytes
# are converted to BookooWeightData, BookooTimerData, etc. These are placeholders.

# It might be more useful to define helper functions that return these byte arrays
# or even pre-constructed mock data objects if not testing byte parsing.


def get_mock_weight_data_bytes(weight_grams: float, stable: bool = True) -> bytes:
    """Illustrative: Creates a mock weight data byte string.
    Assumes weight is grams * 10, sent as 2 bytes, followed by stability byte.
    This is a simplified example.
    """
    if not 0 <= weight_grams <= 6553.5:  # Max for 2 bytes if units are 0.1g
        raise ValueError("Weight out of illustrative range for 2 bytes (0.1g units)")

    weight_tenths_of_gram = int(round(weight_grams * 10))
    stability_byte = 0 if stable else 1

    # Pack as big-endian short (2 bytes)
    packed_weight = weight_tenths_of_gram.to_bytes(2, "big")
    packed_stability = stability_byte.to_bytes(1, "big")

    return packed_weight + packed_stability


# Example usage of the helper:
WEIGHT_BYTES_20_0G_STABLE = get_mock_weight_data_bytes(
    20.0, stable=True
)  # b'\x00\xC8\x00'
WEIGHT_BYTES_35_5G_UNSTABLE = get_mock_weight_data_bytes(
    35.5, stable=False
)  # b'\x01\x63\x01'
