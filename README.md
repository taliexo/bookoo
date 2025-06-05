# Bookoo Scale Integration

This is a Home Assistant integration for the BOOKOO Bluetooth scale. It provides sensors for weight, flow rate, and the timer, along with buttons for basic scale control. This fork significantly enhances the integration by adding robust espresso shot session logging, including automatic detection of the scale's auto-timer events and detailed data capture.

Based on the original documentation at https://github.com/BooKooCode/OpenSource/

## Features

*   Standard sensors: Weight, Flow Rate, Timer, Battery.
*   Control buttons: Tare, Reset Timer, Start Timer (on scale), Stop Timer (on scale), Tare and Start Timer (on scale).
*   **Advanced Espresso Shot Session Tracking:**
    *   Automatic detection of shot start/stop based on the scale's built-in auto-timer function (requires scale firmware that supports this via 0xFF12 characteristic, message 0x0D).
    *   Manual shot session control through dedicated Home Assistant services and buttons.
    *   Detailed data capture for each shot, including:
        *   Precise start and end times (UTC).
        *   Shot duration.
        *   Final beverage weight.
        *   Flow profile (time-series data of flow rate in grams/second).
        *   Scale's internal timer profile (time-series data of the scale's timer reading in milliseconds).
        *   User-defined parameters (e.g., bean weight, coffee name, grind setting) linked from `input_*` helper entities.
    *   Fires a `bookoo_shot_completed` event with a comprehensive payload upon shot completion.
*   **Configurable Minimum Shot Duration:** Filter out accidental or very short timer activations that aren't actual shots.
*   **Configurable Maximum Shot Duration:** Automatically stop and flag shots that run excessively long.
*   **Linkable Input Helpers:** Associate your own `input_number`, `input_text`, etc., helpers with each shot to record parameters like bean dose, coffee type, grind setting, and more.

## Installation

1.  **HACS Installation (Recommended):**
    *   Add this repository as a custom repository in HACS (Home Assistant Community Store).
    *   Search for "Bookoo Scale Integration" (or the name defined in HACS) and install it.
2.  **Manual Installation:**
    *   Copy the `custom_components/bookoo` directory from this repository into your Home Assistant `config/custom_components/` directory.
3.  **Restart Home Assistant.**
4.  After restarting, your Bookoo Themis scale should be automatically discovered by Home Assistant if Bluetooth is configured correctly. You can also add it manually via the Integrations page.

## Configuration

Basic device connection is handled during the initial setup via the Integrations page in Home Assistant.

To access advanced espresso shot logging features, configure the integration's options:
1.  Go to **Settings > Devices & Services > Integrations**.
2.  Find your Bookoo Scale integration instance.
3.  Click on **"Configure"** (or the options/three-dot menu if already configured).

### Options

*   **Minimum Shot Duration (seconds):**
    *   Defines the minimum duration (in seconds) for a shot to be considered valid and trigger the `bookoo_shot_completed` event. This helps filter out brief timer activations that aren't actual espresso shots.
    *   Default: 10 seconds. Set to 0 to record all timer events if minimum duration is the only concern.
*   **Maximum Shot Duration (seconds):**
    *   Defines the maximum duration (in seconds) for a shot. If a shot exceeds this duration, it will be automatically stopped and its status recorded as "aborted_too_long".
    *   This is useful for preventing excessively long recordings due to issues like forgetting to stop the timer or a scale malfunction.
    *   Default: 120 seconds (2 minutes). Set to 0 to disable the maximum duration check.
*   **Linked Input Entities:**
    *   This section allows you to link existing Home Assistant `input_number`, `input_text`, or other helper entities to your espresso shots. When a shot is started, the current values of these linked entities will be captured and included in the `bookoo_shot_completed` event data.
    *   **How to use:**
        1.  Create your desired helper entities in Home Assistant (e.g., an `input_number.espresso_bean_dose` for bean weight, `input_text.espresso_coffee_name` for the coffee blend).
        2.  In the Bookoo integration options, select the corresponding helper entity from the dropdown list for each parameter you want to track (e.g., "Bean Weight Input Entity", "Coffee Name Input Entity").
    *   The values will be available in the `input_parameters` dictionary of the `bookoo_shot_completed` event data (e.g., `event_data.input_parameters.bean_weight`).
*   **Auto-Stop Feature Options:**
    *   Configure parameters for automatically stopping a shot based on flow rate characteristics (e.g., when flow drops below a certain threshold for a set duration after a stable extraction phase). This can help automate the end of a shot.
*   **Bluetooth Timeouts:**
    *   Configure timeouts for Bluetooth connection attempts and command executions to fine-tune reliability in your environment.

## Entities

This integration provides the following entities:

### Sensors

*   **Weight (`sensor.bookoo_weight`):** Current weight on the scale (grams).
*   **Flow Rate (`sensor.bookoo_flow_rate`):** Current flow rate (g/s).
*   **Timer (`sensor.bookoo_timer`):** Current reading of the scale's timer (seconds).
*   **Battery (`sensor.bookoo_battery`):** Battery level of the scale (%).
*   **NEW - Current Shot Duration (`sensor.bookoo_current_shot_duration`):** Live duration of the currently active espresso shot (seconds). Resets to 0 when no shot is active.
*   **NEW - Last Shot Duration (`sensor.bookoo_last_shot_duration`):** Duration of the last completed espresso shot (seconds).
*   **NEW - Last Shot Final Weight (`sensor.bookoo_last_shot_final_weight`):** Final beverage weight of the last completed shot (grams).
*   **NEW - Last Shot Start Time (`sensor.bookoo_last_shot_start_time`):** UTC timestamp when the last shot started.
*   **NEW - Last Shot Status (`sensor.bookoo_last_shot_status`):** Status of the last shot (e.g., "completed", "aborted_disconnected", "aborted_too_short", "aborted_too_long").
*   **NEW - Current Shot Channeling Status (`sensor.bookoo_current_shot_channeling_status`):** Real-time assessment of channeling during an active shot (e.g., "None", "Mild Channeling", "Moderate Channeling"). Updates live.
*   **NEW - Current Shot Pre-infusion Duration (`sensor.bookoo_current_shot_pre_infusion_duration`):** Duration of the detected pre-infusion phase in seconds during an active shot. Updates live.
*   **NEW - Current Shot Extraction Uniformity (`sensor.bookoo_current_shot_extraction_uniformity`):** A metric (typically 0-1) indicating the uniformity of the extraction based on flow rate analysis during an active shot. Updates live.
*   **NEW - Current Shot Quality Score (`sensor.bookoo_current_shot_quality_score`):** A unified score (0-100%) representing the overall quality of the ongoing shot, derived from uniformity and channeling status. Updates live.

### Binary Sensors

*   **Connected (`binary_sensor.bookoo_connected`):** Indicates if Home Assistant is currently connected to the scale.
*   **NEW - Shot in Progress (`binary_sensor.bookoo_shot_in_progress`):** Indicates if an espresso shot session is currently active.
*   **NEW - Current Shot Pre-infusion Active (`binary_sensor.bookoo_current_shot_pre_infusion_active`):** Indicates if the pre-infusion phase is currently active during a shot. Updates live.

### Buttons

*   **Tare (`button.bookoo_tare`):** Tares the scale.
*   **Reset Timer (`button.bookoo_reset_timer`):** Resets the timer on the scale.
*   **Start Timer (`button.bookoo_start`):** Starts the timer on the scale (does not start a full HA shot session).
*   **Stop Timer (`button.bookoo_stop`):** Stops the timer on the scale (does not stop an HA shot session).
*   **Tare and Start Timer (`button.bookoo_tare_and_start`):** Tares the scale and then starts the timer on the scale (does not start a full HA shot session).
*   **NEW - Start Espresso Shot (`button.bookoo_start_shot_session`):** Manually initiates a full espresso shot session. This will tare the scale, start the timer on the scale, and begin logging all shot data.
*   **NEW - Stop Espresso Shot (`button.bookoo_stop_shot_session`):** Manually stops the currently active espresso shot session. This will stop the timer on the scale and finalize the shot data logging, firing the `bookoo_shot_completed` event.

## Services

This integration provides the following services that can be called from automations or scripts:

*   `bookoo.start_shot`
    *   **Description:** Manually initiates an espresso shot session. It tares the scale, starts the timer on the scale, and begins logging detailed shot data (including linked input parameters).
    *   **Service Data Fields:** None.

*   `bookoo.stop_shot`
    *   **Description:** Manually stops the currently active espresso shot session. It stops the timer on the scale, finalizes the shot data logging, and fires the `bookoo_shot_completed` event if the shot meets the minimum duration criteria.
    *   **Service Data Fields:** None.

## Events

This integration fires the following event:

*   `bookoo_shot_completed`
    *   **Description:** Fired when an espresso shot session is completed or terminated for reasons other than being too short. This can be triggered automatically by the scale's auto-timer stop, manually via the `bookoo.stop_shot` service/button, if the scale disconnects, or if the shot exceeds the configured `Maximum Shot Duration`. Shots that are shorter than the configured `Minimum Shot Duration` (and not due to disconnection or forced stop) will *not* fire this event but will update the 'Last Shot Status' sensor to 'aborted_too_short'.
    *   **Event Data (payload fields):**
        *   `device_id` (string): The Home Assistant device ID for the Bookoo scale.
        *   `entry_id` (string): The config entry ID for the Bookoo scale integration.
        *   `start_time_utc` (string): ISO 8601 formatted UTC timestamp of when the shot started.
        *   `end_time_utc` (string): ISO 8601 formatted UTC timestamp of when the shot ended.
        *   `duration_seconds` (float): Total duration of the shot in seconds (e.g., `28.75`).
        *   `final_weight_grams` (float): Final beverage weight in grams as reported by the scale at the end of the shot (e.g., `36.2`).
        *   `flow_profile` (list of lists/tuples): Time-series data of the flow rate. Each inner list/tuple is `[elapsed_seconds_since_shot_start, flow_rate_grams_per_second]`. Example: `[[0.5, 0.0], [1.0, 0.5], [1.5, 1.8], ...]`.
        *   `scale_timer_profile` (list of lists/tuples): Time-series data of the scale's internal timer reading. Each inner list/tuple is `[elapsed_seconds_since_shot_start, scale_timer_reading_milliseconds]`. Example: `[[0.5, 500], [1.0, 1000], ...]`.
        *   `input_parameters` (dictionary): Key-value pairs of data captured from any linked input helper entities at the start of the shot. Keys are `bean_weight` and `coffee_name`. Example: `{"bean_weight": "18.0", "coffee_name": "Ethiopia Yirgacheffe"}`.
        *   `start_trigger` (string): Indicates how the shot was initiated (e.g., `"scale_auto"`, `"ha_service"`).
        *   `stop_reason` (string): Indicates how the shot was stopped (e.g., `"scale_auto_dict"` (for scale's auto-timer stop via decoded event), `"ha_service"`, `"disconnected"`).
        *   `status` (string): The final status of the shot (e.g., `"completed"`, `"aborted_disconnected"`, `"aborted_too_long"`).
        *   `average_flow_rate_gps` (float): Average flow rate in grams per second for the shot.
        *   `peak_flow_rate_gps` (float): Peak flow rate in grams per second observed during the shot.
        *   `time_to_first_flow_seconds` (float | null): Time in seconds from shot start until flow rate first exceeds a small threshold (e.g., 0.2 g/s). Can be `null` if no significant flow is detected.
        *   `time_to_peak_flow_seconds` (float | null): Time in seconds from shot start when the peak flow rate was achieved. Can be `null` if no flow profile data is available.
