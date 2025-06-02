# Bookoo Scale Integration

This is a Home Assistant integration for the BOOKOO Bluetooth scale. It provides sensors for weight, flow rate and the timer along with buttons for tare, timer start, stop, reset and "tare and start timer".

Based on the documentation at https://github.com/BooKooCode/OpenSource/

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
    *   Default: 5 seconds. Set to 0 to record all timer events.
*   **Linked Input Entities:**
    *   This section allows you to link existing Home Assistant `input_number`, `input_text`, or other helper entities to your espresso shots. When a shot is started, the current values of these linked entities will be captured and included in the `bookoo_shot_completed` event data.
    *   **How to use:**
        1.  Create your desired helper entities in Home Assistant (e.g., an `input_number.espresso_bean_dose` for bean weight, `input_text.espresso_coffee_name` for the coffee blend).
        2.  In the Bookoo integration options, select the corresponding helper entity from the dropdown list for each parameter you want to track (e.g., "Bean Weight Input Entity", "Coffee Name Input Entity").
    *   The values will be available in the `event_data.input_parameters` dictionary of the `bookoo_shot_completed` event.

## Entities

This integration provides the following entities:

### Sensors

*   **Weight (`sensor.bookoo_weight`):** Current weight on the scale (grams).
*   **Flow Rate (`sensor.bookoo_flow_rate`):** Current flow rate (mL/second).
*   **Timer (`sensor.bookoo_timer`):** Current reading of the scale's timer (seconds).
*   **Battery (`sensor.bookoo_battery`):** Battery level of the scale (%).
*   **NEW - Current Shot Duration (`sensor.bookoo_current_shot_duration`):** Live duration of the currently active espresso shot (seconds). Resets to 0 when no shot is active.
*   **NEW - Last Shot Duration (`sensor.bookoo_last_shot_duration`):** Duration of the last completed espresso shot (seconds).
*   **NEW - Last Shot Final Weight (`sensor.bookoo_last_shot_final_weight`):** Final beverage weight of the last completed shot (grams).
*   **NEW - Last Shot Start Time (`sensor.bookoo_last_shot_start_time`):** UTC timestamp when the last shot started.
*   **NEW - Last Shot Status (`sensor.bookoo_last_shot_status`):** Status of the last shot (e.g., "completed", "aborted_disconnected", "aborted_too_short").

### Binary Sensors

*   **Connected (`binary_sensor.bookoo_connected`):** Indicates if Home Assistant is currently connected to the scale.
*   **NEW - Shot in Progress (`binary_sensor.bookoo_shot_in_progress`):** Indicates if an espresso shot session is currently active.

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
    *   **Description:** Fired when an espresso shot session is completed (either automatically by the scale's auto-timer stop, manually via the `bookoo.stop_shot` service/button, or due to disconnection) and the shot's duration meets the configured `Minimum Shot Duration`.
    *   **Event Data (payload fields):**
        *   `device_id` (string): The Home Assistant device ID for the Bookoo scale.
        *   `entry_id` (string): The config entry ID for the Bookoo scale integration.
        *   `start_time_utc` (string): ISO 8601 formatted UTC timestamp of when the shot started.
        *   `end_time_utc` (string): ISO 8601 formatted UTC timestamp of when the shot ended.
        *   `duration_seconds` (float): Total duration of the shot in seconds (e.g., `28.75`).
        *   `final_weight_grams` (float): Final beverage weight in grams as reported by the scale at the end of the shot (e.g., `36.2`).
        *   `flow_profile_gps` (list of lists/tuples): Time-series data of the flow rate. Each inner list/tuple is `[elapsed_seconds_since_shot_start, flow_rate_grams_per_second]`. Example: `[[0.5, 0.0], [1.0, 0.5], [1.5, 1.8], ...]`.
        *   `scale_timer_profile_ms` (list of lists/tuples): Time-series data of the scale's internal timer reading. Each inner list/tuple is `[elapsed_seconds_since_shot_start, scale_timer_reading_milliseconds]`. Example: `[[0.5, 500], [1.0, 1000], ...]`.
        *   `input_parameters` (dictionary): Key-value pairs of data captured from any linked input helper entities at the start of the shot. Keys are derived from the option selected (e.g., `bean_weight_grams`, `coffee_name`). Example: `{"bean_weight_grams": "18.0", "coffee_name": "Ethiopia Yirgacheffe"}`.
        *   `start_trigger` (string): Indicates how the shot was initiated (e.g., `"scale_auto"`, `"ha_service"`).
        *   `stop_reason` (string): Indicates how the shot was stopped (e.g., `"scale_auto_stop"`, `"ha_service"`, `"disconnected"`).
        *   `status` (string): The final status of the shot (e.g., `"completed"`, `"aborted_disconnected"`). Note: Shots aborted for being too short (less than `Minimum Shot Duration`) do not fire this event.

## Example Automations

### 1. Log Shot Data to InfluxDB

This example assumes you have the InfluxDB integration set up in Home Assistant.

```yaml
automation:
  - alias: "Log Bookoo Espresso Shot to InfluxDB"
    trigger:
      - platform: event
        event_type: bookoo_shot_completed
    action:
      - service: influxdb.write
        data_template:
          measurement: "espresso_shot"
          tags:
            device_id: "{{ trigger.event.data.device_id }}"
            coffee_name: "{{ trigger.event.data.input_parameters.coffee_name | default('Unknown') }}"
            start_trigger: "{{ trigger.event.data.start_trigger }}"
            stop_reason: "{{ trigger.event.data.stop_reason }}"
            status: "{{ trigger.event.data.status }}"
          fields:
            duration_seconds: "{{ trigger.event.data.duration_seconds }}"
            final_weight_grams: "{{ trigger.event.data.final_weight_grams }}"
            bean_weight_grams: "{{ trigger.event.data.input_parameters.bean_weight_grams | float(0) }}"
            # For flow_profile_gps and scale_timer_profile_ms, you might need a more complex
            # script or AppDaemon app to parse and log time-series data effectively if InfluxDB's
            # direct write service doesn't handle list-of-lists well for fields.
            # Alternatively, log key metrics like average flow rate if calculated and added to event.
          timestamp: "{{ trigger.event.data.start_time_utc }}" # Use shot start time for the InfluxDB point
```

**Note on Time-Series Data:** Logging the full `flow_profile_gps` and `scale_timer_profile_ms` arrays directly into a single InfluxDB field might not be optimal for querying. Consider:
*   Calculating aggregate metrics (e.g., average flow rate, time to first drip) in the coordinator and adding them to the event payload.
*   Using a Python script or AppDaemon app triggered by the event to process these arrays and write multiple points to InfluxDB if detailed time-series analysis is required.

### 2. Send a Notification on Shot Completion

```yaml
automation:
  - alias: "Notify on Espresso Shot Completion"
    trigger:
      - platform: event
        event_type: bookoo_shot_completed
    action:
      - service: notify.mobile_app_your_phone # Replace with your notification service
        data:
          title: "Espresso Shot Completed!"
          message: >
            Shot duration: {{ trigger.event.data.duration_seconds }}s, 
            Weight: {{ trigger.event.data.final_weight_grams }}g. 
            {{ trigger.event.data.input_parameters.coffee_name | default('') }} 
            ({{ trigger.event.data.input_parameters.bean_weight_grams | default('?') }}g beans).
```
