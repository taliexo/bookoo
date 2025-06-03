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
    *   Default: 10 seconds (as per current coordinator code). Set to 0 to record all timer events.
*   **Linked Input Entities:**
    *   This section allows you to link existing Home Assistant `input_number`, `input_text`, or other helper entities to your espresso shots. When a shot is started, the current values of these linked entities will be captured and included in the `bookoo_shot_completed` event data.
    *   **How to use:**
        1.  Create your desired helper entities in Home Assistant (e.g., an `input_number.espresso_bean_dose` for bean weight, `input_text.espresso_coffee_name` for the coffee blend).
        2.  In the Bookoo integration options, select the corresponding helper entity from the dropdown list for each parameter you want to track (e.g., "Bean Weight Input Entity", "Coffee Name Input Entity").
    *   The values will be available in the `input_parameters` dictionary of the `bookoo_shot_completed` event data (e.g., `event_data.input_parameters.bean_weight`).

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
*   **NEW - Last Shot Status (`sensor.bookoo_last_shot_status`):** Status of the last shot (e.g., "completed", "aborted_disconnected", "aborted_too_short").
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
    *   **Description:** Fired when an espresso shot session is completed. This can be triggered automatically by the scale's auto-timer stop, manually via the `bookoo.stop_shot` service/button, or if the scale disconnects during an active shot. Shots that are shorter than the configured `Minimum Shot Duration` (and not due to disconnection or forced stop) will *not* fire this event but will update the 'Last Shot Status' sensor to 'aborted_too_short'.
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
        *   `status` (string): The final status of the shot (e.g., `"completed"`, `"aborted_disconnected"`).
        *   `average_flow_rate_gps` (float): Average flow rate in grams per second for the shot.
        *   `peak_flow_rate_gps` (float): Peak flow rate in grams per second observed during the shot.
        *   `time_to_first_flow_seconds` (float | null): Time in seconds from shot start until flow rate first exceeds a small threshold (e.g., 0.2 g/s). Can be `null` if no significant flow is detected.
        *   `time_to_peak_flow_seconds` (float | null): Time in seconds from shot start when the peak flow rate was achieved. Can be `null` if no flow profile data is available.

## Example Automations

### 1. Log Shot Data via MQTT

A flexible way to log your espresso shot data is to publish it to an MQTT topic. From there, various tools (like Telegraf, Node-RED, or custom scripts) can subscribe to this topic and write the data to InfluxDB or other databases.

**Example Automation: Publish Shot Data to MQTT**

This automation will publish the full `EVENT_BOOKOO_SHOT_COMPLETED` event data as a JSON payload to a specified MQTT topic.

```yaml
alias: Publish Bookoo Espresso Shot to MQTT
trigger:
  - platform: event
    event_type: bookoo_shot_completed
action:
  - service: mqtt.publish
    data:
      topic: "bookoo/shot/completed" # Choose your desired MQTT topic
      payload_template: "{{ trigger.event.data | tojson }}" # Convert event data to JSON string
      retain: false # Set to true if you want the last message to be retained
```

**Consuming MQTT Data and Sending to InfluxDB:**

Once the data is on the MQTT topic, you have several options to get it into InfluxDB:

1.  **Telegraf:**
    *   Telegraf has an [MQTT consumer input plugin](https://github.com/influxdata/telegraf/tree/master/plugins/inputs/mqtt_consumer) and an [InfluxDB output plugin](https://github.com/influxdata/telegraf/tree/master/plugins/outputs/influxdb_v2).
    *   You can configure Telegraf to subscribe to `bookoo/shot/completed`, parse the JSON payload (using `data_format = "json"` and potentially `json_v2` processor for complex parsing/tagging), and write it to InfluxDB. This is a very popular and efficient method.

2.  **Node-RED:**
    *   Use an "MQTT in" node to subscribe to the topic.
    *   Use a "function" node to transform the JSON payload if needed.
    *   Use an "InfluxDB out" node to write the data.

3.  **Custom Script (Python, etc.):**
    *   Write a script that subscribes to the MQTT topic using a client library (e.g., Paho MQTT for Python) and then writes to InfluxDB using an InfluxDB client library.

**Advantages of MQTT:**
*   **Decoupling:** Home Assistant only needs to publish the data; other services handle the ingestion into InfluxDB.
*   **Flexibility:** Multiple subscribers can consume the shot data for different purposes.
*   **Resilience:** If your InfluxDB instance is temporarily down, MQTT can often buffer messages (depending on broker configuration and QoS).

This MQTT approach provides a clean separation of concerns and leverages a standard messaging protocol.

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
            Weight: {{ trigger.event.data.final_weight_grams }}g,
            Avg Flow: {{ trigger.event.data.average_flow_rate_gps }} g/s.
            {{ trigger.event.data.input_parameters.coffee_name | default('') }}
            ({{ trigger.event.data.input_parameters.bean_weight | default('?') }}g beans).
```

## Template Sensor Examples

Leverage Home Assistant's template sensors to create customized sensors based on the data provided by the Bookoo integration. Here are a few examples to get you started. You would typically add these to your `configuration.yaml` under the `template:` section, or in a dedicated `templates.yaml` file.

### 1. Shot Quality Assessment (Text)

This sensor provides a human-readable assessment of the current espresso shot's quality based on the `sensor.bookoo_current_shot_quality_score` and `sensor.bookoo_current_shot_channeling_status`.

```yaml
template:
  - sensor:
      - name: "Bookoo Shot Quality Assessment"
        unique_id: bookoo_shot_quality_assessment
        icon: mdi:coffee-check-outline
        state: >
          {% set score = states('sensor.bookoo_current_shot_quality_score') | float(0) %}
          {% set channeling = states('sensor.bookoo_current_shot_channeling_status') %}
          {% set is_shot_active = is_state('binary_sensor.bookoo_shot_in_progress', 'on') %}

          {% if not is_shot_active %}
            Idle
          {% elif score >= 90 and channeling == 'No Channeling' %}
            Excellent! ({{ score | round(0) }}%)
          {% elif score >= 80 and (channeling == 'No Channeling' or channeling == 'Mild Channeling') %}
            Great Shot ({{ score | round(0) }}%){% if channeling == 'Mild Channeling' %} - Mild Channeling{% endif %}
          {% elif score >= 70 %}
            Good Shot ({{ score | round(0) }}%){% if channeling not in ['No Channeling', 'Unknown'] %} - {{ channeling }}{% endif %}
          {% elif score >= 50 %}
            Fair Shot ({{ score | round(0) }}%){% if channeling not in ['No Channeling', 'Unknown'] %} - {{ channeling }}. Consider grind/tamp.{% else %} - Consider grind/tamp.{% endif %}
          {% elif score > 0 %}
            Poor Shot ({{ score | round(0) }}%){% if channeling not in ['No Channeling', 'Unknown'] %} - Significant {{ channeling }}. Check puck prep.{% else %} - Check puck prep.{% endif %}
          {% else %}
            Awaiting Data
          {% endif %}
        attributes:
          quality_score: "{{ states('sensor.bookoo_current_shot_quality_score') | float(None) }}"
          channeling_status: "{{ states('sensor.bookoo_current_shot_channeling_status') }}"
          shot_active: "{{ is_state('binary_sensor.bookoo_shot_in_progress', 'on') }}"
```

*(More template sensor examples can be added here.)*

## Querying Data from InfluxDB

If you are logging your espresso shot data to InfluxDB (e.g., via MQTT and Telegraf as described above), here are some InfluxQL queries you can use. These assume your Telegraf (or other consumer) configuration results in a measurement named `espresso_shot` with tags and fields derived from the JSON payload.

**Assumed Measurement (after MQTT processing):** `espresso_shot`

**Measurement:** `espresso_shot`

**Tags:**
*   `device_id`
*   `coffee_name`
*   `bean_weight_grams_input`
*   `start_trigger`
*   `stop_reason`
*   `status`

**Fields:**
*   `duration_seconds`
*   `final_weight_grams`
*   `average_flow_rate_gps`
*   `peak_flow_rate_gps`
*   `time_to_first_flow_seconds`
*   `time_to_peak_flow_seconds`
*   `bean_weight_grams` (field version of the input, converted to float)

### Example InfluxQL Queries

1.  **Get all data for the last 5 completed shots:**
    ```sql
    SELECT * FROM "espresso_shot" WHERE "status" = 'completed' ORDER BY time DESC LIMIT 5
    ```

2.  **Calculate average shot duration and final weight for today:**
    ```sql
    SELECT MEAN("duration_seconds") AS "avg_duration", MEAN("final_weight_grams") AS "avg_weight" FROM "espresso_shot" WHERE time >= today() AND "status" = 'completed'
    ```

3.  **Get all data for shots using a specific coffee (e.g., "Ethiopia Yirgacheffe"):**
    ```sql
    SELECT * FROM "espresso_shot" WHERE "coffee_name" = 'Ethiopia Yirgacheffe' AND "status" = 'completed' ORDER BY time DESC
    ```

4.  **Calculate average flow rate for shots where bean dose was 18g:**
    ```sql
    SELECT MEAN("average_flow_rate_gps") AS "avg_flow_rate" FROM "espresso_shot" WHERE "bean_weight_grams_input" = '18.0' AND "status" = 'completed'
    ```
    *Note: `bean_weight_grams_input` is a tag and stores the value as a string. If you logged `bean_weight_grams` as a field (float), you could query it as `WHERE "bean_weight_grams" = 18.0`.*

5.  **Count the number of shots per day for the last 7 days:**
    ```sql
    SELECT COUNT("duration_seconds") AS "num_shots" FROM "espresso_shot" WHERE time >= now() - 7d AND "status" = 'completed' GROUP BY time(1d) fill(0)
    ```

6.  **Get shots with an average flow rate greater than 2.0 g/s:**
    ```sql
    SELECT * FROM "espresso_shot" WHERE "average_flow_rate_gps" > 2.0 AND "status" = 'completed' ORDER BY time DESC
    ```
