# Bookoo Home Assistant Integration Architecture

## 1. Overview

The Bookoo Home Assistant integration allows users to connect their Bookoo smart espresso scales to Home Assistant. It provides real-time data such as weight, flow rate, and shot timings, enables control over the scale (e.g., tare, power), and manages espresso shot sessions including detailed analytics and history.

## 2. Core Components

-   **`BookooCoordinator` (`coordinator.py`)**: This is the central orchestrator of the integration. It manages:
    -   Bluetooth Low Energy (BLE) connection to the Bookoo scale via the `aiobookoov2` library.
    -   Parsing incoming data from the scale (weight, commands, timer).
    -   Regularly updating Home Assistant entities with the latest state.
    -   Handling service calls (e.g., start/stop shot, tare).
    -   Coordinating with the `SessionManager` for shot lifecycle events.
    -   Managing configuration options and their updates.

-   **`SessionManager` (`session_manager.py`)**: Responsible for the logic and state of an active espresso shot session. It handles:
    -   Starting and stopping shot sessions based on triggers (services, auto-timer from scale).
    -   Collecting time-series data for weight, flow rate, and scale timer during a shot.
    -   Calculating post-shot analytics (e.g., average flow, peak flow, channeling detection) using `ShotAnalyzer`.
    -   Preparing the data payload for the `bookoo_shot_completed` event.
    -   Interfacing with `storage.py` to save shot records.

-   **`aiobookoov2` (External Library)**: The underlying Python library that handles the direct BLE communication with the Bookoo scale. It abstracts the complexities of Bluetooth characteristics, notifications, and command encoding/decoding.

-   **Config Flow (`config_flow.py`)**: Manages the user setup process for the integration. This includes:
    -   Discovering Bookoo scales via Bluetooth.
    -   Guiding the user through the connection and initial setup.
    -   Handling options flow for users to customize integration settings post-setup (e.g., linked entities, auto-stop parameters, timeouts).

-   **`ShotAnalyzer` (`analytics.py`)**: A class responsible for performing detailed analysis on completed shot data, such as detecting channeling, identifying pre-infusion, and calculating extraction uniformity.

## 3. Entities

The integration exposes various entities to Home Assistant:

-   **Sensors (`sensor.py`)**: Provide real-time and post-shot information, including:
    -   Current weight on the scale.
    -   Live flow rate (calculated).
    -   Shot duration.
    -   Status messages from the scale.
    -   Real-time analytics (e.g., channeling status, pre-infusion active).
    -   Metrics from the last completed shot (e.g., final weight, average flow rate, shot quality score).

-   **Buttons (`button.py`)**: Allow users to trigger actions on the scale:
    -   Start Shot Session (if not auto-detected).
    -   Stop Shot Session.
    -   Tare Scale.
    -   Power Off Scale (if supported by scale firmware).
    -   Other direct commands (e.g., reset timer).

## 4. Data Flow

1.  **Scale to `aiobookoov2`**: The Bookoo scale sends data (weight, timer, command responses) via BLE notifications.
2.  **`aiobookoov2` to `BookooCoordinator`**: `aiobookoov2` decodes this data and passes it to `BookooCoordinator` via callbacks (`characteristic_update_callback`).
3.  **`BookooCoordinator` Processing**: The coordinator processes this raw data. If a shot is active, it passes relevant data points (weight, time) to the `SessionManager`.
4.  **`SessionManager` During Shot**: `SessionManager` appends data to profiles (flow, weight, timer). The `BookooCoordinator` may also use `ShotAnalyzer` for real-time insights (e.g., pre-infusion, channeling) based on these profiles.
5.  **Coordinator Updates Entities**: `BookooCoordinator` calls `async_update_listeners()` to signal Home Assistant that its state (and thus the state of its entities) has changed. Entities then fetch the latest data from the coordinator.
6.  **Shot Completion**: When a shot ends (via service call, scale button, or auto-stop):
    -   `SessionManager` finalizes data collection.
    -   `BookooCoordinator` (or `SessionManager` via `ShotAnalyzer`) calculates final shot analytics.
    -   `SessionManager` prepares a `BookooShotCompletedEventDataModel`.
    -   This model instance is dispatched as a `bookoo_shot_completed` Home Assistant event.
    -   The data is also passed to `storage.py` to be saved persistently using Home Assistant's `Store`.

## 5. Key Events

-   **`bookoo_shot_completed`**: Fired when a shot session is completed or aborted.
    -   **Payload**: Contains a serialized `BookooShotCompletedEventDataModel` (from `types.py`), including all session parameters, profiles (flow, weight, timer), and calculated analytics.
    -   **Purpose**: Allows users or other automations to react to completed shots and use the detailed shot data.

## 6. Configuration

-   **`manifest.json`**: Defines integration metadata, dependencies, and versioning.
-   **`const.py`**: Contains constants used throughout the integration (e.g., domain name, service names, default option values, configuration keys).
-   **Options Flow (`config_flow.py -> BookooOptionsFlowHandler`)**: Allows runtime configuration of:
    -   Linked entities (e.g., `input_number` for bean weight, `input_text` for coffee name) to associate with shot data.
    -   Auto-stop feature parameters (thresholds, durations for flow stability and cutoff).
    -   Bluetooth connection and command timeouts.

## 7. Storage

-   **`storage.py`**: Handles the persistent storage of completed shot records.
    -   Uses Home Assistant's `Store` helper class, which saves data as JSON in the `.storage` directory.
    -   Provides functions to add new shot records and retrieve historical shots.
    -   Previously used an SQLite database, but migrated to `Store` for better HA integration.
