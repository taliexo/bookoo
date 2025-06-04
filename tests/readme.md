# Testing for the Bookoo Home Assistant Custom Component

This directory contains the automated tests for the Bookoo custom component. These tests help ensure the reliability and correctness of the component's functionality.

## Prerequisites

Before running the tests, ensure you have the following installed:
- Python (version specified in the project, e.g., 3.10+)
- `pip` for installing dependencies

It's highly recommended to use a Python virtual environment to manage project dependencies.

## Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install development dependencies:**
    The necessary dependencies for running tests and development are specified in `requirements_dev.txt` in the project root. For Home Assistant custom components, tests often rely on `pytest` and specific Home Assistant testing fixtures.

    Install them using:
    ```bash
    pip3 install -r requirements_dev.txt
    ```
    Runtime dependencies for the component itself are listed in `custom_components/bookoo/manifest.json` and are typically handled by Home Assistant when the component is loaded.

## Running Tests

All tests are run using `pytest`.

1.  **Run all tests:**
    Navigate to the root directory of the `bookoo` project (i.e., `/Users/peach/CascadeProjects/rewrite/bookoo/`) and run:
    ```bash
    python3 -m pytest
    ```
    Or, more verbosely with output capture disabled (`-s`) and verbose mode (`-v`):
    ```bash
    python3 -m pytest -s -v
    ```

2.  **Run tests in a specific file:**
    To run tests only within a particular file, specify the path to the file:
    ```bash
    python3 -m pytest tests/test_coordinator.py
    ```

3.  **Run a specific test class or function:**
    Use `::` to specify a class or function:
    ```bash
    python3 -m pytest tests/test_coordinator.py::TestBookooCoordinator
    python3 -m pytest tests/test_coordinator.py::TestBookooCoordinator::test_coordinator_initialization
    ```

4.  **Run tests with coverage:**
    If `pytest-cov` is installed, you can generate a coverage report:
    ```bash
    python3 -m pytest --cov=custom_components.bookoo --cov-report=html
    ```
    This will create an `htmlcov` directory with the coverage report.

## Test Structure

-   **`conftest.py`**: This file contains shared fixtures used across multiple test files. Fixtures help set up common preconditions for tests (e.g., mocked Home Assistant instances, mocked BookooScale objects, coordinator instances).
-   **`test_init.py`**: Tests for the main component setup (`async_setup_entry`, `async_unload_entry`) found in `custom_components/bookoo/__init__.py`.
-   **`test_coordinator.py`**: Tests for the `BookooCoordinator` class (`custom_components/bookoo/coordinator.py`), which manages data updates and communication with the Bookoo scale.
-   **`test_session_manager.py`**: Tests for the `SessionManager` class (`custom_components/bookoo/session_manager.py`), responsible for handling shot session logic.
-   **`test_config_flow.py`**: Tests for the component's configuration flow (`custom_components/bookoo/config_flow.py`), covering user setup and options.
-   **`test_sensor.py`**: Tests for any sensor entities provided by the Bookoo component.
-   **`test_analytics.py`**: Tests for shot analytics and quality scoring logic.

## Linting and Formatting

This project uses `ruff` for linting and formatting, and `mypy` for static type checking. These are typically run via pre-commit hooks. To run them manually:

```bash
ruff check .
ruff format .
mypy custom_components/bookoo/ tests/
```

Ensure all linters and tests pass before committing changes.
