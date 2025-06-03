# Resource Monitor (`resource_monitor.py`)

## Introduction

The `ResourceMonitor` class is a component of the MindX system (Augmentic Project) designed to continuously track key system resources: CPU utilization, virtual memory usage, and disk space usage across one or more configured file system paths. It operates asynchronously, providing real-time data and triggering alert/resolve callbacks when usage levels cross defined thresholds. This aids in maintaining system stability and providing insights for performance optimization or self-improvement actions.

## Explanation

### Core Features

-   **Multi-Resource Monitoring:**
    -   **CPU:** Tracks overall system CPU utilization percentage.
    -   **Memory:** Tracks overall system virtual memory utilization percentage.
    -   **Disk:** Monitors disk usage percentage for a configurable list of file system paths (e.g., `/`, `/var/log`, `/data`). Each path can have its own specific alert threshold.
-   **Configurable Thresholds:**
    -   Alert thresholds for CPU, memory, and individual disk paths are loaded from the global `Config` object (e.g., `monitoring.resource.max_cpu_percent`, `monitoring.resource.disk_paths` which can specify per-path thresholds).
    -   These limits can be dynamically updated at runtime via `set_resource_limits()`.
-   **Asynchronous Operation:**
    -   The core monitoring logic runs in a dedicated `asyncio.Task` (`_monitor_resources_loop`).
    -   Resource querying (using `psutil`) is performed asynchronously using `loop.run_in_executor` to prevent blocking the main event loop, especially for potentially blocking disk I/O operations.
-   **Callback Mechanism for Alerts & Resolutions:**
    -   `register_alert_callback(callback)`: Allows other system components to register asynchronous callback functions. These callbacks are invoked when a resource usage exceeds its defined threshold (respecting debouncing logic).
    -   `register_resolve_callback(callback)`: Registers callbacks that are invoked when a resource, previously in an alert state, returns to usage levels below its threshold.
    -   **Callback Signature:** Callbacks are expected to be `async def` and receive arguments: `(monitor_instance: ResourceMonitor, resource_type: ResourceType, current_value: float, path_if_disk: Optional[str] = None)`. This provides rich context to the handler.
-   **Alert Debouncing/Throttling:**
    -   To avoid excessive notifications ("alert spam") if a resource remains critical, an alert for a specific resource is only re-triggered after a configurable `re_alert_interval_seconds` has passed since the last alert for that same resource.
    -   The system tracks the "alert active" state for each resource (or disk path).
-   **Dynamic Usage Updates & Querying:**
    -   Internal metrics (`cpu_usage`, `memory_usage`, `disk_usage_map`) are updated periodically by the monitoring loop.
    -   `update_current_usage()` can be called manually (e.g., by `get_resource_usage`) to get the most recent readings outside the regular interval if needed.
    -   `get_resource_usage()`: Returns a dictionary with the latest known resource usage percentages.
    -   `get_resource_limits()`: Returns a dictionary of the currently configured alert thresholds.
-   **Error Handling:**
    -   The monitoring loop includes `try-except` blocks to catch errors during resource querying or within callback executions, preventing the monitor itself from crashing.
    -   Handles `FileNotFoundError` for configured disk paths that may no longer exist, optionally removing them from the watch list.
-   **Singleton Pattern:** The module provides `get_resource_monitor()` (synchronous, for initial setup) and `get_resource_monitor_async()` (asynchronous, preferred for use within async code) factory functions to access a single, shared instance of the `ResourceMonitor`. A `reset_instance_async()` class method is available for testing.

### `ResourceType` Enum
An internal `ResourceType` enum (`CPU`, `MEMORY`, `DISK`) is used to categorize resource types, primarily for passing to callbacks and managing internal alert states.

## Technical Details

-   **Dependencies:** Uses the `psutil` library (`pip install psutil`) for querying system resource information.
-   **Path Handling:** Employs `pathlib.Path` and `os.path.normpath` for consistent handling of disk paths. Paths are resolved to absolute, normalized forms for use as dictionary keys and in `psutil` calls. `Path.resolve(strict=False)` is used for configured disk paths to allow monitoring of mount points that might not "exist" in the VFS until mounted.
-   **Configuration Keys (from `Config`):**
    -   `monitoring.resource.enabled`: (bool) Master switch for starting monitoring on init.
    -   `monitoring.resource.interval`: (float) Seconds between resource checks.
    -   `monitoring.resource.max_cpu_percent`: (float) CPU alert threshold.
    -   `monitoring.resource.max_memory_percent`: (float) Memory alert threshold.
    -   `monitoring.resource.disk_paths`: (list) Can be `["/", "/data"]` or `[{"path": "/", "threshold": 88.0}, {"path": "/data", "threshold": 92.0}]`.
    -   `monitoring.resource.max_disk_percent`: (float) Default disk threshold if not specified per-path.
    -   `monitoring.resource.re_alert_interval_seconds`: (float) Cooldown before re-alerting for a sustained issue.
-   **Concurrency:** An `asyncio.Lock` (`_lock`) is used by the factory functions (`get_resource_monitor_async`) to ensure thread-safe/task-safe singleton instantiation. Callbacks are individually wrapped in `try-except` to prevent one failing callback from disrupting others or the monitor loop.

## Usage

1.  **Obtaining an Instance:**
    ```python
    # In async code (preferred):
    # from mindx.monitoring.resource_monitor import get_resource_monitor_async
    # monitor = await get_resource_monitor_async()

    # In sync code (e.g., application startup):
    from mindx.monitoring.resource_monitor import get_resource_monitor
    monitor = get_resource_monitor()
    ```

2.  **Starting/Stopping Monitoring:**
    Typically, the `CoordinatorAgent` will start monitoring based on configuration.
    ```python
    # monitor.start_monitoring(interval=10.0) # Default interval from config if not passed
    # ...
    # await monitor_instance.shutdown_for_reset() # For testing or clean shutdown of the task
    # monitor.stop_monitoring() # Signals the loop to end
    ```

3.  **Registering Callbacks:**
    ```python
    from mindx.monitoring.resource_monitor import ResourceMonitor, ResourceType, Optional

    async def my_critical_alert_handler(
        monitor: ResourceMonitor, 
        rtype: ResourceType, 
        value: float, 
        path: Optional[str] = None
    ):
        path_info = f" on path '{path}'" if path else ""
        print(f"CRITICAL ALERT HANDLER: {rtype.name}{path_info} is at {value:.1f}%!")
        # Example: current_all_usage = monitor.get_resource_usage()
        # print(f"Full context: {current_all_usage}")
        # if rtype == ResourceType.MEMORY and value > 95.0:
        #     print("Attempting emergency cleanup or shutdown signal!")

    monitor.register_alert_callback(my_critical_alert_handler)
    ```

4.  **Querying Current Usage/Limits:**
    ```python
    current_usage = monitor.get_resource_usage()
    print(f"CPU: {current_usage['cpu_percent']:.1f}%")
    for disk_path, usage_percent in current_usage.get('disk_usage_map', {}).items():
        print(f"Disk '{disk_path}': {usage_percent:.1f}%")
    ```

The `ResourceMonitor` is designed to be a reliable, low-overhead component providing essential system health telemetry to the broader MindX system, enabling proactive responses and informing self-improvement strategies.
