# LLM Performance Monitor (`performance_monitor.py`) - Production Candidate

## Introduction

The `PerformanceMonitor` is a specialized component within the MindX system (Augmentic Project) responsible for tracking, aggregating, and reporting on the performance of interactions with Large Language Models (LLMs). It collects detailed metrics such as latency, token usage, cost, success rates, and error types, providing crucial insights for system optimization, model selection strategies, and informing self-improvement processes.

## Explanation

### Core Features

-   **Contextual Metric Tracking:**
    -   Metrics are logged against a flexible key that can include `model_id` (e.g., "gemini-1.5-pro"), `task_type` (e.g., "code_generation", "system_analysis"), and `initiating_agent_id` (e.g., "SelfImprovementAgent_v3.5"). This allows for fine-grained performance analysis.
    -   Tuple keys like `("gemini-1.5-pro", "code_generation")` are serialized to strings (e.g., `"gemini-1.5-pro|code_generation"`) for JSON persistence.
-   **Comprehensive Metrics Collection:** For each unique key, the monitor tracks:
    -   `requests`: Total number of LLM calls.
    -   `successes` & `failures`: Counts of successful and failed requests.
    -   `total_latency_ms`: Sum of all request latencies in milliseconds for better precision.
    -   `total_input_tokens` & `total_output_tokens`: Aggregated token counts.
    -   `total_cost_usd`: Aggregated estimated costs in USD (if this information is provided with each request).
    -   `error_types`: A dictionary counting occurrences of different error types (e.g., "APIError", "TimeoutError", "RateLimitError") for failed requests.
    -   `recent_latencies_ms_samples`: A list storing the N (configurable via `monitoring.performance.max_recent_latencies_samples`, default 100) most recent latencies. This sample is used for calculating approximate P50, P90, and P99 latency percentiles.
    -   `requests_since_last_save`: Internal counter for batched saving.
-   **Derived Metrics Calculation (`get_metrics_for_key`):** This synchronous method computes:
    -   `success_rate`.
    -   Average latency (in milliseconds).
    -   P50 (median), P90, P99 latency percentiles using Python's `statistics` module on the `recent_latencies_ms_samples`.
    -   Average input, output, and total tokens per request.
    -   Average cost (`avg_cost_usd`) per request and the `total_cost_usd` for the given key.
-   **Persistence:**
    -   Metrics are persisted to a JSON file. The path is configurable via `monitoring.performance.metrics_file_path` (default: `PROJECT_ROOT/data/performance_metrics.json`). Persistence can be disabled.
    -   Metrics are loaded from this file upon `PerformanceMonitor` initialization.
    -   **Saving Strategies:**
        1.  **Batched by Count (if periodic save off):** Saves after every N requests for a specific metric key (N configured by `monitoring.performance.save_every_n_requests`).
        2.  **Periodic Asynchronous Save:** If enabled (`monitoring.performance.enable_periodic_save` is true and interval > 0), a background `asyncio.Task` (`_periodic_save_worker`) saves all metrics at a configurable interval (`monitoring.performance.periodic_save_interval_seconds`). This is generally preferred for high-throughput systems to reduce frequent I/O.
        3.  **On Shutdown:** A final save is performed when the monitor's `shutdown()` method is called.
-   **Reporting (`get_performance_report`):** Generates a human-readable multi-line string report summarizing all tracked metrics, including overall statistics and per-key details, sorted by request count.
-   **Singleton Pattern:** Accessed via `get_performance_monitor()` (synchronous) or `get_performance_monitor_async()` (asynchronous factory functions) to ensure a single instance across the application. `reset_instance_async()` is available for testing.
-   **Asynchronous Operations:**
    -   `record_request` is `async` as it might trigger an asynchronous save operation (`_save_metrics_async`).
    -   File operations for loading/saving are performed asynchronously using `loop.run_in_executor` to avoid blocking the main event loop.
    -   An `asyncio.Lock` (`_lock`) protects concurrent access to the shared metrics data during updates and file operations.

## Technical Details

-   **Data Structure:** Uses `defaultdict` for flexible storage of metrics. The nested `error_types` is also a `defaultdict(int)`.
-   **Configuration:** Relies on `mindx.utils.config.Config` for all its settings (file paths, save intervals, sample sizes, persistence toggles).
-   **Latency Percentiles:** Calculated using `statistics.median` and `statistics.quantiles` on the `recent_latencies_ms_samples`. This provides an approximation based on recent performance. For very high accuracy over all historical data, more advanced streaming percentile algorithms (e.g., t-digest, HDRHistogram) would be required.
-   **Error Handling:** Includes `try-except` blocks for file operations, JSON parsing, and SDK import attempts.
-   **Shutdown Protocol:** The `shutdown()` method ensures the periodic save task is cancelled and a final save of metrics is attempted.

## Usage

1.  **Obtaining an Instance:**
    ```python
    # In async code (preferred for consistency with monitor's async nature):
    # from mindx.monitoring.performance_monitor import get_performance_monitor_async
    # perf_monitor = await get_performance_monitor_async()

    # In sync code (e.g., application startup, less ideal if async features are core):
    from mindx.monitoring.performance_monitor import get_performance_monitor
    perf_monitor = get_performance_monitor()
    ```

2.  **Recording an LLM Request:**
    This is typically done by the `LLMHandler` or any component that directly makes LLM calls.
    ```python
    # Example after an LLM call to "gemini-pro" for a "translation" task
    await perf_monitor.record_request(
        model_id="gemini-pro",
        success=True,
        latency_seconds=2.15,
        input_tokens=120,
        output_tokens=350,
        cost_usd=0.0007, # Example cost
        task_type="translation",
        initiating_agent_id="TranslationServiceV1",
        error_type=None # Or "APITimeout", "ContentFilter" if success=False
    )
    ```

3.  **Accessing Metrics & Reports:**
    ```python
    # Get metrics for a specific model and task type
    # Key construction must match how it was recorded
    key_tuple = ("gemini-pro", "translation", "TranslationServiceV1") 
    metrics = perf_monitor.get_metrics_for_key(key_tuple)
    if metrics["requests"] > 0:
        print(f"Success rate for {metrics['key_str']}: {metrics['success_rate']:.2%}")
        print(f"P90 Latency: {metrics['p90_latency_ms'] / 1000.0:.3f}s")

    # Get all metrics (keys are serialized strings)
    # all_metrics_data = perf_monitor.get_all_metrics()
    # for serialized_key, metric_values in all_metrics_data.items():
    #     print(f"Data for {serialized_key}: {metric_values['requests']} requests.")

    # Get human-readable report
    # report_string = perf_monitor.get_performance_report()
    # print(report_string)
    ```

4.  **Resetting Metrics (e.g., for testing or periodic archive & reset):**
    ```python
    # Reset metrics for all keys related to "gemini-pro"
    # await perf_monitor.reset_metrics(key_prefix_serialized="gemini-pro") 
    
    # Reset all metrics entirely
    # await perf_monitor.reset_metrics() 
    ```

5.  **Application Shutdown:**
    It's crucial to call `shutdown()` to ensure any pending metrics are saved and background tasks are stopped cleanly.
    ```python
    # In your application's main asynchronous shutdown sequence:
    # perf_monitor = await get_performance_monitor_async() # Get instance
    # await perf_monitor.shutdown()
    ```

The `PerformanceMonitor` is a cornerstone for understanding and improving the efficiency and reliability of LLM usage within the MindX system. Its data can directly feed into the `CoordinatorAgent`'s system analysis, driving targeted self-improvement initiatives.
