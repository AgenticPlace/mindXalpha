# Multi-Model Agent (MMA) (`multimodel_agent.py`) - Production Candidate Stub v2

## Introduction

The `MultiModelAgent` (MMA) is a key component in the MindX system's orchestration layer (Augmentic Project). It is responsible for managing and executing tasks that require interaction with various Large Language Models (LLMs). Its core functionalities include:

-   Loading and managing definitions of available LLMs and their capabilities from external YAML configuration files.
-   Interacting with a `ModelRegistry` to obtain concrete LLM client handlers.
-   Employing a `ModelSelector` to choose the most appropriate LLM for a given task based on capabilities, runtime performance, cost, and specific task requirements.
-   Processing tasks, including a robust retry mechanism for transient failures.
-   Updating dynamic runtime statistics for models (success rate, latency) to inform future selections.
-   Optionally managing an internal priority queue for tasks, processed by asynchronous worker co-routines.

This version represents a more detailed and functional stub, ready for integration and further development of its peripheral dependencies like the `ModelRegistry` and specific `LLMHandler` implementations.

## Explanation

### Core Components & Workflow

1.  **`TaskType`, `TaskPriority`, `TaskStatus` Enums:** Standardized enumerations defining the nature, importance, and lifecycle state of tasks handled by the MMA.
2.  **`Task` Data Class:** Encapsulates all information related to a single unit of work for an LLM. Key attributes include:
    *   `task_id`, `task_type`, `prompt`, `priority`.
    *   `context`: Arbitrary dictionary for task-specific data. Can include `generation_params` to override default LLM call settings.
    *   `requirements`: Dictionary specifying constraints for model selection (e.g., `min_context_length`, `target_model_id`, `supports_streaming`).
    *   `status`, `assigned_model` (full ID like "provider/model_api_name"), `result`, `error`, timestamps, `attempts`, `max_attempts`.
    *   `callback`: An optional asynchronous function to call upon task completion or failure.
    *   `history`: A log of significant events during the task's processing lifecycle.
3.  **`ModelCapability` Data Class:** Stores static definitions and dynamic runtime statistics for each LLM known to the system.
    *   **Static (from YAML):** `model_id` (e.g., "ollama/llama3:8b-instruct"), `provider`, `model_name_for_api` (the specific name/tag used in API calls), `capabilities` (a dictionary mapping `TaskType` enum to a proficiency score, 0.0-1.0), `resource_usage` (e.g., `{"cost_per_kilo_total_tokens": 0.02}`), `max_context_length`, `supports_streaming`, `supports_function_calling`.
    *   **Dynamic:** `availability`, `success_rate`, `average_latency_ms`. These are initialized from config (if available under `orchestration.multimodel_agent.model_initial_stats.<model_id>`) or to defaults, and then updated after each task execution using an exponential moving average (`update_runtime_stats`).
    *   The `update_runtime_stats` method also interfaces with the global `PerformanceMonitor` to log detailed metrics.
    *   `from_dict` class method parses capability data loaded from provider-specific YAML files.
4.  **Initialization (`__init__` and `async initialize`):**
    *   **Synchronous `__init__`:** Sets up basic attributes, `Config`, `BeliefSystem`, `ModelRegistry`, and `ModelSelector` instances. Defines the path to model capability configuration files (e.g., `PROJECT_ROOT/data/mindx_models_config/`).
    *   **Asynchronous `initialize()`: This method MUST be called after creating an MMA instance.**
        *   `_load_model_capabilities_from_files()`: Scans the `models_config_dir` (e.g., `data/mindx_models_config/`) for provider-specific YAML files (e.g., `ollama.yaml`, `gemini.yaml`). Each file is expected to have a top-level `models` key, under which individual models for that provider are defined with their static capabilities. `ModelCapability` objects are created and stored in `self.model_capabilities`.
        *   `_initialize_model_handlers_from_registry()`: Populates `self.model_handlers` (mapping provider names to `LLMInterface` instances) by retrieving initialized handlers from the `ModelRegistry`. This ensures that the MMA uses centrally managed and configured LLM clients.
        *   Starts asynchronous task worker loops (`_task_worker_loop`) if `num_workers` (from config) is greater than 0.

5.  **Task Creation & Submission:**
    *   `create_task()`: A factory method to instantiate `Task` objects with proper validation of `TaskType` and `TaskPriority`.
    *   `submit_task()`: Adds a created `Task` object to an internal `asyncio.PriorityQueue`. Tasks are prioritized based on their `Task.priority` value (higher numerical value means higher processing priority) and then by creation time.

6.  **Task Worker Loop (`_task_worker_loop`):**
    *   Multiple asynchronous worker coroutines (number configurable) can run concurrently.
    *   Each worker continuously attempts to fetch the highest priority task from `self.task_queue`.
    *   If a task is retrieved, the worker calls `self.process_task()` to handle its execution.
    *   After `process_task` returns, the worker calls `task.callback(processed_task)` if a callback was provided with the task.

7.  **Model Selection (`select_model_for_task`):**
    *   This method is now a wrapper around the dedicated `ModelSelector` instance (`self.model_selector`).
    *   It prepares `selection_data` (including all loaded `model_capabilities`, the `task.task_type`, `task.context`, `task.requirements`, number of models needed (usually 1), and any `excluded_models` from previous failed attempts in a retry cycle).
    *   Calls `self.model_selector.select_models(selection_data)` and returns the chosen `model_id`.

8.  **Task Processing (`process_task`):**
    *   This is the core execution logic for a single task and includes a retry mechanism.
    *   **Retry Loop:** Iterates up to `task.max_attempts` or a session-specific retry limit.
        1.  **Model Selection:** Calls `select_model_for_task`, passing a set of `session_failed_models` to exclude models that failed within the current `process_task` call's retry sequence.
        2.  **Handler Acquisition:** Retrieves the appropriate `LLMInterface` handler for the selected model's provider from `self.model_handlers`.
        3.  **Execution:** Calls `handler.generate()` (as defined in `LLMInterface`) with the task's prompt, the specific `model_name_for_api` from the `ModelCapability` object, and any generation parameters merged from `task.context.generation_params` and `task.requirements`.
        4.  **Outcome Processing:**
            *   **Success:** Updates task status to `COMPLETED`, stores the LLM's response in `task.result`, calculates latency.
            *   **Failure (Exception from handler or error string from handler):** Logs the error, updates `task.error`.
            *   **Stats Update:** In both cases, calls `capability.update_runtime_stats()` for the chosen model, which updates its internal success rate and average latency, and also makes an async call to `PerformanceMonitor.record_request()` with detailed metrics.
            *   If the attempt failed and more retries are allowed for the session, it adds the failed `model_id` to `session_failed_models` and sleeps with exponential backoff before the next attempt in the loop.
    *   After the loop, the task's final status (`COMPLETED` or `FAILED`) is set. The task is moved from `active_tasks` to `completed_tasks`.
    *   The agent's shared `BeliefSystem` is updated with the task's final status.

### Configuration (`Config` keys relevant to MMA)

-   `orchestration.multimodel_agent.task_max_attempts`: Default maximum retries for a task if not specified per-task.
-   `orchestration.multimodel_agent.models_config_dir`: Path (string, relative to `PROJECT_ROOT`) to the directory containing YAML files that define model capabilities (e.g., `data/mindx_models_config`).
-   `orchestration.multimodel_agent.stats_smoothing_factor` (alpha): For EMA calculation of model runtime stats.
-   `orchestration.multimodel_agent.min_availability_threshold`: Models with availability below this are not selected.
-   `orchestration.multimodel_agent.selection_weights`: (Passed to `ModelSelector`) Dictionary defining weights for factors like capability match, success rate, latency, cost in model scoring.
-   `orchestration.multimodel_agent.provider_preferences`: (Passed to `ModelSelector`) Dictionary weighting different LLM providers.
-   `orchestration.multimodel_agent.retry_delay_seconds`: Base delay for retries between task attempts.
-   `orchestration.multimodel_agent.num_workers`: Number of asynchronous worker coroutines to spawn for processing the internal task queue.
-   `orchestration.multimodel_agent.max_retries_per_process_call`: Max retries within a single call to `process_task`.
-   `orchestration.multimodel_agent.selection_debug_mode`: Boolean to enable verbose logging from `ModelSelector`.
-   `orchestration.multimodel_agent.model_initial_stats.<model_id>.*`: Optional section to pre-set initial dynamic stats for specific models.

## Technical Details

-   **Asynchronous Design:** Fully `async` for non-blocking task processing and LLM interactions.
-   **YAML for Capabilities:** Model capabilities are defined externally in YAML files, organized by provider (e.g., `ollama.yaml`, `gemini.yaml`). This allows easy addition/modification of model information without code changes. Each model entry in YAML should specify its `model_name_for_api` (the tag used by the provider's SDK), `capabilities` (as a map of `TaskType` string to score), `resource_usage`, `max_context_length`, etc.
-   **Dependency on `ModelRegistry` & `LLMFactory`:** The MMA fetches pre-initialized `LLMInterface` compliant handlers from the `ModelRegistry`. The `ModelRegistry` itself uses `LLMFactory` to create these handlers based on global `Config`.
-   **Dependency on `ModelSelector`:** The complex logic of scoring and selecting models is now delegated to the `ModelSelector` class.
-   **Task Lifecycle Management:** Clearly defined states for tasks (`TaskStatus`) and robust handling of retries and finalization.
-   **Dynamic Performance Adaptation:** By updating `ModelCapability` runtime stats and feeding these to the `ModelSelector`, the system can dynamically adapt its model preferences based on observed performance.

## Usage

The `MultiModelAgent` is a foundational service within MindX, typically instantiated once (as a singleton via its factory) and used by higher-level agents like the `CoordinatorAgent` or specialized task-specific agents.

```python
# Conceptual Usage (e.g., from CoordinatorAgent)

# async def setup_mma(shared_belief_system):
#     # MMA factory handles singleton and calls initialize()
#     mma_instance = await get_multimodel_agent_async(belief_system=shared_belief_system)
#     return mma_instance

# async def coordinator_uses_mma(mma: MultiModelAgent, performance_monitor: PerformanceMonitor):
#     # Example: Create a task
#     my_task_def = await mma.create_task(
#         task_type=TaskType.CODE_GENERATION,
#         prompt="Generate a Python function that calculates the Fibonacci sequence.",
#         priority=TaskPriority.HIGH,
#         requirements={"min_context_length": 2048, "target_provider": "ollama"},
#         context={"initiating_agent_id": "Coordinator_SelfImprovementLogic"}
#     )

#     # Option 1: Submit to MMA's internal queue (if workers are enabled)
#     # await mma.submit_task(my_task_def)
#     # Task will be processed by a worker. Results might be via callback or polling task status.

#     # Option 2: Process task directly (if Coordinator manages execution flow)
#     processed_task = await mma.process_task(my_task_def)
    
#     if processed_task.status == TaskStatus.COMPLETED:
#         logger.info(f"MMA Task {processed_task.task_id} succeeded. Result: {str(processed_task.result)[:100]}...")
#         # PerformanceMonitor.record_request would have been called internally by the LLMHandler
#         # via ModelCapability.update_runtime_stats
#     else:
#         logger.error(f"MMA Task {processed_task.task_id} failed. Error: {processed_task.error}")

#     # On application shutdown:
#     # await mma.shutdown() 
