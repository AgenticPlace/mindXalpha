# Multi-Model Agent (MMA) (`multimodel_agent.py`) - Production Candidate (Core Logic)

## Introduction

The `MultiModelAgent` (MMA) is a sophisticated component within the MindX system (Augmentic Project) responsible for managing and executing tasks that require interactions with various Large Language Models (LLMs). It handles model selection based on dynamically loaded capabilities and runtime performance, task queuing (optional), execution with retries, and updates model statistics. It relies on a `ModelRegistry` to provide concrete LLM handlers and `Config` for its operational settings.

## Explanation

### Core Components & Workflow

1.  **`TaskType`, `TaskPriority`, `TaskStatus` Enums:** Define the nature, importance, and lifecycle state of tasks managed by the MMA.
2.  **`Task` Data Class:** Encapsulates all information related_to a single unit of work for an LLM, including:
    *   `task_id`, `task_type`, `prompt`, `priority`.
    *   `context`: Arbitrary data relevant to the task.
    *   `requirements`: Specific needs for model selection (e.g., `min_context_length`, `target_model_id`, `supports_streaming`).
    *   `status`, `assigned_model`, `result`, `error`, timestamps, `attempts`, `max_attempts`.
    *   `history`: A log of events during the task's processing.
3.  **`ModelCapability` Data Class:** Stores both static and dynamic information about each available LLM:
    *   **Static:** `model_id` (system-wide unique, e.g., "ollama/llama3:8b"), `provider`, `model_name_for_api` (the name used in API calls), `capabilities` (a dictionary mapping `TaskType` to a proficiency score, 0.0-1.0), `resource_usage` (e.g., cost per token), `max_context_length`, `supports_streaming`, `supports_function_calling`.
    *   **Dynamic:** `availability`, `success_rate`, `average_latency_ms`. These are updated after each task execution using an exponential moving average (`update_runtime_stats`) to reflect recent performance.
    *   `from_dict` class method parses capability data loaded from YAML files.
4.  **Initialization (`__init__` and `async initialize`):**
    *   Synchronous `__init__`: Sets up basic attributes, configuration, and references to `BeliefSystem` and `ModelRegistry`. Defines the path to model capability configuration files (e.g., `data/mindx_models_config/`).
    *   Asynchronous `initialize()`: This **must be called** after creating an MMA instance. It performs:
        *   `_load_model_capabilities_from_files()`: Scans the `models_config_dir` for provider-specific YAML files (e.g., `ollama.yaml`, `gemini.yaml`). Each file defines one or more models for that provider, along with their static capabilities. `ModelCapability` objects are created and stored in `self.model_capabilities`.
        *   `_initialize_model_handlers_from_registry()`: Populates `self.model_handlers` by retrieving initialized `LLMInterface` instances from the `ModelRegistry` for each provider found in the loaded capabilities.
        *   Starts asynchronous task worker loops (`_task_worker_loop`) if the MMA's internal task queue is to be used autonomously.

5.  **Task Creation & Queuing:**
    *   `create_task()`: Factory method to instantiate `Task` objects.
    *   `add_task_to_queue()`: Adds a task to an internal `asyncio.PriorityQueue`, ordered by task priority (higher numerical value = higher logical priority).
    *   `_task_worker_loop()`: (If MMA manages its own queue) Asynchronous workers continuously fetch tasks from the priority queue and pass them to `process_task`.

6.  **Model Selection (`select_model_for_task`):**
    *   This is a sophisticated selection algorithm that scores available models based on:
        *   **Capability Match:** Score for the required `TaskType`.
        *   **Runtime Stats:** Current `success_rate` and `average_latency_ms`.
        *   **Cost:** `cost_per_token` or similar from `resource_usage`.
        *   **Provider Preference:** Configurable weights for different LLM providers (e.g., prefer Gemini over Ollama if both are suitable).
        *   **Task Requirements:** Penalizes or excludes models that don't meet specific requirements in `task.requirements` (e.g., context length, streaming support, specific target model).
        *   **Availability:** Filters out models below a minimum availability threshold.
    *   Returns the `model_id` of the best-scoring model, or `None` if no suitable model is found.

7.  **Task Processing (`process_task`):**
    *   This is the core execution logic for a single task, designed to be robust with retries.
    *   **Retry Loop:** Iterates up to `task.max_attempts`.
        1.  **Model Selection:** Calls `select_model_for_task`, excluding any models that have already failed *within this current `process_task` call's retry session*.
        2.  **Handler Acquisition:** Retrieves the appropriate `LLMInterface` handler for the selected model's provider from `self.model_handlers`.
        3.  **Execution:** Calls `handler.generate()` (or a similar method from `LLMInterface`) with the task's prompt, the specific `model_name_for_api` from the capability object, and any generation parameters from `task.context` or `task.requirements`.
        4.  **Outcome:**
            *   **Success:** Updates task status to `COMPLETED`, stores the result, calculates latency, and calls `capability.update_runtime_stats(success=True, ...)`. The retry loop is broken.
            *   **Failure (Exception from handler):** Logs the error, updates task status, calls `capability.update_runtime_stats(success=False, ...)`, adds the failed model to a session-specific exclusion list, and if attempts remain, sleeps with exponential backoff before retrying.
    *   After the loop (either success or max retries exhausted), the task's final status is set, and it's moved from `active_tasks` to `completed_tasks`.
    *   Optionally updates the `BeliefSystem` with the task's final status.

### Configuration (`Config` keys used by MMA)

-   `orchestration.multimodel_agent.task_max_attempts`: Default max retries for a task.
-   `orchestration.multimodel_agent.models_config_dir`: Path (relative to `PROJECT_ROOT`) to YAML files defining model capabilities.
-   `orchestration.multimodel_agent.stats_smoothing_factor` (alpha): For EMA calculation of runtime stats.
-   `orchestration.multimodel_agent.min_availability_threshold`: Models below this availability are not selected.
-   `orchestration.multimodel_agent.selection_weights`: Dictionary for capability, success, latency, cost, requirements match scores.
-   `orchestration.multimodel_agent.provider_preferences`: Dictionary weighting different LLM providers.
-   `orchestration.multimodel_agent.retry_delay_seconds`: Base delay for retries (used with exponential backoff).
-   `orchestration.multimodel_agent.num_workers`: Number of concurrent task processing workers if MMA uses its internal queue.

## Technical Details

-   **Asynchronous Design:** Core operations like task processing and worker loops are `async`.
-   **YAML for Capabilities:** Model capabilities are defined externally in YAML files (e.g., `data/mindx_models_config/ollama.yaml`, `data/mindx_models_config/gemini.yaml`), making it easy to add or update model information without code changes.
    -   Each YAML file typically represents a provider.
    -   Inside, models are listed with their `capabilities` (map of `TaskType` string to score), `resource_usage`, `max_context_length`, `model_tag` (for API), etc.
-   **Dependency on `ModelRegistry`:** The MMA does not create `LLMHandler` instances itself. It retrieves them from the `ModelRegistry`, which is responsible for initializing and caching these handlers based on global `Config` (API keys, base URLs).
-   **Dynamic Stats Update:** `ModelCapability.update_runtime_stats` provides a mechanism for the system to learn and adapt its model preferences based on observed performance. (Note: Persisting these updated stats across MMA restarts would require additional logic to save/load `ModelCapability` objects themselves).

## Usage

The `MultiModelAgent` is typically instantiated and initialized by a higher-level orchestrator like the `CoordinatorAgent`.

```python
# In CoordinatorAgent or similar:
from mindx.orchestration.multimodel_agent import get_multimodel_agent_async, TaskType, Task, TaskPriority

# Get/Create MMA instance (factory handles singleton)
# mma_belief_system = BeliefSystem() # Assuming a shared or MMA-specific belief system
# multi_model_agent = await get_multimodel_agent_async(belief_system=mma_belief_system)
# The factory now calls initialize internally:
# await multi_model_agent.initialize() # This loads capabilities and handlers

# Example: Create and process a task
# coordinator_identified_task = await multi_model_agent.create_task(
#     task_type=TaskType.CODE_GENERATION,
#     prompt="Write a Python function to calculate factorial.",
#     priority=TaskPriority.HIGH,
#     requirements={"target_provider": "ollama"} # Example requirement
# )

# If MMA manages its own queue:
# await multi_model_agent.add_task_to_queue(coordinator_identified_task)
# (Worker loops would then pick it up)

# If Coordinator processes tasks directly through MMA:
# processed_task = await multi_model_agent.process_task(coordinator_identified_task)
# if processed_task.status == TaskStatus.COMPLETED:
#     print(f"Task Result: {processed_task.result}")
# else:
#     print(f"Task Failed: {processed_task.error}")

# To shutdown workers gracefully (if MMA started them):
# await multi_model_agent.shutdown()
