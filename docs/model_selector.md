# Model Selector (`model_selector.py`) - Production Candidate

## Introduction

The `ModelSelector` class is a dedicated component within the MindX orchestration layer (Augmentic Project). Its primary responsibility is to choose the most suitable Large Language Model(s) for a given task based on a comprehensive set of criteria. This includes static model capabilities, dynamic runtime performance statistics, cost considerations, task-specific requirements, and configurable preferences. By encapsulating this logic, it provides a flexible and extensible way for other agents, like the `MultiModelAgent` or `CoordinatorAgent`, to make informed model selections.

## Explanation

### Core Functionality

1.  **Initialization (`__init__`):**
    *   Takes an optional `Config` object (defaults to the global singleton).
    *   Loads default selection weights (e.g., for capability match, success rate, latency, cost) and provider preferences from the configuration. These can be overridden by task-specific context.
    *   Loads the minimum availability threshold below which models are not considered.

2.  **Model Selection (`select_models`):**
    *   This is the main public method. It takes `selection_data` dictionary as input, which must include:
        -   `model_capabilities`: A dictionary where keys are `model_id` strings (e.g., `"ollama/llama3:8b-instruct"`) and values are `ModelCapability` objects (defined in `multimodel_agent.py`).
        -   `task_type`: A `TaskType` enum instance indicating the nature of the task.
        -   `num_models`: The desired number of top models to return.
    *   Optional fields in `selection_data`:
        -   `context`: A dictionary for task-specific information that can influence selection (see "Contextual Adjustments" below).
        -   `excluded_models`: A set of `model_id`s to ignore during this selection.
        -   `debug_mode`: A boolean to enable verbose logging of the selection process.
    *   **Workflow:**
        1.  Filters out models that are explicitly excluded or fall below the `min_availability_threshold`.
        2.  For each remaining candidate model, it calls `_calculate_model_score` to get a numerical score.
        3.  Applies any direct score adjustments specified in `context["model_score_adjustments"]`.
        4.  Sorts the scored models in descending order.
        5.  If multiple models have the same top score, a tie-breaking mechanism is used: prefers lower cost (e.g., `cost_per_kilo_total_tokens` from `ModelCapability.resource_usage`), then lower average latency (`ModelCapability.average_latency_ms`).
        6.  Returns a list of the top `num_models_to_select` `model_id`s.

3.  **Score Calculation (`_calculate_model_score`):**
    *   This private method computes a score for a single model against a specific task type and context.
    *   **Factors Considered (with configurable weights):**
        -   `capability_match`: The model's declared proficiency score for the given `TaskType` (from `ModelCapability.get_capability_score()`).
        -   `success_rate`: The model's historical success rate (from `ModelCapability.success_rate`).
        -   `latency_factor`: An inverse normalized measure of the model's average latency (lower latency results in a higher factor).
        -   `cost_factor`: An inverse normalized measure of the model's cost (e.g., cost per 1K total tokens). Lower cost results in a higher factor. "Free" models get a high cost factor.
        -   `provider_preference`: A configurable weight applied based on the model's provider (e.g., preferring "gemini" over "ollama").
        -   `requirements_match`: A factor (initially 1.0) that is penalized if the model fails to meet specific task requirements defined in `context["task_requirements"]` (e.g., `min_context_length`, `supports_streaming`, `supports_function_calling`, or if a `target_model_id` or `target_provider` is specified and doesn't match).
    *   The final score is a weighted sum of these factors and their corresponding values from `self.selection_weights`.
    *   `debug_mode` enables detailed logging of each component's contribution to the score.

### Contextual Adjustments

The `select_models` method allows for dynamic adjustments to the selection logic via the `context` dictionary in `selection_data`:

-   `context["weight_adjustments"]`: A dictionary to temporarily multiply the global selection weights for specific factors (e.g., `{"latency_factor": 1.5}` to make latency more important for this task).
-   `context["model_score_adjustments"]`: A dictionary to directly multiply the final calculated score of specific models (e.g., `{"ollama/llama3:8b": 1.2}` to boost its chances).
-   `context["task_requirements"]`: A dictionary specifying hard requirements for the task, used in `_calculate_model_score` to penalize or exclude models (e.g., `{"min_context_length": 16000, "supports_streaming": true}`).

## Technical Details

-   **Configuration:** Relies heavily on the global `Config` object for default weights, thresholds, and provider preferences (e.g., keys under `orchestration.model_selector.*`).
-   **Dependencies:** Expects `ModelCapability` objects (as defined in `multimodel_agent.py`) to be provided, which contain both static definitions and dynamic runtime statistics.
-   **Scoring Philosophy:** The scoring is designed to be a blend of a model's inherent suitability for a task type, its observed reliability and speed, its cost-effectiveness, and adherence to specific operational requirements.
-   **Extensibility:** The factor-based scoring and contextual adjustments make it relatively easy to add new scoring criteria or refine existing ones.

## Usage

The `ModelSelector` is typically used by components like the `MultiModelAgent` or the `CoordinatorAgent` when they need to choose an LLM for a task.

```python
# Assume 'config' is an initialized Config instance
# Assume 'all_model_capabilities' is a Dict[str, ModelCapability]

from mindx.orchestration.model_selector import ModelSelector
from mindx.orchestration.multimodel_agent import TaskType # Assuming TaskType is here

selector = ModelSelector(config=config)

selection_criteria = {
    "model_capabilities": all_model_capabilities,
    "task_type": TaskType.CODE_GENERATION,
    "num_models": 1,
    "excluded_models": {"ollama/some_old_model"}, # Don't pick this one
    "context": {
        "task_requirements": {
            "min_context_length": 8192,
            "supports_function_calling": True
        },
        "weight_adjustments": { # For this specific task, make capability match very important
            "capability_match": 3.0 
        },
        "model_score_adjustments": { # Slightly prefer a specific model if it's a candidate
            "gemini/gemini-1.5-pro-latest": 1.1 
        }
    },
    "debug_mode": True 
}

selected_model_ids = selector.select_models(selection_criteria)

if selected_model_ids:
    print(f"Selected model for Code Generation: {selected_model_ids[0]}")
else:
    print("No suitable model found for Code Generation with the given criteria.")
