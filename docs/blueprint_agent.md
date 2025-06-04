# Blueprint Agent (`blueprint_agent.py`)

## Introduction

The `BlueprintAgent` is a specialized strategic agent within the MindX system (Augmentic Project). Its primary purpose is to analyze the current overall state of the MindX system and, using a Large Language Model (LLM), generate a high-level "blueprint" or roadmap for the *next iteration of MindX's own evolution*. This blueprint outlines key focus areas, specific development goals for MindX components, and potential metrics for success.

It acts as a meta-level planner, helping to guide the direction of MindX's self-improvement efforts. It gathers context from various parts of the MindX system (via the `CoordinatorAgent` and `BeliefSystem`) to inform its LLM-driven analysis.

## Explanation

### Core Responsibilities & Workflow

1.  **Initialization (`__init__`):**
    *   Takes its own `agent_id`, a reference to the shared `BeliefSystem`, and a crucial reference to the live `CoordinatorAgent` instance.
    *   Initializes its own `LLMHandler` based on configuration (e.g., `evolution.blueprint_agent.llm.*`). This LLM is used specifically for generating the strategic blueprint.

2.  **System State Aggregation (`_gather_mindx_system_state_summary`):**
    *   Before prompting the LLM, this method collects a snapshot of the current MindX system. This includes:
        *   **Codebase Structure:** Number of scanned capabilities and example module paths (from `CoordinatorAgent.system_capabilities_cache` or by triggering a scan).
        *   **Improvement Backlog:** Summary statistics from `CoordinatorAgent.improvement_backlog` (total items, pending, pending approval, top priorities).
        *   **Recent Campaign History:** Outcomes of recent improvement campaigns dispatched by the `CoordinatorAgent`.
        *   **Monitoring Data:** Snapshots of current resource usage (from `ResourceMonitor`) and LLM performance summaries (from `PerformanceMonitor`).
        *   **Known Limitations/TODOs:** Conceptually, this could be sourced from a dedicated belief, a project TODO file, or hardcoded examples of current system weaknesses.
        *   **Core Agent Versions:** Conceptual versions of key MindX agents.

3.  **Blueprint Generation (`generate_next_evolution_blueprint`):**
    *   This is the main public method. It takes the current MindX version string and an optional `high_level_directive` (e.g., "Focus on improving SIA safety," "Increase overall system efficiency").
    *   **Prompt Construction:** It constructs a detailed prompt for its LLM (`self.llm_handler`). The prompt includes:
        *   The agent's role as a Chief Architect AI for MindX.
        *   The current MindX version and the directive.
        *   The aggregated `system_state_summary`.
        *   Clear instructions on the desired output format: a JSON object with keys like `blueprint_title`, `target_mindx_version_increment`, `focus_areas` (each with `area_title` and a list of `development_goals`), `key_performance_indicators`, and `potential_risks_challenges`. Each `development_goal` should specify a `target_component_module_path`, `description`, and `justification`.
    *   **LLM Interaction:** Calls the LLM to generate the blueprint. `json_mode=True` is requested.
    *   **Output Parsing & Validation:** Robustly parses the LLM's JSON response, attempting to extract the blueprint even if it's wrapped in markdown or has minor formatting issues. Basic validation checks if essential keys are present.
    *   **Belief Update:** The generated blueprint JSON is stored in the `BeliefSystem` under a key like `mindx.evolution.blueprint.latest`.
    *   Returns the blueprint dictionary.

4.  **Interaction with Other Agents (Conceptual):**
    *   **Input:** The `BlueprintAgent` is likely invoked by the `CoordinatorAgent` (e.g., as part of a very high-level autonomous cycle or by a developer command routed through the Coordinator) or by a top-level MindX scheduler.
    *   **Output:** The generated blueprint is primarily for consumption by developers or a higher-level meta-strategic component of MindX. The `CoordinatorAgent` could, for instance, be tasked to:
        *   Parse the `development_goals` from the blueprint.
        *   Add these goals (with their priorities) to its own `improvement_backlog`.
        *   The regular autonomous improvement loop of the Coordinator would then pick these up and dispatch them to the `SelfImprovementAgent` (SIA) for tactical execution.

## Technical Details

-   **Singleton Pattern:** Implemented as a singleton, accessed via `get_blueprint_agent_async()`.
-   **Asynchronous:** Core method `generate_next_evolution_blueprint` is `async`.
-   **LLM-Reliant:** The quality and strategic value of the generated blueprint depend heavily on the capabilities of the configured LLM and the comprehensive nature of the system state summary provided in the prompt.
-   **Configuration:** Uses `mindx.utils.config.Config` for its own LLM settings (`evolution.blueprint_agent.llm.*`).
-   **Contextual Awareness:** Relies on accessing state information from the `CoordinatorAgent` (backlog, history) and monitors to provide rich context to its LLM.

## Usage

The `BlueprintAgent` is a strategic planning component.

1.  **Initialization (typically done by a main application orchestrator):**
    ```python
    # from mindx.evolution.blueprint_agent import get_blueprint_agent_async
    # from mindx.core.belief_system import BeliefSystem
    # from mindx.orchestration.coordinator_agent import get_coordinator_agent_mindx_async
    # config = Config()
    # shared_bs = BeliefSystem()
    # coordinator = await get_coordinator_agent_mindx_async(belief_system_instance=shared_bs, config_override=config)
    # bp_agent = await get_blueprint_agent_async(belief_system=shared_bs, coordinator_ref=coordinator, config_override=config)
    ```

2.  **Generating a Blueprint:**
    ```python
    # current_version = config.get("system.version", "0.4.0")
    # directive = "Prioritize enhancing the SelfImprovementAgent's evaluation capabilities for the next iteration."
    #
    # evolution_blueprint = await bp_agent.generate_next_evolution_blueprint(
    #     current_mindx_version=current_version,
    #     high_level_directive=directive
    # )
    # 
    # if "error" not in evolution_blueprint:
    #     print(f"Blueprint Title: {evolution_blueprint.get('blueprint_title')}")
    #     # The Coordinator or a developer would then act on this blueprint.
    #     # For example, add development_goals to the Coordinator's backlog.
    #     for area in evolution_blueprint.get("focus_areas", []):
    #         for goal_sugg in area.get("development_goals", []):
    #             # coordinator.add_to_improvement_backlog(goal_sugg, source=f"blueprint_{bp_agent.agent_id}")
    #             pass 
    ```
    The `example_main_agi_self_improvement()` in the `.py` file shows a more complete standalone test.

The `BlueprintAgent` adds a layer of meta-cognition to MindX, allowing the system to reason about its own future development path and guide its self-improvement efforts more strategically.
