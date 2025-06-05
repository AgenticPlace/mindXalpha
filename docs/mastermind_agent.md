# Mastermind Agent (`mastermind_agent.py`) - Apex Orchestrator & Evolutionary Director v2

## Introduction

The `MastermindAgent` is designed as the apex strategic entity within the MindX system, an Augmentic Project initiative. Its purpose is to oversee the long-term evolution, set overarching system goals, and direct meta-level self-improvement of the entire MindX ecosystem. It achieves this by:

-   Employing its own internal `BDIAgent` for strategic planning and campaign execution.
-   Interacting with the `CoordinatorAgent` to gather system-wide intelligence and to delegate the execution of tactical improvement campaigns (which are ultimately handled by the `SelfImprovementAgent` CLI).
-   Utilizing an `IDManagerAgent` for provisioning secure cryptographic identities, envisioning a future where MindX can autonomously design and instantiate new agents or tools.
-   Operating an autonomous loop to periodically reassess system state and initiate new strategic directives.

## Explanation

### Core Responsibilities & Workflow

1.  **Singleton with Named Instances (`get_instance`):**
    *   An asynchronous factory method `get_instance(agent_id: str, ...)` allows for creating or retrieving uniquely named `MastermindAgent` instances. This supports scenarios with multiple Masterminds or for testing. Instances are cached. `reset_all_instances_for_testing()` is provided.

2.  **Initialization (`__init__`):**
    *   Requires an `agent_id`, a reference to the shared `BeliefSystem`, and a crucial reference to the main `CoordinatorAgent`.
    *   Establishes a dedicated data directory (e.g., `PROJECT_ROOT/data/mastermind_work/<sanitized_agent_id>/`) for its persistent state, including `mastermind_campaigns_history.json` and `mastermind_objectives.json`.
    *   Initializes its own `LLMHandler` for high-level strategic reasoning, configured via `mastermind_agent.<agent_id>.llm.*` settings in `Config`.
    *   **Internal `BDIAgent`:** Instantiates a `BDIAgent` configured for its strategic domain (e.g., `mastermind_strategy_<mastermind_agent_id>`). This BDI agent is the engine for Mastermind's campaign planning and execution loop.
    *   **Registers Custom BDI Actions:** Defines and registers specialized action handlers (`_bdi_action_*` methods) for its internal BDI agent. These actions are the strategic primitives Mastermind can execute.
    *   **`IDManagerAgent`:** Instantiates or gets an `IDManagerAgent` instance. This is used when a strategic plan involves the conceptual creation of a new agent or tool that requires a secure, verifiable identity.
    *   **Autonomous Loop:** If enabled in config (`mastermind_agent.<agent_id>.autonomous_loop.enabled`), it starts an asynchronous background task (`_mastermind_autonomous_worker`) to periodically initiate new strategic campaigns.

3.  **Main Orchestration Method (`manage_mindx_evolution`):**
    *   This is the primary public method for tasking the `MastermindAgent`. It takes a `top_level_directive` (e.g., "Improve overall system fault tolerance by 15%") and `max_mastermind_bdi_cycles`.
    *   It assigns a unique `run_id` for this campaign.
    *   It sets this directive as the primary, high-priority goal for its internal `BDIAgent`.
    *   It then executes `self.bdi_agent.run()`. The BDI agent's plan, composed of the custom strategic actions, will drive the campaign.
    *   The final outcome of the BDI agent's run (summarizing the campaign's success, failure, or status) is logged to `strategic_campaigns_history` and returned. High-level objectives in `mastermind_objectives.json` are also updated.

4.  **Mastermind's Internal BDI Action Handlers (`_bdi_action_*` methods):**
    These are the high-level strategic actions executed by Mastermind's BDI agent:
    *   **`_bdi_action_observe_mindx_state(params)`**:
        -   Queries the `CoordinatorAgent` (by initiating a `SYSTEM_ANALYSIS` interaction with it, focused by `params.analysis_focus`) to get a summary of current MindX system health, performance metrics, and the state of the Coordinator's improvement backlog.
        -   Stores this aggregated state summary in its BDI's beliefs (e.g., `mindx_current_state_summary`).
    *   **`_bdi_action_formulate_campaign_goal(params)`**:
        -   Takes a high-level directive (from BDI's current goal or `params`) and the observed MindX state summary.
        -   Uses `self.llm_handler` (Mastermind's own LLM) to formulate a more specific, actionable strategic campaign goal for the MindX system.
        -   Stores this formulated goal in BDI beliefs (e.g., `current_formulated_campaign_goal_for_bdi`).
    *   **`_bdi_action_launch_improvement_campaign(params)`**:
        -   Takes a formulated campaign goal (from BDI beliefs or `params`).
        -   Initiates a high-level task with the `CoordinatorAgent`. This is typically done by creating a `SYSTEM_ANALYSIS` interaction for the Coordinator, with the Mastermind's campaign goal serving as a strong `analysis_context`. The Coordinator's own autonomous processes (or a `StrategicEvolutionAgent` if integrated there) would then pick up and detail this campaign.
        -   This action marks the campaign as "delegated\_to\_coordinator" in Mastermind's BDI beliefs and records a Mastermind Campaign ID.
    *   **`_bdi_action_request_new_entity_identity(params)`**:
        -   Takes an `entity_tag` (e.g., "new_parser_tool_v1") and `entity_description`.
        -   Calls `self.id_manager_agent.create_new_wallet()` to generate a new cryptographic identity.
        -   Stores the resulting `public_address` in BDI beliefs, associated with the `entity_tag`.
    *   **`_bdi_action_initiate_new_component_development(params)`**:
        -   Highly conceptual and represents an advanced capability.
        -   Takes a description of a new agent/component and optionally a pre-generated identity (public address).
        -   Formulates this as a very high-level strategic campaign goal (e.g., "Develop and integrate new agent: <description>...") and launches this campaign using `_bdi_action_launch_improvement_campaign`. This implies that the underlying MindX machinery (Coordinator, SEA, SIA) is capable of generating entirely new code files, testing, and integrating them.
    *   **`_bdi_action_review_campaign_outcomes(params)`**:
        -   Takes a `campaign_id` (Mastermind's own campaign ID).
        -   Fetches related outcome data from its `strategic_campaigns_history`.
        -   Uses `self.llm_handler` to assess whether the campaign broadly met its original strategic objective based on the BDI outcome messages and status.
        -   Updates BDI beliefs with this assessment, which can inform future strategic decisions.

5.  **Autonomous Loop (`_mastermind_autonomous_worker`):**
    *   If enabled in config, this `asyncio.Task` runs periodically (e.g., every few hours).
    *   In each cycle, it typically calls `self.manage_mindx_evolution()` with a default, broad, proactive directive (e.g., "Proactively monitor and strategically enhance overall MindX system health, capabilities, and efficiency...").

### Interaction Hierarchy

-   **User/External System:** Can give top-level directives to `MastermindAgent`.
-   **`MastermindAgent`:**
    -   Uses its internal `BDIAgent` for strategic planning.
    -   Uses its internal `LLMHandler` for formulating strategic goals and reviewing campaign outcomes.
    -   Uses `IDManagerAgent` for identity provisioning.
    -   Delegates broad campaign initializations or requests for system state to the `CoordinatorAgent`.
-   **`CoordinatorAgent`:**
    -   Receives tasks/directives from `MastermindAgent`.
    -   Performs detailed `SYSTEM_ANALYSIS` (using its own LLM, monitors, codebase scan).
    -   Manages an `improvement_backlog`.
    -   Its autonomous loop (or a `StrategicEvolutionAgent` it manages) picks items from this backlog.
    -   For tactical code changes, it invokes the `SelfImprovementAgent` (SIA) CLI.
-   **`SelfImprovementAgent` (SIA):**
    -   Executes specific file modification tasks, with all its safety (iteration dirs, self-tests, fallbacks) and verification mechanisms. Reports detailed JSON results.

## Technical Details

-   **Asynchronous:** Fully `async` architecture.
-   **Configuration:** Driven by `Config`, with specific settings typically under `mastermind_agent.<agent_id>.*`.
-   **Data Persistence:** Stores its campaign history and high-level objectives in JSON files within its dedicated data directory (`PROJECT_ROOT/data/mastermind_work/<agent_id>/`).
-   **Internal BDI:** The use of an internal `BDIAgent` provides a structured way for Mastermind to manage its complex, long-running strategic objectives and adapt its approach.
-   **Decoupling:** Mastermind does not directly modify code or manage low-level LLM tasks. It delegates these, maintaining a high-level strategic focus.

## Usage

The `MastermindAgent` is the apex controller, designed for setting long-term evolutionary paths for MindX.

1.  **Initialization (Typically once at main application startup):**
    ```python
    # In main_app.py or similar
    # ... initialize Config, BeliefSystem ...
    # coordinator = await get_coordinator_agent_mindx_async(...)
    # mastermind = await MastermindAgent.get_instance(
    #     agent_id="mindx_prime_overseer", # Or from config
    #     belief_system_instance=shared_bs,
    #     coordinator_agent_instance=coordinator,
    #     # config_override can be passed if needed
    # )
    ```

2.  **Tasking the Mastermind (e.g., from an admin interface or diagnostic event):**
    ```python
    # await mastermind.manage_mindx_evolution(
    #     top_level_directive="Expand MindX's capabilities to include automated documentation generation for all new modules.",
    #     max_mastermind_bdi_cycles=50 # Allow significant internal BDI cycles for this campaign
    # )
    ```

3.  **Autonomous Operation:**
    If configured with `mastermind_agent.<agent_id>.autonomous_loop.enabled = true` in `.env` or `mindx_config.json`, the Mastermind will periodically initiate campaigns based on its default directive.

The `MastermindAgent` represents a significant step towards a more fully autonomous and strategically evolving AI system. Its effectiveness hinges on the quality of its LLM-driven strategic formulation and the capabilities of the agents it orchestrates.
