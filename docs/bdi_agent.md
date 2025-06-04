# BDI Agent (`bdi_agent.py`) - Production Candidate v3 (Enhanced Intelligence & Integration)

## Introduction

The `BDIAgent` (Belief-Desire-Intention) within the MindX system (Augmentic Project) is designed for goal-directed autonomous behavior. This version (v3) significantly enhances its "intelligence" by deeply integrating Large Language Models (LLMs) for core BDI cognitive functions: **plan generation**, **subgoal decomposition**, rudimentary **failure analysis**, and conceptual **plan monitoring**. It operates using a shared `BeliefSystem` for robust knowledge management and interacts with configured tools. This version also introduces more dynamic parameter resolution for actions based on beliefs or prior action results.

## Explanation

### Core BDI Components & LLM Integration

1.  **Beliefs (Shared `BeliefSystem`):**
    *   The agent is initialized with and **critically relies on** an instance of the shared `mindx.core.belief_system.BeliefSystem`.
    *   All persistent beliefs are stored here, namespaced: `bdi.<domain_name>.beliefs.<category>.<key>`.
    *   `update_belief()`, `get_belief()`, and `get_beliefs_by_prefix()` methods interact asynchronously with the shared `BeliefSystem`.
    *   An internal `_internal_state` dictionary holds transient, private agent status (e.g., `RUNNING`, `last_action_details`, `current_run_id`). An `agent_status` belief is also updated in the shared system.

2.  **Desires (Goal Management):**
    *   `primary_goal_description` and `primary_goal_id`: Store the main objective for a given run.
    *   `priority_queue`: A list of goal entries: `{"id": str, "goal": str, "priority": int, "status": str, "added_at": float, "parent_goal_id": Optional[str], "plan_history": List[str]}`.
        -   `plan_history`: Stores IDs of plans attempted for this goal, aiding in avoiding repeated failed strategies.
    *   `set_goal()`: Adds/updates goals. Goals are sorted by priority (desc) and then by age (oldest first).

3.  **Intentions (Plan Management):**
    *   `current_plan_id`: Unique ID for the active plan.
    *   `current_plan_actions`: List of action dictionaries. Each action has an `id`, `type`, `params`, and (conceptually for LLM guidance) `preconditions` and `effects`.
    *   `current_goal_id_for_plan`: Links the plan to the goal it aims to achieve.
    *   `plan_status`: Tracks plan state (e.g., `PENDING_GENERATION`, `READY`, `FAILED_PLANNING`, `COMPLETED`).

### Enhanced BDI Cycle with LLM-driven Intelligence (`run` method)

1.  **Initialization (`__init__`):**
    *   Configures its internal `LLMHandler` based on BDI-specific or default LLM settings from `Config`. This LLM is for the BDI's "internal thought processes."
    *   Initializes tools (e.g., `WebSearchTool`, `NoteTakingTool`, `SummarizationTool`) which may have their own LLM handlers.

2.  **Perceive (`perceive`):**
    *   Updates beliefs based on `external_input`.
    *   **LLM-Assisted Plan Monitoring (`_is_plan_still_valid_async`):** Conceptually, can use an LLM to assess if the current plan's preconditions are still met given new beliefs. If not, marks the plan `INVALID` to trigger replanning. (LLM call is stubbed but framework exists).

3.  **Deliberate (`deliberate`):**
    *   Selects the highest-priority "pending" `current_goal_entry`.
    *   **LLM-driven Subgoal Decomposition:** If a goal is complex and needs a plan (or prior plans failed), it calls `_generate_subgoals_llm()`.
        *   `_generate_subgoals_llm()`: Prompts `self.llm_handler` to break the goal into 2-4 subgoals (JSON list of strings).
        *   New subgoals are added to the `priority_queue`. The parent goal is marked e.g. `decomposed_pending_subgoals`.
        *   Deliberation then might pick a new (sub)goal.

4.  **Plan (`plan`):**
    *   If no valid plan exists for the `current_goal_entry`, it uses `self.llm_handler` to generate one.
    *   **Enhanced LLM-driven Planning:**
        *   The prompt includes the goal, available action types, a summary of relevant beliefs (via `_get_belief_summary_for_prompt`), and crucially, information about previously *failed plan IDs* for this goal. This encourages the LLM to try different approaches.
        *   It requests a JSON list of action dictionaries. The LLM is expected to determine appropriate `params` for actions.
    *   Handles robust parsing of the LLM's JSON plan. Updates beliefs on planning failure.

5.  **Execute (`execute_current_intention`):**
    *   **Dynamic Parameter Resolution (`_resolve_action_params`):** Before executing an action, its parameters are processed. Placeholders like `"$belief.knowledge.some_key"` or `"$last_action_result.field"` are resolved by fetching values from the `BeliefSystem` or the result of the previously executed action.
    *   Dispatches the action (with resolved params) to the appropriate handler in `_action_handlers`.

6.  **Action Completion (`action_completed`):**
    *   Updates `_internal_state` and logs action outcome to the shared `BeliefSystem`.
    *   If failed, the plan and goal are marked `FAILED`. The `run` loop may then call `_analyze_failure_reason`.

7.  **LLM-driven Failure Analysis (`_analyze_failure_reason`):**
    *   If an action/plan fails, this method is called (if enabled by config).
    *   Constructs a prompt for `self.llm_handler` with details of the failed action, goal, error, and relevant beliefs.
    *   The LLM analyzes the likely root cause.
    *   This analysis is stored as a belief and can inform subsequent replanning attempts for the same goal (by being included in the planning prompt).

### Action Handlers and Dynamic Parameters

-   Action handlers (`_execute_search_web`, `_execute_llm_cognitive_action`, etc.) remain.
-   **`_resolve_action_params(params)` and `_resolve_param_value(value)`:** These new helper methods are called by `execute_current_intention` before an action handler is invoked. They look for string values in action parameters starting with:
    -   `"$belief.<category>.<key>"`: Fetches the value from `BeliefSystem`.
    -   `"$last_action_result"` or `"$last_action_result.<field>"`: Uses the result of the immediately preceding successful action.
    -   This allows plans generated by the LLM to be more dynamic and data-driven, using information gathered or produced earlier in the plan.
-   **`UPDATE_BELIEF` Action:** A new action type allowing the LLM plan to directly instruct the BDI agent to update/add a belief in the shared `BeliefSystem`.

## Technical Details

-   **Asynchronous Operations:** Core BDI cycle and action handlers are `async`.
-   **LLM for Core Cognition:** LLMs are now central to planning, subgoal decomposition (conceptual), plan monitoring (conceptual), and failure analysis.
-   **Configuration:** Key behaviors (enable subgoal decomposition, LLM choices, planning tokens/temp) are configurable via `Config`.
-   **Prompt Engineering:** The effectiveness of the "intelligence" heavily relies on the quality of prompts sent to the internal LLM for planning, decomposition, and analysis. These prompts now include more context (beliefs, failure history).

## How to Interact with this BDI Agent (from Coordinator or other systems)

1.  **Instantiation:**
    ```python
    from mindx.core.bdi_agent import BDIAgent
    from mindx.core.belief_system import BeliefSystem # Get shared instance
    
    shared_bs = BeliefSystem() # Or get_belief_system()
    my_bdi_agent = BDIAgent(
        domain="specific_task_domain",
        belief_system_instance=shared_bs,
        # initial_goal can be set here or via set_goal() later
    )
    ```

2.  **Setting Goals:**
    A goal is a textual description of what the agent should achieve.
    ```python
    my_bdi_agent.set_goal("Research latest advancements in quantum computing and summarize findings.", 
                          priority=1, is_primary=True, goal_id="research_qc_001")
    # Add more goals; they will be prioritized
    my_bdi_agent.set_goal("Identify potential ethical concerns of quantum computing.", 
                          priority=2, parent_goal_id="research_qc_001") 
    ```

3.  **Providing External Input/Perceptions:**
    Before or during a `run`, external information can be fed into the shared `BeliefSystem` which the BDI agent will perceive. For direct input to a specific `run` call:
    ```python
    initial_context_for_run = {
        "known_research_sources": ["arxiv.org", "nature.com"],
        "deadline_hint": "Report due by end of week." 
    }
    # The BDI's perceive method will add these to its namespaced beliefs.
    ```

4.  **Executing the BDI Cycle:**
    The `run()` method starts the agent's perceive-deliberate-plan-execute loop.
    ```python
    # To run the agent for a certain number of cycles or until primary goal completion/failure:
    max_bdi_cycles = 20 
    final_status_message = await my_bdi_agent.run(
        max_cycles=max_bdi_cycles,
        external_input=initial_context_for_run # Optional initial input for the first cycle
    )
    print(f"BDI Run Completed. Outcome: {final_status_message}")
    ```
    The `run()` method returns a string summarizing the final outcome (e.g., goal achieved, failed, timed out).

5.  **Observing BDI Agent State (via Shared BeliefSystem):**
    Other agents can monitor the BDI agent's progress by querying the shared `BeliefSystem`. Key beliefs updated by this BDI agent include:
    *   `bdi.<domain>.beliefs.agent_status`: Overall status ("RUNNING", "COMPLETED_IDLE", etc.).
    *   `bdi.<domain>.beliefs.current_goal_processing`: Info about the goal it's currently focused on.
    *   `bdi.<domain>.beliefs.action_history.<action_id>`: Details about each executed action.
    *   `bdi.<domain>.beliefs.plan_details.<plan_id>.status`: Status of plans.
    *   `bdi.<domain>.beliefs.failure_analysis.<action_id>`: LLM's analysis of a failure.
    *   `bdi.<domain>.beliefs.run_history.<run_id>.outcome`: Summary of a completed `run()`.
    *   Any beliefs created by its actions (e.g., `bdi.<domain>.beliefs.knowledge.search_results...`, `bdi.<domain>.beliefs.action_results...`).

6.  **Inspecting Internal State (for debugging/advanced use):**
    While beliefs are the primary way to observe, for deeper debugging, one might (if holding the instance) inspect:
    *   `my_bdi_agent.desires["priority_queue"]`
    *   `my_bdi_agent.intentions["current_plan_actions"]`
    *   `my_bdi_agent._internal_state`

7.  **Shutting Down:**
    ```python
    await my_bdi_agent.shutdown() # Placeholder for any cleanup
    ```

This more "intelligent" `BDIAgent`, by leveraging LLMs for its core cognitive deliberation and planning steps, and by using a shared `BeliefSystem`, becomes a more capable and integrated component for autonomous task execution within the MindX architecture. Its effectiveness, however, is now strongly tied to the quality of its LLM interactions and the design of its prompts.
