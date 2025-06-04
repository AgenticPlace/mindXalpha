# Plan Management System (`plan_management.py`)

## Introduction

The `PlanManager` class within the MindX learning module (Augmentic Project) provides a structured system for creating, tracking, and orchestrating the execution of multi-step plans. Each plan is designed to achieve a specific goal for an agent (e.g., a `BDIAgent` or `StrategicEvolutionAgent`). It supports defining dependencies between actions within a plan and can execute actions sequentially or with limited parallelism.

## Explanation

### Core Components

1.  **`PlanSt` Enum (Plan Status):**
    Defines the lifecycle status of an entire plan:
    -   `PENDING_GENERATION`: Plan content is being generated (e.g., by an LLM).
    -   `READY`: Plan is defined and ready for execution to begin.
    -   `IN_PROGRESS`: Plan execution has started; one or more actions are running or pending.
    -   `COMPLETED_SUCCESS`: All actions in the plan completed successfully.
    -   `FAILED_ACTION`: One or more actions in the plan failed, and (if critical) this caused the plan to fail.
    -   `FAILED_VALIDATION`: Plan was deemed invalid before or during execution (e.g., unmet preconditions not resolvable).
    -   `PAUSED`: Plan execution is temporarily paused (not fully implemented in this version).
    -   `CANCELLED`: Plan was explicitly cancelled.

2.  **`ActionSt` Enum (Action Status):**
    Defines the lifecycle status of a single action within a plan:
    -   `PENDING`: Action is defined but waiting for its dependencies to be met or for its turn in sequential execution.
    -   `READY_TO_EXECUTE`: All dependencies are met, and it can be picked up by an executor (relevant for parallel execution).
    -   `IN_PROGRESS`: Action execution has started.
    -   `COMPLETED_SUCCESS`: Action completed successfully.
    -   `FAILED`: Action execution failed.
    -   `SKIPPED_DEPENDENCY`: Action was skipped because one of its prerequisite actions failed or was skipped.
    -   `CANCELLED`: Action was cancelled.

3.  **`Action` Class:**
    Represents a single, atomic step within a plan.
    -   `id`: Unique identifier for the action instance within the plan.
    -   `type`: A string identifier for the type of action (e.g., "SEARCH_WEB", "ANALYZE_DATA"). This type is mapped to a handler function by the owning agent.
    -   `params`: A dictionary of parameters required by the action handler. Can include placeholders (see "Dynamic Parameter Resolution").
    -   `description`: Optional human-readable description of the action's purpose.
    *   `status`: Current `ActionSt`.
    *   `result`, `error_message`, `started_at`, `completed_at`, `attempt_count`.
    *   `dependency_ids`: A list of `id`s of other actions in the *same plan* that must complete successfully before this action can start.
    *   `is_critical`: Boolean. If a critical action fails, the entire plan is typically marked as failed immediately.

4.  **`Plan` Class:**
    Represents a sequence of actions designed to achieve a specific `goal_id`.
    -   `id`: Unique identifier for the plan instance.
    *   `goal_id`: The ID of the goal this plan aims to achieve.
    *   `description`: Optional human-readable description of the plan's overall strategy.
    *   `actions`: An ordered list of `Action` objects.
    *   `status`: Current `PlanSt`.
    *   `created_at`, `last_updated_at`, `started_at`, `completed_at`, `created_by`.
    *   `current_action_idx`: Tracks progress for sequential execution.
    *   `action_results`: A dictionary mapping `action_id` to its execution result (can be complex data or an error message).

5.  **`PlanManager` Class:**
    -   **Initialization (`__init__`):** Takes an `agent_id` (for logging) and a mandatory `action_executor` callback.
        -   `action_executor`: An `async` function provided by the owning agent (e.g., `BDIAgent`, `StrategicEvolutionAgent`). This function is responsible for actually performing the work defined by an `Action` object (e.g., calling a tool, prompting an LLM). Its signature is `async def my_executor(action: Action) -> Tuple[bool, Any]`, returning success status and result/error.
    -   **Plan Creation (`create_plan`):** Creates a `Plan` object from a `goal_id` and a list of action data dictionaries.
    -   **Plan Execution (`execute_plan`):**
        *   This is the main entry point to run a plan. It sets the plan's status to `IN_PROGRESS`.
        *   It calls either `_execute_plan_sequential` or `_execute_plan_parallel_with_dependencies` based on `self.parallel_execution_enabled`.
        *   Updates the plan's final status upon completion or failure.
    -   **Sequential Execution (`_execute_plan_sequential`):** Iterates through actions one by one. An action is executed only if its dependencies (if any, though less critical in pure sequential) are met (i.e., their results are available and they were successful). If a critical action fails, plan execution halts.
    -   **Parallel Execution with Dependencies (`_execute_plan_parallel_with_dependencies`):**
        *   Manages a pool of concurrently running action tasks (up to `self.max_parallel_actions`).
        *   In each iteration, it identifies all `PENDING` actions whose dependencies (other actions in the *same plan* that must be `COMPLETED_SUCCESS`) are met.
        *   It launches new `asyncio.Task`s for these executable actions, up to the concurrency limit.
        *   Uses `asyncio.wait(..., return_when=asyncio.FIRST_COMPLETED)` to process actions as they finish.
        *   When an action completes, its status is updated. If successful, it might unblock other dependent actions. If a critical action fails, the entire plan status is set to `FAILED_ACTION`, and an attempt is made to cancel any remaining running tasks for that plan.
    -   **Action State Updates (`_update_action_state`):** Internal helper to update an action's status, store its result, and potentially update the overall plan status (e.g., if all actions complete or a critical one fails).
    -   **Dynamic Parameter Resolution (`_resolve_action_params_from_plan_results`):**
        *   Before an action is passed to the `action_executor`, this method inspects its `params`.
        *   If a parameter value is a string starting with `"$action_result.<source_action_id>.<optional_field_path>"` (e.g., `"$action_result.search_act_1.summary"`), it attempts to retrieve the result of the specified `source_action_id` (which must have already completed successfully within the *same plan*) and extract the relevant data.
        *   This allows actions in a plan to dynamically use the outputs of previous actions in the same plan.
    -   **Status & Querying:** Methods like `get_plan`, `get_current_plan`, `update_plan_status`, `get_all_plans`, etc.

## Technical Details

-   **Asynchronous:** Designed for `asyncio` environments. Plan execution and action execution are asynchronous.
-   **Action Execution Delegation:** The `PlanManager` itself does *not* know how to perform specific actions (like "SEARCH_WEB"). It relies on the `action_executor` callback provided by its owning agent. This keeps the `PlanManager` generic.
-   **Dependency Management:** Supports defining dependencies between actions within a single plan using `dependency_ids` in `Action` objects. Parallel execution respects these dependencies.
-   **Configuration:** Parallel execution behavior (enabled, max concurrent) is configurable via `Config` (e.g., `plan_manager.<agent_id>.parallel_execution.*`).
-   **Error Handling:** Failures in actions, especially critical ones, can lead to the entire plan being marked as failed.

## Usage

The `PlanManager` is intended to be a component *within* an intelligent agent (like `BDIAgent` or the refactored `StrategicEvolutionAgent`).

```python
# Conceptual usage within an owning agent (e.g., StrategicEvolutionAgent)

# class StrategicEvolutionAgent:
#     def __init__(self, agent_id, ...):
#         self.agent_id = agent_id
#         self.plan_manager = PlanManager(agent_id=self.agent_id, action_executor=self._dispatch_sea_action)
#         # ...

#     async def _dispatch_sea_action(self, action: Action) -> Tuple[bool, Any]:
#         # This is the 'action_executor' callback for the PlanManager
#         # It maps action.type to this agent's specific methods
#         logger.info(f"SEA '{self.agent_id}': Dispatching action {action.type} (ID: {action.id})")
#         if action.type == "PERFORM_SYSTEM_WIDE_ANALYSIS":
#             # Assuming self.system_analyzer exists and has this async method
#             analysis_results = await self.system_analyzer.analyze_system_for_improvements(
#                 analysis_focus_hint=action.params.get("focus_hint", "general")
#             )
#             return True, analysis_results # Analysis result becomes action result
#         elif action.type == "REQUEST_SIA_MODIFICATION":
#             # This would call self.coordinator_agent.handle_user_input(...)
#             # to trigger a COMPONENT_IMPROVEMENT interaction, effectively calling SIA CLI
#             coord_response = await self.coordinator_agent.handle_user_input(
#                 # ... construct content and metadata for Coordinator ...
#                 interaction_type=InteractionType.COMPONENT_IMPROVEMENT,
#                 metadata={
#                     "target_component": action.params.get("target_component_path"),
#                     "analysis_context": action.params.get("improvement_goal_for_sia"),
#                     # ... other params for SIA ...
#                 }
#             )
#             success = coord_response.get("status") == InteractionStatus.COMPLETED.value and \
#                       coord_response.get("response",{}).get("status") == "SUCCESS" # Check SIA's JSON output status
#             return success, coord_response.get("response") # Return full SIA JSON output
#         # ... other action handlers ...
#         else:
#             return False, f"Unknown action type for SEA: {action.type}"


#     async def manage_campaign(self, campaign_goal_id: str, strategic_actions_data: List[Dict]):
#         my_plan = self.plan_manager.create_plan(
#             goal_id=campaign_goal_id,
#             actions_data=strategic_actions_data,
#             description=f"Strategic plan for campaign goal {campaign_goal_id}"
#         )
#         
#         logger.info(f"SEA '{self.agent_id}': Executing plan {my_plan.id} for goal {campaign_goal_id}")
#         final_plan_state = await self.plan_manager.execute_plan(my_plan.id)
#         
#         if final_plan_state.status == PlanSt.COMPLETED_SUCCESS:
#             logger.info(f"SEA '{self.agent_id}': Plan {my_plan.id} completed successfully!")
#         else:
#             logger.warning(f"SEA '{self.agent_id}': Plan {my_plan.id} ended with status {final_plan_state.status.name}. Reason: {final_plan_state.failure_reason}")
#         
#         return final_plan_state.to_dict()
