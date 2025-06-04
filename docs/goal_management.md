# Goal Management System (`goal_management.py`)

## Introduction

This module provides a robust `GoalManager` class and associated structures for managing an agent's goals within the MindX system (Augmentic Project). It allows for adding goals with priorities, tracking their status, handling dependencies between goals, and retrieving the highest priority actionable goal using a min-heap based priority queue. It also introduces abstract and concrete `GoalPrioritizationStrategy` classes for more complex sorting of goals if needed outside the priority queue's direct use.

## Explanation

### Core Components

1.  **`GoalSt` Enum (Goal Status):**
    Defines the possible states a goal can be in:
    -   `PENDING`: Newly added or dependencies now met, ready for consideration.
    -   `ACTIVE`: Currently being pursued (e.g., a plan is being executed for it). (Note: This status is more conceptual in current `GoalManager` as `get_highest_priority_pending_goal` focuses on `PENDING`. An agent using it would set its own "active goal" state).
    -   `COMPLETED_SUCCESS`: Successfully achieved.
    -   `COMPLETED_NO_ACTION`: Goal was valid but determined to require no action (e.g., already met).
    -   `FAILED_PLANNING`: The agent failed to generate a viable plan for this goal.
    -   `FAILED_EXECUTION`: A plan was generated, but its execution failed.
    -   `PAUSED_DEPENDENCY`: The goal cannot be pursued yet because one or more of its prerequisite goals are not completed.
    -   `CANCELLED`: The goal was explicitly cancelled.

2.  **`Goal` Class:**
    Represents a single goal with the following attributes:
    -   `id`: Unique identifier (UUID).
    -   `description`: Textual description of the goal.
    -   `priority`: Integer (e.g., 1-10, higher number means higher priority). Clamped to a valid range on creation.
    -   `status`: Current `GoalSt` of the goal.
    -   `created_at`, `last_updated_at`: Timestamps.
    -   `metadata`: Optional dictionary for additional information.
    -   `parent_goal_id`: ID of a higher-level goal if this is a subgoal.
    -   `subgoal_ids`: List of IDs of subgoals generated from this goal.
    -   `dependency_ids`: List of IDs of goals that must be completed *before* this goal can start.
    -   `dependent_ids`: List of IDs of goals that depend on *this* goal's completion.
    -   `current_plan_id`: ID of the plan currently associated with achieving this goal (if any).
    -   `attempt_count`: How many times has an attempt been made to achieve this goal.
    -   `failure_reason`: Textual reason if the goal failed.
    -   `source`: String indicating the origin of the goal (e.g., "llm_analysis", "user_directive").
    -   Implements `__lt__` for direct use in `heapq` (prioritizes by `-priority`, then `created_at`).

3.  **`GoalManager` Class:**
    -   **Initialization (`__init__`):** Takes an `agent_id` for logging and namespacing. Initializes an empty dictionary `self.goals` to store `Goal` objects by their ID, and an empty list `self.priority_queue` which will be managed as a min-heap by the `heapq` module.
    -   **`add_goal(...)`:**
        *   Creates a new `Goal` object.
        *   Performs simple de-duplication based on goal description (updates priority of existing non-terminal goal if new one is higher).
        *   Adds the goal to `self.goals`.
        *   If the goal has dependencies, it registers them and sets the goal's status to `PAUSED_DEPENDENCY` if any dependency is not yet `COMPLETED_SUCCESS`.
        *   If the goal is active (no unmet dependencies and `PENDING`), it's pushed onto the `self.priority_queue` (using negated priority for max-heap behavior with `heapq`).
    -   **`get_goal(goal_id)`:** Retrieves a `Goal` object by its ID.
    -   **`get_highest_priority_pending_goal()`:**
        *   This is the primary method for an agent to fetch its next task.
        *   It intelligently inspects the `priority_queue` (min-heap).
        *   It pops items, checks if the corresponding `Goal` object is still `PENDING` and if all its `dependency_ids` are `COMPLETED_SUCCESS`.
        *   Returns the first such actionable `Goal` found, or `None`. Non-actionable items are temporarily held and then pushed back onto the heap to maintain queue integrity.
    -   **`update_goal_status(goal_id, status, failure_reason)`:**
        *   Updates the status of a specified goal.
        *   If a goal's status changes to `COMPLETED_SUCCESS`, it triggers `_activate_dependent_goals` to potentially unblock goals that depended on it.
        *   If a goal reaches a terminal status (completed, failed, cancelled), `_rebuild_priority_queue_from_active_goals` is called to ensure the queue only contains actionable items.
    -   **`_activate_dependent_goals(completed_goal_id)`:** Iterates through goals that depended on the `completed_goal_id`. If all other dependencies for a dependent goal are also met, its status is set to `PENDING` and it's effectively re-added/re-prioritized in the queue (via `update_goal_priority` which rebuilds).
    -   **`update_goal_priority(goal_id, new_priority)`:** Changes a goal's priority and rebuilds the priority queue.
    -   **`_rebuild_priority_queue_from_active_goals()`:** Clears and reconstructs the `priority_queue` using only goals that are currently in a non-terminal status (e.g., `PENDING`, `PAUSED_DEPENDENCY`).
    -   **`add_dependency(goal_id, depends_on_goal_id)`:** Establishes a dependency. Updates the dependent goal's status to `PAUSED_DEPENDENCY` if the prerequisite is not yet complete. Includes a basic circular dependency check.
    -   **`_would_create_circular_dependency(...)`:** A recursive helper to detect potential cycles before adding a dependency.
    -   **Query Methods:** `get_all_goals(status_filter)`, `get_goal_status_summary()`, `get_dependencies()`, `get_dependents()`.

4.  **`GoalPrioritizationStrategy` (Abstract Base Class & Concrete Strategies):**
    *   Defines an interface for different ways to sort a list of goals. This is separate from the `GoalManager`'s internal priority queue, which always uses priority and timestamp. These strategies could be used by an agent to *select* from a list of available goals *before* adding them to the `GoalManager`, or for presenting goals to a user.
    *   **`SimplePriorityThenTimeStrategy`:** Sorts by `-priority`, then `created_at`.
    *   **`UrgencyStrategy`:** Calculates an "urgency score" (`priority + (waiting_time * factor)`) and sorts by that.
    *   (The `DependencyAwareStrategy` and `CompositeStrategy` from the previous prompt's `prioritization_strategies.py` are good candidates to include here as well, but were omitted in this generation for brevity as `GoalManager` now handles dependencies directly for its queue.)

## Technical Details

-   **Priority Queue:** Uses Python's `heapq` module for an efficient min-heap. Priorities are negated before pushing so that numerically higher priorities (e.g., 10) result in smaller heap values, effectively creating a max-priority queue. Creation timestamp is used as a tie-breaker.
-   **Goal Representation:** `Goal` objects store comprehensive information about each goal.
-   **Dependencies:** A goal can depend on multiple other goals. A goal only becomes `PENDING` (and thus eligible for the priority queue and selection by `get_highest_priority_pending_goal`) once all its dependencies are `COMPLETED_SUCCESS`.
-   **Status Management:** Careful updates to goal statuses are crucial for the `get_highest_priority_pending_goal` logic and for activating dependent goals.
-   **Logging:** Informative logging helps trace goal lifecycle events.

## Usage (Conceptual within an Agent like `StrategicEvolutionAgent` or `BDIAgent`)

The `GoalManager` would be a component within a more complex agent.

```python
# Conceptual usage inside an agent (e.g., StrategicEvolutionAgent)
# class StrategicEvolutionAgent:
#     def __init__(self, agent_id: str, ...):
#         self.goal_manager = GoalManager(agent_id=agent_id)
#         self.current_strategic_goal: Optional[Goal] = None
#         # ...

#     async def set_new_strategic_objective(self, description: str, priority: int):
#         new_goal = self.goal_manager.add_goal(description, priority, source="coordinator_directive")
#         logger.info(f"SEA: New strategic objective added: {new_goal.id} - {new_goal.description}")

#     async def strategic_cycle(self):
#         if self.current_strategic_goal and self.current_strategic_goal.status not in [GoalSt.COMPLETED_SUCCESS, GoalSt.FAILED_PLANNING, GoalSt.FAILED_EXECUTION]:
#             logger.info(f"SEA: Continuing with current strategic goal: {self.current_strategic_goal.description}")
#         else:
#             self.current_strategic_goal = self.goal_manager.get_highest_priority_pending_goal()

#         if not self.current_strategic_goal:
#             logger.info("SEA: No current strategic goals to pursue.")
#             return

#         logger.info(f"SEA: Current strategic goal: {self.current_strategic_goal.description} (Status: {self.current_strategic_goal.status.name})")
#         self.goal_manager.update_goal_status(self.current_strategic_goal.id, GoalSt.ACTIVE)
        
#         # ... logic to plan and execute for self.current_strategic_goal ...
#         # This would involve creating a plan using PlanManager, and then for each
#         # step in that plan (like "ANALYZE_X", "REQUEST_SIA_MODIFICATION_Y"),
#         # the SEA would perform actions.

#         # Example: After attempting part of the plan for the strategic goal
#         # plan_step_success = await self._execute_current_plan_step_for_strategic_goal(...) 
#         # if not plan_step_success:
#         #     self.goal_manager.update_goal_status(self.current_strategic_goal.id, GoalSt.FAILED_EXECUTION, "A plan step failed.")
#         # elif self._check_if_strategic_goal_achieved(self.current_strategic_goal):
#         #     self.goal_manager.update_goal_status(self.current_strategic_goal.id, GoalSt.COMPLETED_SUCCESS)
