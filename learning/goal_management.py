# mindx/learning/goal_management.py
"""
Goal Management System for strategic agents in MindX.
Provides GoalManager using a priority queue and various prioritization strategies.
"""
import time
import uuid
import heapq # For priority queue implementation
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod

# Assuming this module is part of the mindx.learning package
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

class GoalSt(Enum): # Renamed to avoid conflict if original GoalStatus is imported elsewhere
    """Constants for goal status values within the GoalManager."""
    PENDING = "PENDING"         # Newly added, not yet considered for planning
    ACTIVE = "ACTIVE"           # Currently being pursued (a plan exists or is being generated)
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS" # Successfully achieved
    COMPLETED_NO_ACTION = "COMPLETED_NO_ACTION" # Goal determined to require no action (e.g., already met)
    FAILED_PLANNING = "FAILED_PLANNING"   # Agent failed to generate a viable plan
    FAILED_EXECUTION = "FAILED_EXECUTION"  # Plan execution failed
    PAUSED_DEPENDENCY = "PAUSED_DEPENDENCY" # Waiting for other goals
    CANCELLED = "CANCELLED"       # Explicitly cancelled

class Goal: # pragma: no cover
    """Represents a goal within the GoalManager."""
    def __init__(self, 
                 description: str, 
                 priority: int, 
                 goal_id: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 parent_goal_id: Optional[str] = None,
                 source: Optional[str] = None): # Source of the goal (e.g., "llm_analysis", "user_directive")
        
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Goal description must be a non-empty string")
        if not isinstance(priority, int) or not (1 <= priority <= 10): # Define a clear priority range, e.g., 1-10
            logger.warning(f"Goal '{description}': Invalid priority {priority}, clamping to range 1-10.")
            priority = max(1, min(10, priority))

        self.id: str = goal_id or f"goal_{str(uuid.uuid4())[:8]}"
        self.description: str = description
        self.priority: int = priority # Higher number means higher priority
        self.status: GoalSt = GoalSt.PENDING
        self.created_at: float = time.time()
        self.last_updated_at: float = self.created_at
        self.metadata: Dict[str, Any] = metadata or {}
        self.parent_goal_id: Optional[str] = parent_goal_id
        self.subgoal_ids: List[str] = [] # IDs of subgoals generated from this goal
        self.dependency_ids: List[str] = [] # IDs of goals this goal depends on
        self.dependent_ids: List[str] = [] # IDs of goals that depend on this goal
        self.current_plan_id: Optional[str] = None
        self.attempt_count: int = 0
        self.failure_reason: Optional[str] = None
        self.source: Optional[str] = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "description": self.description, "priority": self.priority,
            "status": self.status.value, "created_at": self.created_at, 
            "last_updated_at": self.last_updated_at, "metadata": self.metadata,
            "parent_goal_id": self.parent_goal_id, "subgoal_ids": self.subgoal_ids,
            "dependency_ids": self.dependency_ids, "dependent_ids": self.dependent_ids,
            "current_plan_id": self.current_plan_id, "attempt_count": self.attempt_count,
            "failure_reason": self.failure_reason, "source": self.source
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        goal = cls(description=data["description"], priority=data["priority"], goal_id=data["id"], 
                   metadata=data.get("metadata"), parent_goal_id=data.get("parent_goal_id"),
                   source=data.get("source"))
        goal.status = GoalSt(data.get("status", GoalSt.PENDING.value))
        goal.created_at = data.get("created_at", time.time())
        goal.last_updated_at = data.get("last_updated_at", goal.created_at)
        goal.subgoal_ids = data.get("subgoal_ids", [])
        goal.dependency_ids = data.get("dependency_ids", [])
        goal.dependent_ids = data.get("dependent_ids", [])
        goal.current_plan_id = data.get("current_plan_id")
        goal.attempt_count = data.get("attempt_count", 0)
        goal.failure_reason = data.get("failure_reason")
        return goal
    
    def __lt__(self, other: 'Goal') -> bool: # For heapq behavior (higher priority, older = smaller)
        if not isinstance(other, Goal): return NotImplemented
        # Max-heap behavior for priority (higher number is better)
        # Min-heap behavior for timestamp (older is better)
        return (-self.priority, self.created_at) < (-other.priority, other.created_at)

class GoalManager:
    """
    Manages goals for an agent using a priority queue and supports dependencies.
    """
    def __init__(self, agent_id: str):
        self.agent_id: str = agent_id
        self.goals: Dict[str, Goal] = {}  # goal_id -> Goal object
        # Priority queue stores (neg_priority, created_at_timestamp, goal_id)
        # heapq is a min-heap, so negate priority for max-heap behavior.
        self.priority_queue: List[Tuple[int, float, str]] = [] 
        self.log_prefix = f"GoalManager ({self.agent_id}):"
        logger.info(f"{self.log_prefix} Initialized.")

    def add_goal(self, description: str, priority: int = 5, # Default to medium-high priority (assuming 1-10 range)
                 goal_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
                 parent_goal_id: Optional[str] = None, dependency_ids: Optional[List[str]] = None,
                 source: Optional[str] = None) -> Goal:
        """Adds a new goal or updates an existing one if description matches."""
        # Check for existing goal with same description (simple deduplication)
        for existing_goal in self.goals.values(): # pragma: no cover
            if existing_goal.description == description and existing_goal.status not in [GoalSt.COMPLETED_SUCCESS, GoalSt.FAILED, GoalSt.CANCELLED]:
                logger.info(f"{self.log_prefix} Goal with description '{description}' already exists (ID: {existing_goal.id}). Updating priority if higher.")
                if priority > existing_goal.priority:
                    self.update_goal_priority(existing_goal.id, priority)
                return existing_goal # Return existing goal

        new_goal = Goal(description, priority, goal_id, metadata, parent_goal_id, source)
        self.goals[new_goal.id] = new_goal
        
        # Add to priority queue only if no unsatisfied dependencies
        can_be_active = True
        if dependency_ids:
            new_goal.dependency_ids = list(set(dependency_ids)) # Ensure unique
            for dep_id in new_goal.dependency_ids:
                if dep_id not in self.goals: # pragma: no cover
                    logger.warning(f"{self.log_prefix} Goal '{new_goal.id}' has non-existent dependency '{dep_id}'. Marking as PAUSED_DEPENDENCY.")
                    new_goal.status = GoalSt.PAUSED_DEPENDENCY; can_be_active = False; break
                dependency_goal = self.goals[dep_id]
                dependency_goal.dependent_ids.append(new_goal.id) # Register back-link
                if dependency_goal.status != GoalSt.COMPLETED_SUCCESS:
                    can_be_active = False # At least one dependency not met
        
        if can_be_active:
            heapq.heappush(self.priority_queue, (-new_goal.priority, new_goal.created_at, new_goal.id))
        else:
            new_goal.status = GoalSt.PAUSED_DEPENDENCY
            logger.info(f"{self.log_prefix} Goal '{new_goal.description}' (ID: {new_goal.id}) added but PAUSED due to unmet dependencies.")


        logger.info(f"{self.log_prefix} Added goal '{new_goal.description}' (ID: {new_goal.id}), Prio: {new_goal.priority}, Status: {new_goal.status.name}")
        return new_goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        return self.goals.get(goal_id)

    def get_highest_priority_pending_goal(self) -> Optional[Goal]:
        """Gets the highest priority PENDING goal whose dependencies are met."""
        temp_queue = []
        selected_goal: Optional[Goal] = None
        while self.priority_queue:
            neg_prio, ts, goal_id = heapq.heappop(self.priority_queue)
            goal = self.goals.get(goal_id)
            if not goal or goal.status != GoalSt.PENDING: # Already processed or no longer exists
                continue 
            
            # Check dependencies
            dependencies_met = True
            for dep_id in goal.dependency_ids:
                dep_goal = self.goals.get(dep_id)
                if not dep_goal or dep_goal.status != GoalSt.COMPLETED_SUCCESS:
                    dependencies_met = False; break
            
            if dependencies_met:
                selected_goal = goal
                # Add this one back as it's still pending, just selected now
                heapq.heappush(temp_queue, (neg_prio, ts, goal_id))
                break # Found the highest priority actionable goal
            else: # Dependencies not met, put back in temp queue
                heapq.heappush(temp_queue, (neg_prio, ts, goal_id))
        
        # Restore non-selected items to main queue
        while temp_queue:
            heapq.heappush(self.priority_queue, heapq.heappop(temp_queue))
            
        if selected_goal: logger.debug(f"{self.log_prefix} Highest priority pending goal: {selected_goal.id} ('{selected_goal.description}')")
        else: logger.debug(f"{self.log_prefix} No actionable pending goals found in priority queue.")
        return selected_goal


    def update_goal_status(self, goal_id: str, status: GoalSt, failure_reason: Optional[str] = None) -> bool:
        goal = self.goals.get(goal_id)
        if not goal: # pragma: no cover
            logger.error(f"{self.log_prefix} Cannot update status for non-existent goal ID '{goal_id}'.")
            return False
        if not isinstance(status, GoalSt): # pragma: no cover
            logger.error(f"{self.log_prefix} Invalid status type for goal ID '{goal_id}': {type(status)}. Must be GoalSt enum.")
            return False

        previous_status = goal.status
        goal.status = status
        goal.last_updated_at = time.time()
        if failure_reason: goal.failure_reason = failure_reason
        elif status != GoalSt.FAILED_EXECUTION and status != GoalSt.FAILED_PLANNING : goal.failure_reason = None # Clear if not a fail status

        logger.info(f"{self.log_prefix} Updated goal '{goal.description}' (ID: {goal_id}) from {previous_status.name} to {status.name}.")

        # If a goal is completed, check its dependents
        if status == GoalSt.COMPLETED_SUCCESS:
            self._activate_dependent_goals(goal_id)
        
        # If a goal becomes non-pending (active, completed, failed), it should not be selected again by get_highest_priority_pending_goal.
        # The get_highest_priority_pending_goal method implicitly handles this by checking status.
        # No need to explicitly remove from heapq here, as stale entries are skipped.
        # However, for very long-running systems, periodic _rebuild_priority_queue might be useful.
        # Let's rebuild if a terminal status is reached for cleanliness.
        if status in [GoalSt.COMPLETED_SUCCESS, GoalSt.FAILED_EXECUTION, GoalSt.FAILED_PLANNING, GoalSt.CANCELLED, GoalSt.COMPLETED_NO_ACTION]:
            self._rebuild_priority_queue_from_active_goals()

        return True

    def _activate_dependent_goals(self, completed_goal_id: str): # pragma: no cover
        """Checks dependent goals of a completed goal and activates them if all their dependencies are now met."""
        goal = self.goals.get(completed_goal_id)
        if not goal or not goal.dependent_ids: return

        for dependent_id in goal.dependent_ids:
            dependent_goal = self.goals.get(dependent_id)
            if dependent_goal and dependent_goal.status == GoalSt.PAUSED_DEPENDENCY:
                all_deps_met = True
                for dep_id_for_dependent in dependent_goal.dependency_ids:
                    dep_g = self.goals.get(dep_id_for_dependent)
                    if not dep_g or dep_g.status != GoalSt.COMPLETED_SUCCESS:
                        all_deps_met = False; break
                if all_deps_met:
                    logger.info(f"{self.log_prefix} All dependencies met for '{dependent_goal.description}' (ID: {dependent_id}). Setting to PENDING.")
                    dependent_goal.status = GoalSt.PENDING # Change status
                    # Add to priority queue if not already (or update its entry if it was there but non-pending)
                    # For simplicity, rebuild ensures it's correctly added or re-prioritized.
                    self.update_goal_priority(dependent_id, dependent_goal.priority) # This will re-add to queue if needed

    def update_goal_priority(self, goal_id: str, new_priority: int) -> bool: # pragma: no cover
        goal = self.goals.get(goal_id)
        if not goal: logger.error(f"{self.log_prefix} Goal ID '{goal_id}' not found for priority update."); return False
        if not (1 <= new_priority <= 10): new_priority = max(1, min(10, new_priority)); logger.warning(f"{self.log_prefix} Clamped priority for {goal_id} to {new_priority}.")
        
        goal.priority = new_priority
        goal.last_updated_at = time.time()
        logger.info(f"{self.log_prefix} Updated priority for goal '{goal.description}' (ID: {goal_id}) to {new_priority}.")
        
        # Rebuild queue to reflect new priority. More efficient than finding and updating in-place for heapq.
        self._rebuild_priority_queue_from_active_goals()
        return True

    def _rebuild_priority_queue_from_active_goals(self): # pragma: no cover
        """Rebuilds the priority queue from goals that are PENDING or PAUSED_DEPENDENCY."""
        new_queue = []
        for goal_id, goal_data in self.goals.items():
            # Only PENDING goals whose direct dependencies are met, or PAUSED_DEPENDENCY (to be re-evaluated)
            # should be in the active consideration for the queue.
            # get_highest_priority_pending_goal will do the dependency check.
            # So, here we just add all non-terminal goals.
            if goal_data.status not in [GoalSt.COMPLETED_SUCCESS, GoalSt.FAILED_EXECUTION, GoalSt.FAILED_PLANNING, GoalSt.CANCELLED, GoalSt.COMPLETED_NO_ACTION]:
                new_queue.append((-goal_data.priority, goal_data.created_at, goal_id))
        
        self.priority_queue = new_queue
        heapq.heapify(self.priority_queue)
        logger.debug(f"{self.log_prefix} Rebuilt priority queue. Current size: {len(self.priority_queue)}")

    def add_dependency(self, goal_id: str, depends_on_goal_id: str) -> bool: # pragma: no cover
        """goal_id will only start after depends_on_goal_id is COMPLETED_SUCCESS."""
        if goal_id not in self.goals or depends_on_goal_id not in self.goals:
            logger.error(f"{self.log_prefix} One or both goal IDs not found for adding dependency: {goal_id}, {depends_on_goal_id}"); return False
        if goal_id == depends_on_goal_id: logger.warning(f"{self.log_prefix} Goal cannot depend on itself: {goal_id}"); return False
        # Check for circular dependencies (simplified check, a full graph traversal is more robust)
        if self._would_create_circular_dependency(goal_id, depends_on_goal_id): return False

        goal = self.goals[goal_id]
        dependency_goal = self.goals[depends_on_goal_id]

        if depends_on_goal_id not in goal.dependency_ids: goal.dependency_ids.append(depends_on_goal_id)
        if goal_id not in dependency_goal.dependent_ids: dependency_goal.dependent_ids.append(goal_id)
        
        # If the dependency is not yet met, and the goal was PENDING, set it to PAUSED_DEPENDENCY
        if dependency_goal.status != GoalSt.COMPLETED_SUCCESS and goal.status == GoalSt.PENDING:
            self.update_goal_status(goal_id, GoalSt.PAUSED_DEPENDENCY)
            # Rebuild queue as a PENDING goal might have been removed implicitly
            self._rebuild_priority_queue_from_active_goals()

        logger.info(f"{self.log_prefix} Added dependency: Goal '{goal.description}' now depends on '{dependency_goal.description}'.")
        return True

    def _would_create_circular_dependency(self, goal_id_a: str, goal_id_b: str, visited: Optional[Set[str]] = None) -> bool: # pragma: no cover
        """Checks if making goal_id_a depend on goal_id_b would create a cycle."""
        if visited is None: visited = set()
        if goal_id_a in visited: return True # Cycle detected

        visited.add(goal_id_a)
        # If goal_id_b already depends on goal_id_a (directly or transitively)
        # We need to check if goal_id_a is reachable from goal_id_b through its *dependents*
        # This function actually checks: if A depends on B, does B (or its deps) depend on A?
        
        # If B is A, it's a self-loop
        if goal_id_a == goal_id_b: return True

        # Check if any of B's dependencies is A or leads back to A
        dependency_goal_b = self.goals.get(goal_id_b)
        if dependency_goal_b:
            for dep_of_b in dependency_goal_b.dependency_ids:
                if self._would_create_circular_dependency(goal_id_a, dep_of_b, visited.copy()): # Pass copy of visited
                    logger.warning(f"{self.log_prefix} Circular dependency detected: '{goal_id_a}' -> ... -> '{dep_of_b}' -> '{goal_id_b}' (and trying to add '{goal_id_a}' depends on '{goal_id_b}').")
                    return True
        return False

    def get_all_goals(self, status_filter: Optional[List[GoalSt]] = None) -> List[Goal]: # pragma: no cover
        """Returns a list of all goals, optionally filtered by status."""
        all_goals = list(self.goals.values())
        if status_filter:
            return [g for g in all_goals if g.status in status_filter]
        return all_goals

    def get_goal_status_summary(self) -> Dict[str, int]: # pragma: no cover
        """Returns a summary count of goals by status."""
        summary: Dict[str, int] = defaultdict(int)
        for goal in self.goals.values():
            summary[goal.status.name] += 1
        return dict(summary)
        
# --- Goal Prioritization Strategies ---
class GoalPrioritizationStrategy(ABC): # pragma: no cover
    @abstractmethod
    def sort_goals(self, goals: List[Goal]) -> List[Goal]: pass
    def get_name(self) -> str: return self.__class__.__name__

class SimplePriorityThenTimeStrategy(GoalPrioritizationStrategy): # pragma: no cover
    """Sorts by priority (desc), then creation time (asc/oldest first)."""
    def sort_goals(self, goals: List[Goal]) -> List[Goal]:
        return sorted(goals, key=lambda g: (-g.priority, g.created_at))

class UrgencyStrategy(GoalPrioritizationStrategy): # pragma: no cover
    """Calculates urgency: priority + (waiting_time * factor)."""
    def __init__(self, urgency_factor: float = 0.0001): # Factor per second
        self.urgency_factor = urgency_factor
    def sort_goals(self, goals: List[Goal]) -> List[Goal]:
        now = time.time()
        # Calculate urgency score without modifying original goal objects for sorting
        scored_goals = [(goal, goal.priority + ((now - goal.created_at) * self.urgency_factor)) for goal in goals]
        # Sort by urgency score (desc)
        return [g_obj for g_obj, score in sorted(scored_goals, key=lambda x: -x[1])]

# Example usage of GoalManager (typically within an agent like BDI or StrategicEvolutionAgent)
async def _goal_manager_example(): # pragma: no cover
    gm = GoalManager(agent_id="test_agent")
    g1_id = gm.add_goal("Research Topic A", priority=7, source="user_request").id
    g2_id = gm.add_goal("Write report on Topic A", priority=6, source="derived").id
    gm.add_dependency(g2_id, g1_id) # g2 depends on g1
    g3_id = gm.add_goal("Review report on Topic A", priority=5, source="derived").id
    gm.add_dependency(g3_id, g2_id)
    gm.add_goal("Urgent Task X", priority=10, source="critical_alert")

    prioritizer = SimplePriorityThenTimeStrategy()
    
    next_goal = gm.get_highest_priority_pending_goal()
    while next_goal:
        print(f"\nNext Goal to process: {next_goal.description} (ID: {next_goal.id}, Prio: {next_goal.priority})")
        await asyncio.sleep(0.1) # Simulate work
        # Assume it's done for this example
        print(f"Completing goal: {next_goal.description}")
        gm.update_goal_status(next_goal.id, GoalSt.COMPLETED_SUCCESS)
        
        # Re-evaluate next goal
        # In a real BDI agent, this would be part of its main loop
        # The get_highest_priority_pending_goal re-evaluates dependencies implicitly
        
        # For this example, let's re-sort the remaining pending goals based on a strategy
        # (Though get_highest_priority_pending_goal uses the internal heapq)
        all_pending = gm.get_all_goals(status_filter=[GoalSt.PENDING, GoalSt.PAUSED_DEPENDENCY])
        sorted_pending = prioritizer.sort_goals(all_pending)
        print("Current pending goals (sorted by strategy):")
        for g_s in sorted_pending:
            print(f" - {g_s.description} (Prio: {g_s.priority}, Status: {g_s.status.name})")

        next_goal = gm.get_highest_priority_pending_goal()

    print("\nFinal Goal Statuses:")
    for gid, g_data in gm.goals.items():
        print(f"- ID: {gid[:8]}, Desc: {g_data.description[:30]}..., Status: {g_data.status.name}")

# if __name__ == "__main__": # pragma: no cover
#     asyncio.run(_goal_manager_example())
