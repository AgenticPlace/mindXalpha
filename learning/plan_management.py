# mindx/learning/plan_management.py
"""
Plan Management System for strategic agents in MindX.
Provides PlanManager for creating, tracking, and executing multi-step plans.
Supports sequential and conceptual parallel execution of actions.
"""
import time
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable, Set
from enum import Enum

from mindx.utils.logging_config import get_logger
from mindx.utils.config import Config # For configuration options

logger = get_logger(__name__)

class PlanSt(Enum): # Renamed to avoid conflict
    """Status of an entire plan."""
    PENDING_GENERATION = "PENDING_GENERATION" # Plan content is being generated
    READY = "READY"                  # Plan is defined and ready for execution
    IN_PROGRESS = "IN_PROGRESS"      # Plan execution has started
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"  # All actions completed successfully
    FAILED_ACTION = "FAILED_ACTION"      # One or more actions failed
    FAILED_VALIDATION = "FAILED_VALIDATION" # Plan deemed invalid before/during execution
    PAUSED = "PAUSED"                # Plan execution is paused
    CANCELLED = "CANCELLED"            # Plan was cancelled

class ActionSt(Enum): # Renamed to avoid conflict
    """Status of a single action within a plan."""
    PENDING = "PENDING"              # Action is waiting for dependencies or its turn
    READY_TO_EXECUTE = "READY_TO_EXECUTE" # All dependencies met, can be picked up by an executor
    IN_PROGRESS = "IN_PROGRESS"      # Action execution has started
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"  # Action completed successfully
    FAILED = "FAILED"                # Action execution failed
    SKIPPED_DEPENDENCY = "SKIPPED_DEPENDENCY" # Skipped because a dependency failed/was skipped
    CANCELLED = "CANCELLED"            # Action was cancelled

class Action: # pragma: no cover
    """Represents a single action within a plan."""
    def __init__(self, 
                 action_type: str, 
                 params: Optional[Dict[str, Any]] = None,
                 action_id: Optional[str] = None,
                 description: Optional[str] = None, # Human-readable description of this step's purpose
                 dependency_ids: Optional[List[str]] = None, # IDs of actions that must complete before this one
                 critical: bool = False): # If this action fails, does the whole plan fail?
        self.id: str = action_id or f"action_{str(uuid.uuid4())[:8]}"
        self.type: str = action_type.upper() # Standardize action type to uppercase
        self.params: Dict[str, Any] = params or {}
        self.description: Optional[str] = description
        self.status: ActionSt = ActionSt.PENDING
        self.result: Any = None
        self.error_message: Optional[str] = None
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.attempt_count: int = 0
        self.dependency_ids: List[str] = dependency_ids or []
        self.is_critical: bool = critical

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "type": self.type, "params": self.params, "description": self.description,
            "status": self.status.value, "result": str(self.result)[:200] if self.result else None, # Truncate result for dict
            "error_message": self.error_message, "started_at": self.started_at, "completed_at": self.completed_at,
            "attempt_count": self.attempt_count, "dependency_ids": self.dependency_ids, "is_critical": self.is_critical
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        action = cls(action_type=data["type"], params=data.get("params"), action_id=data["id"],
                     description=data.get("description"), dependency_ids=data.get("dependency_ids"),
                     critical=data.get("is_critical", False))
        action.status = ActionSt(data.get("status", ActionSt.PENDING.value))
        action.result = data.get("result") # Result is not fully serialized/deserialized here
        action.error_message = data.get("error_message")
        action.started_at = data.get("started_at"); action.completed_at = data.get("completed_at")
        action.attempt_count = data.get("attempt_count",0)
        return action

class Plan: # pragma: no cover
    """Represents a plan to achieve a specific goal."""
    def __init__(self, goal_id: str, plan_id: Optional[str] = None, 
                 description: Optional[str] = None, actions: Optional[List[Action]] = None,
                 created_by: Optional[str] = None): # e.g., "llm_planner", "manual"
        self.id: str = plan_id or f"plan_{str(uuid.uuid4())[:8]}"
        self.goal_id: str = goal_id
        self.description: Optional[str] = description
        self.actions: List[Action] = actions or [] # Ordered list of Action objects
        self.status: PlanSt = PlanSt.READY if actions else PlanSt.PENDING_GENERATION
        self.created_at: float = time.time()
        self.last_updated_at: float = self.created_at
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.created_by: Optional[str] = created_by
        self.current_action_idx: int = 0 # For sequential execution tracking
        self.action_results: Dict[str, Any] = {} # Stores results of completed actions by action_id
        self.failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "goal_id": self.goal_id, "description": self.description,
            "actions": [a.to_dict() for a in self.actions], "status": self.status.value,
            "created_at": self.created_at, "last_updated_at": self.last_updated_at,
            "started_at": self.started_at, "completed_at": self.completed_at,
            "created_by": self.created_by, "current_action_idx": self.current_action_idx,
            "action_results": {k: str(v)[:200] for k,v in self.action_results.items()}, # Truncate results
            "failure_reason": self.failure_reason
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Plan':
        plan = cls(goal_id=data["goal_id"], plan_id=data["id"], description=data.get("description"),
                   actions=[Action.from_dict(ad) for ad in data.get("actions", [])],
                   created_by=data.get("created_by"))
        plan.status = PlanSt(data.get("status", PlanSt.READY.value if plan.actions else PlanSt.PENDING_GENERATION.value))
        plan.created_at = data.get("created_at", time.time()); plan.last_updated_at = data.get("last_updated_at", plan.created_at)
        plan.started_at = data.get("started_at"); plan.completed_at = data.get("completed_at")
        plan.current_action_idx = data.get("current_action_idx", 0)
        plan.action_results = data.get("action_results", {}) # Results are not fully rehydrated
        plan.failure_reason = data.get("failure_reason")
        return plan

class PlanManager:
    """
    Manages the lifecycle of plans for an agent, including creation (conceptual),
    tracking, and orchestrating execution of actions within a plan.
    """
    def __init__(self, agent_id: str, 
                 action_executor: Callable[[Action], Awaitable[Tuple[bool, Any]]], # Agent provides this
                 config_override: Optional[Config] = None,
                 test_mode: bool = False):
        
        if hasattr(self, '_initialized') and self._initialized and not test_mode: # pragma: no cover
            return

        self.agent_id = agent_id
        self.config = config_override or Config()
        self.plans: Dict[str, Plan] = {}  # plan_id -> Plan object
        self.log_prefix = f"PlanManager ({self.agent_id}):"
        
        # Action executor is a callback provided by the owning agent (e.g., BDI or StrategicEvolutionAgent)
        # It's an async function: `async def execute_action_handler(action: Action) -> Tuple[bool, Any]`
        self.action_executor = action_executor
        
        # Parallel execution settings
        self.parallel_execution_enabled: bool = self.config.get(f"plan_manager.{self.agent_id}.parallel_execution.enabled", False)
        self.max_parallel_actions: int = self.config.get(f"plan_manager.{self.agent_id}.parallel_execution.max_concurrent", 3)
        
        logger.info(f"{self.log_prefix} Initialized. Parallel exec: {self.parallel_execution_enabled}, Max parallel: {self.max_parallel_actions}")
        self._initialized = True


    def create_plan(self, goal_id: str, actions_data: List[Dict[str, Any]], 
                    plan_id: Optional[str] = None, description: Optional[str] = None, 
                    created_by: Optional[str] = None) -> Plan: # pragma: no cover
        """Creates a new plan and stores it."""
        if not goal_id: raise ValueError("Goal ID cannot be empty for a plan.")
        if not actions_data or not isinstance(actions_data, list): raise ValueError("Plan must have a list of actions.")

        actions = [Action(**ad) for ad in actions_data] # Create Action objects
        new_plan = Plan(goal_id, plan_id, description, actions, created_by)
        self.plans[new_plan.id] = new_plan
        logger.info(f"{self.log_prefix} Created plan '{new_plan.id}' for goal '{goal_id}' with {len(actions)} actions.")
        return new_plan

    def get_plan(self, plan_id: str) -> Optional[Plan]: # pragma: no cover
        return self.plans.get(plan_id)

    def update_plan_status(self, plan_id: str, status: PlanSt, failure_reason: Optional[str] = None) -> bool: # pragma: no cover
        plan = self.get_plan(plan_id)
        if not plan: logger.warning(f"{self.log_prefix} Cannot update status for non-existent plan '{plan_id}'."); return False
        
        plan.status = status
        plan.last_updated_at = time.time()
        if failure_reason: plan.failure_reason = failure_reason
        elif status != PlanSt.FAILED_ACTION and status != PlanSt.FAILED_VALIDATION: plan.failure_reason = None

        if status == PlanSt.COMPLETED_SUCCESS or status.name.startswith("FAILED"):
            plan.completed_at = time.time()
            if plan.started_at is None: plan.started_at = plan.created_at # Ensure started_at is set

        logger.info(f"{self.log_prefix} Updated plan '{plan_id}' status to {status.name}.")
        return True

    def _update_action_state(self, plan: Plan, action: Action, status: ActionSt, result: Any = None): # pragma: no cover
        """Internal helper to update an action's state and plan's overall status if needed."""
        action.status = status
        action.result = result # Could be success data or error message string
        action.completed_at = time.time()
        plan.action_results[action.id] = result # Store result

        if status == ActionSt.FAILED:
            action.error_message = str(result)
            logger.warning(f"{self.log_prefix} Action '{action.type}' (ID: {action.id}) in plan '{plan.id}' FAILED. Reason: {action.error_message[:100]}")
            if action.is_critical: # If a critical action fails, the whole plan fails
                self.update_plan_status(plan.id, PlanSt.FAILED_ACTION, f"Critical action '{action.type}' (ID: {action.id}) failed.")
        elif status == ActionSt.COMPLETED_SUCCESS:
            logger.info(f"{self.log_prefix} Action '{action.type}' (ID: {action.id}) in plan '{plan.id}' COMPLETED_SUCCESS.")
        
        # Check if all actions in the plan are now terminal (COMPLETED_SUCCESS, FAILED, SKIPPED_DEPENDENCY, CANCELLED)
        all_actions_terminal = all(act.status in [ActionSt.COMPLETED_SUCCESS, ActionSt.FAILED, ActionSt.SKIPPED_DEPENDENCY, ActionSt.CANCELLED] for act in plan.actions)
        if all_actions_terminal and plan.status not in [PlanSt.COMPLETED_SUCCESS, PlanSt.FAILED_ACTION]: # Avoid re-setting if already terminal
            if any(act.status == ActionSt.FAILED for act in plan.actions):
                self.update_plan_status(plan.id, PlanSt.FAILED_ACTION, "One or more actions in the plan failed.")
            else: # All completed or skipped
                self.update_plan_status(plan.id, PlanSt.COMPLETED_SUCCESS)


    async def execute_plan(self, plan_id: str) -> Plan: # pragma: no cover
        """Executes a plan, either sequentially or with limited parallelism."""
        plan = self.get_plan(plan_id)
        if not plan:
            logger.error(f"{self.log_prefix} Cannot execute non-existent plan '{plan_id}'.")
            # Return a dummy plan object indicating failure, or raise
            return Plan(goal_id="unknown", plan_id=plan_id, actions=[], description="Plan not found") 
        
        if plan.status not in [PlanSt.READY, PlanSt.PAUSED]: # Can resume paused plans
            logger.warning(f"{self.log_prefix} Plan '{plan_id}' not in READY or PAUSED state (is {plan.status.name}). Cannot execute.")
            return plan

        self.update_plan_status(plan_id, PlanSt.IN_PROGRESS)
        plan.started_at = time.time()
        logger.info(f"{self.log_prefix} Starting execution of plan '{plan_id}' for goal '{plan.goal_id}'. Parallel enabled: {self.parallel_execution_enabled}")

        if self.parallel_execution_enabled:
            await self._execute_plan_parallel_with_dependencies(plan)
        else:
            await self._execute_plan_sequential(plan)
        
        # Final status should have been set by _update_action_state or explicitly if all actions done
        if plan.status == PlanSt.IN_PROGRESS: # If loop finishes but not all actions terminal (should not happen with correct logic)
            if all(a.status in [ActionSt.COMPLETED_SUCCESS, ActionSt.SKIPPED_DEPENDENCY] for a in plan.actions):
                self.update_plan_status(plan.id, PlanSt.COMPLETED_SUCCESS)
            else: # Some actions might still be pending if dependencies are complex or there was an issue
                logger.warning(f"{self.log_prefix} Plan '{plan_id}' finished execution loop but status is still IN_PROGRESS and not all actions terminal. This might indicate an issue.")
                # Forcing a FAILED status if not all are completed/skipped could be an option
                # For now, leave as IN_PROGRESS if this state is reached.
        
        logger.info(f"{self.log_prefix} Finished execution of plan '{plan_id}'. Final status: {plan.status.name}")
        return plan

    async def _execute_plan_sequential(self, plan: Plan): # pragma: no cover
        """Executes actions in a plan sequentially."""
        for idx, action in enumerate(plan.actions):
            plan.current_action_idx = idx # For tracking
            if action.status == ActionSt.PENDING or action.status == ActionSt.READY_TO_EXECUTE: # Only execute pending/ready
                # Dependency check for sequential (though less critical than parallel)
                deps_met = all(
                    self.plans[plan.id].action_results.get(dep_id) is not None and # Check if result exists
                    next((a for a in self.plans[plan.id].actions if a.id == dep_id), Action("dummy","dummy")).status == ActionSt.COMPLETED_SUCCESS # And was successful
                    for dep_id in action.dependency_ids
                )
                if not deps_met: # pragma: no cover
                    logger.info(f"{self.log_prefix} Action '{action.type}' (ID: {action.id}) skipped due to unmet dependencies in sequential run.")
                    self._update_action_state(plan, action, ActionSt.SKIPPED_DEPENDENCY, "Unmet dependencies")
                    continue

                action.status = ActionSt.IN_PROGRESS; action.started_at = time.time(); action.attempt_count += 1
                logger.info(f"{self.log_prefix} Executing action (Seq) {idx+1}/{len(plan.actions)}: '{action.type}' (ID: {action.id})")
                
                # Resolve parameters before execution
                resolved_params = await self._resolve_action_params_from_plan_results(plan, action.params)
                action_with_resolved_params = Action(action.type, resolved_params, action.id, action.description, action.dependency_ids, action.is_critical)

                success, result = await self.action_executor(action_with_resolved_params) # Agent provides this
                self._update_action_state(plan, action, ActionSt.COMPLETED_SUCCESS if success else ActionSt.FAILED, result)
                
                if not success and action.is_critical: # pragma: no cover
                    logger.error(f"{self.log_prefix} Critical action '{action.type}' (ID: {action.id}) failed. Halting plan '{plan.id}'.")
                    self.update_plan_status(plan.id, PlanSt.FAILED_ACTION, f"Critical action {action.id} failed.")
                    break # Stop sequential execution on critical failure
            elif action.status in [ActionSt.COMPLETED_SUCCESS, ActionSt.SKIPPED_DEPENDENCY]: # pragma: no cover
                 logger.debug(f"{self.log_prefix} Action '{action.type}' (ID: {action.id}) already {action.status.name}. Skipping.")
            elif action.status == ActionSt.FAILED: # pragma: no cover
                 logger.warning(f"{self.log_prefix} Action '{action.type}' (ID: {action.id}) previously failed. Halting sequential plan '{plan.id}'.")
                 if plan.status != PlanSt.FAILED_ACTION: self.update_plan_status(plan.id, PlanSt.FAILED_ACTION, f"Previously failed action {action.id} encountered.")
                 break


    async def _execute_plan_parallel_with_dependencies(self, plan: Plan): # pragma: no cover
        """Executes actions in parallel, respecting dependencies."""
        action_tasks: Dict[str, asyncio.Task] = {} # action_id -> Task
        # action_id -> Future-like object if using thread pool for sync actions
        
        # Loop until plan is terminal or no more actions can be started
        while plan.status == PlanSt.IN_PROGRESS:
            actions_started_this_iteration = 0
            
            # Identify and start executable actions
            for action in plan.actions:
                if action.status == ActionSt.PENDING and action.id not in action_tasks: # Not yet started or running
                    # Check dependencies
                    deps_met = True
                    for dep_id in action.dependency_ids:
                        dep_action_obj = next((a for a in plan.actions if a.id == dep_id), None)
                        if not dep_action_obj or dep_action_obj.status != ActionSt.COMPLETED_SUCCESS:
                            deps_met = False; break
                    
                    if deps_met:
                        if len(action_tasks) < self.max_parallel_actions:
                            action.status = ActionSt.IN_PROGRESS; action.started_at = time.time(); action.attempt_count += 1
                            logger.info(f"{self.log_prefix} Starting parallel action '{action.type}' (ID: {action.id}) for plan '{plan.id}'")
                            
                            # Resolve parameters just before creating the task
                            # This uses results from actions that might have just completed
                            resolved_params = await self._resolve_action_params_from_plan_results(plan, action.params)
                            action_with_resolved_params = Action(action.type, resolved_params, action.id, action.description, action.dependency_ids, action.is_critical)

                            action_tasks[action.id] = asyncio.create_task(
                                self.action_executor(action_with_resolved_params), name=f"ActionTask_{action.id}"
                            )
                            actions_started_this_iteration += 1
                        else: # Max parallel tasks reached, will try in next iteration
                            action.status = ActionSt.READY_TO_EXECUTE # Mark as ready but waiting for slot
            
            if not action_tasks and actions_started_this_iteration == 0: # No running tasks, no new tasks started
                # This means either all done, or stuck due to dependencies/failures
                break # Exit main while loop

            if not action_tasks: # No tasks were running, and none could be started (e.g. all done)
                await asyncio.sleep(0.01); continue # Brief pause before re-checking plan status

            # Wait for at least one task to complete
            done, pending = await asyncio.wait(list(action_tasks.values()), return_when=asyncio.FIRST_COMPLETED)
            
            for task_obj in done:
                # Find corresponding action_id
                completed_action_id: Optional[str] = None
                for aid, lauf_task in list(action_tasks.items()): # Iterate on copy for safe removal
                    if lauf_task == task_obj:
                        completed_action_id = aid; break
                
                if completed_action_id:
                    try:
                        success, result = await task_obj # Get result, may raise if task had exception
                        self._update_action_state(plan, next(a for a in plan.actions if a.id == completed_action_id), ActionSt.COMPLETED_SUCCESS if success else ActionSt.FAILED, result)
                    except Exception as e_task: # pragma: no cover # Exception within the action_executor task
                        logger.error(f"{self.log_prefix} Task for action ID '{completed_action_id}' in plan '{plan.id}' raised exception: {e_task}", exc_info=True)
                        self._update_action_state(plan, next(a for a in plan.actions if a.id == completed_action_id), ActionSt.FAILED, str(e_task))
                    del action_tasks[completed_action_id] # Remove from running tasks
            
            # If plan failed due to a critical action, exit early
            if plan.status.name.startswith("FAILED"): break

        # After loop, clean up any remaining tasks if plan was forcefully terminated (e.g. FAILED_ACTION)
        if action_tasks: # pragma: no cover
            logger.info(f"{self.log_prefix} Plan '{plan.id}' loop ended with {len(action_tasks)} tasks still notionally running. Cancelling them.")
            for task_to_cancel in action_tasks.values(): task_to_cancel.cancel()
            await asyncio.gather(*action_tasks.values(), return_exceptions=True) # Allow cancellation to propagate

    async def _resolve_action_params_from_plan_results(self, plan: Plan, params: Dict[str, Any]) -> Dict[str, Any]: # pragma: no cover
        """Resolves placeholders in action parameters using results from completed actions in the same plan."""
        resolved_params = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("$action_result."):
                # e.g., "$action_result.action_id_123.output_field"
                parts = value[len("$action_result."):].split(".", 1)
                source_action_id = parts[0]
                field_path_str = parts[1] if len(parts) > 1 else None
                
                source_action_result = plan.action_results.get(source_action_id)
                if source_action_result is not None:
                    if field_path_str:
                        try:
                            current_val = source_action_result
                            for field_part in field_path_str.split('.'):
                                if isinstance(current_val, dict): current_val = current_val.get(field_part)
                                elif hasattr(current_val, field_part): current_val = getattr(current_val, field_part)
                                else: current_val = None; break
                            resolved_params[key] = current_val
                            logger.debug(f"{self.log_prefix} Plan '{plan.id}': Resolved param '{key}' from action '{source_action_id}' field '{field_path_str}' to: {str(current_val)[:50]}")
                        except Exception as e_resolve: # pragma: no cover
                            logger.warning(f"{self.log_prefix} Plan '{plan.id}': Failed to resolve field path '{field_path_str}' in result of action '{source_action_id}': {e_resolve}. Using None for param '{key}'.")
                            resolved_params[key] = None
                    else: # Use the whole result
                        resolved_params[key] = source_action_result
                        logger.debug(f"{self.log_prefix} Plan '{plan.id}': Resolved param '{key}' from whole result of action '{source_action_id}'.")

                else: # pragma: no cover
                    logger.warning(f"{self.log_prefix} Plan '{plan.id}': Action result for '{source_action_id}' not found for param '{key}'. Using None.")
                    resolved_params[key] = None # Or raise error, or keep placeholder?
            elif isinstance(value, dict): # pragma: no cover
                resolved_params[key] = await self._resolve_action_params_from_plan_results(plan, value) # Recurse for nested dicts
            elif isinstance(value, list): # pragma: no cover
                resolved_params[key] = [await self._resolve_action_params_from_plan_results(plan, item) if isinstance(item, dict) else item for item in value] # Recurse for dicts in lists
            else:
                resolved_params[key] = value
        return resolved_params


    def get_all_plans(self) -> List[Plan]: return list(self.plans.values()) # pragma: no cover
    def get_plans_by_status(self, status: PlanSt) -> List[Plan]: return [p for p in self.plans.values() if p.status == status] # pragma: no cover
    def get_plans_for_goal(self, goal_id: str) -> List[Plan]: return [p for p in self.plans.values() if p.goal_id == goal_id] # pragma: no cover

    @classmethod
    def reset_instance(cls): # For testing # pragma: no cover
        cls._instance = None
        logger.debug("PlanManager instance reset.")

# Example usage (typically within an agent like BDI or StrategicEvolutionAgent)
async def _plan_manager_example(agent_instance_for_actions): # pragma: no cover
    # This agent_instance_for_actions must have an async method `execute_my_action_type(action: Action)`
    # that the PlanManager's action_executor will call.
    async def dummy_action_executor(action: Action) -> Tuple[bool, Any]:
        logger.info(f"DummyExecutor: Received action {action.type} with params {action.params}")
        await asyncio.sleep(0.1) # Simulate work
        if action.type == "FAIL_THIS": return False, "Intentional failure from dummy executor"
        return True, {"result_data": f"Successfully executed {action.type} for {action.id}", "params_received": action.params}

    pm = PlanManager(agent_id="test_planner_agent", action_executor=dummy_action_executor)
    pm.parallel_execution_enabled = True # Test parallel

    actions_data = [
        {"type": "ACTION_A", "params": {"p1": "valA"}, "id": "act_a"},
        {"type": "ACTION_B", "params": {"p2": "valB"}, "id": "act_b", "dependency_ids": ["act_a"]},
        {"type": "ACTION_C", "params": {"input": "$action_result.act_a.result_data"}, "id": "act_c", "dependency_ids": ["act_a"]}, # Depends on A's result
        {"type": "ACTION_D", "params": {}, "id": "act_d", "dependency_ids": ["act_b", "act_c"]},
        # {"type": "FAIL_THIS", "params": {}, "id": "act_e", "is_critical": True} # Test critical failure
    ]
    plan_obj = pm.create_plan(goal_id="example_goal_1", actions_data=actions_data, description="Test plan execution")
    
    final_plan_state = await pm.execute_plan(plan_obj.id)
    
    print(f"\n--- Plan Execution Result for Plan ID: {final_plan_state.id} ---")
    print(f"Final Plan Status: {final_plan_state.status.name}")
    if final_plan_state.failure_reason: print(f"Failure Reason: {final_plan_state.failure_reason}")
    print("Action Outcomes:")
    for action_obj in final_plan_state.actions:
        print(f"  - Action ID: {action_obj.id}, Type: {action_obj.type}, Status: {action_obj.status.name}, Result: {str(action_obj.result)[:100]}...")

# if __name__ == "__main__": # pragma: no cover
#     class DummyAgentForExecutor:
#         async def execute_my_action_type(self, action: Action) -> Tuple[bool, Any]:
#             # This is where the agent maps action.type to actual tool calls or LLM calls
#             logger.info(f"DummyAgentExecutor: Received action {action.type} with params {action.params}")
#             await asyncio.sleep(0.1) # Simulate work
#             if action.type == "FAIL_THIS": return False, "Intentional failure from dummy executor"
#             return True, {"result_data": f"Successfully executed {action.type} for {action.id}", "params_received": action.params}
#     dummy_agent = DummyAgentForExecutor()
#     asyncio.run(_plan_manager_example(dummy_agent))
