# mindx/core/bdi_agent.py
import os
import logging
import asyncio
import time
import json
import re 
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable, Set
import uuid

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from mindx.llm.llm_factory import create_llm_handler, LLMHandler
from mindx.core.belief_system import BeliefSystem, BeliefSource

# Conceptual Base class for tools
class BaseTool: # pragma: no cover
    def __init__(self, config: Optional[Config] = None, llm_handler: Optional[LLMHandler] = None, 
                 bdi_agent_ref: Optional['BDIAgent'] = None): # Tools can get a ref to their BDI agent
        self.config = config or Config()
        self.llm_handler = llm_handler
        self.bdi_agent_ref = bdi_agent_ref # To access BDI's beliefs if needed by the tool
    async def execute(self, **kwargs) -> Any: raise NotImplementedError("Tool execute method not implemented.")

logger = get_logger(__name__)

class BDIAgent:
    """
    Belief-Desire-Intention (BDI) agent with LLM-driven intelligence for MindX (v3).
    Uses its internal LLM for planning, subgoal decomposition, failure analysis,
    and plan monitoring. Interacts with a shared BeliefSystem and configured tools.
    This version includes dynamic parameter resolution for actions.
    """

    def __init__(self, 
                 domain: str, 
                 belief_system_instance: BeliefSystem,
                 initial_goal: Optional[str] = None,
                 config_override: Optional[Config] = None,
                 test_mode: bool = False):
        
        if hasattr(self, '_initialized') and self._initialized and not test_mode: # pragma: no cover
            return

        self.domain = domain
        self.config = config_override or Config()
        self.belief_system = belief_system_instance
        self.agent_id = f"bdi_agent_{self.domain.replace(' ','_')}_{str(uuid.uuid4())[:4]}" # Unique ID

        self._internal_state: Dict[str, Any] = {
            "status": "INITIALIZED", "last_action_details": None,
            "current_failure_reason": None, "cycle_count": 0,
            "current_run_id": None # Set at the start of each run
        }
        self.desires: Dict[str, Any] = {
            "primary_goal_description": None, "primary_goal_id": None,
            "priority_queue": [], 
        }
        self.intentions: Dict[str, Any] = {
            "current_plan_id": None, "current_plan_actions": [], 
            "current_goal_id_for_plan": None, "current_action_id_in_plan": None,
            "plan_status": None, 
        }

        bdi_llm_provider = self.config.get(f"bdi.{self.domain}.llm.provider", self.config.get("bdi.default_llm.provider", self.config.get("llm.default_provider")))
        bdi_llm_model = self.config.get(f"bdi.{self.domain}.llm.model", self.config.get("bdi.default_llm.model", self.config.get(f"llm.{bdi_llm_provider}.default_model_for_reasoning", self.config.get(f"llm.{bdi_llm_provider}.default_model"))))
        self.llm_handler: LLMHandler = create_llm_handler(bdi_llm_provider, bdi_llm_model)
        logger.info(f"BDI Agent '{self.agent_id}' (Domain: '{self.domain}') internal LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default'}")

        self.available_tools: Dict[str, BaseTool] = {}
        self._initialize_tools()

        self._action_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Tuple[bool, Any]]]] = {
            "SEARCH_WEB": self._execute_search_web, "TAKE_NOTES": self._execute_take_notes,
            "SUMMARIZE_TEXT": self._execute_summarize_text,
            "ANALYZE_DATA": self._execute_llm_cognitive_action, "SYNTHESIZE_INFO": self._execute_llm_cognitive_action,
            "IDENTIFY_CRITERIA": self._execute_llm_cognitive_action, "EVALUATE_OPTIONS": self._execute_llm_cognitive_action,
            "MAKE_DECISION": self._execute_llm_cognitive_action, "GENERATE_REPORT": self._execute_llm_cognitive_action,
            "DECOMPOSE_GOAL": self._execute_llm_cognitive_action, 
            "ANALYZE_FAILURE": self._execute_llm_cognitive_action,
            "UPDATE_BELIEF": self._execute_update_belief, # New action to directly update beliefs
            "NO_OP": self._execute_no_op, "FAIL_ACTION": self._execute_fail_action,
        }

        if initial_goal: # pragma: no cover
            self.set_goal(initial_goal, priority=1, is_primary=True)

        self._initialized = True
        logger.info(f"BDI agent '{self.agent_id}' for domain '{self.domain}' initialized. Tools: {list(self.available_tools.keys())}")

    def _initialize_tools(self): # pragma: no cover
        tool_configs_domain = self.config.get(f"bdi.{self.domain}.tools", {})
        tool_configs_default = self.config.get("bdi.default_tools", {})
        tool_configs = {**tool_configs_default, **tool_configs_domain}
        
        if tool_configs.get("web_search", {}).get("enabled", False):
            try: from mindx.tools.web_search import WebSearchTool; self.available_tools["web_search"] = WebSearchTool(config=self.config, bdi_agent_ref=self); logger.info(f"BDI '{self.domain}': Initialized WebSearchTool.")
            except ImportError: logger.warning(f"BDI '{self.domain}': WebSearchTool not found.")
            except Exception as e: logger.error(f"BDI '{self.domain}': Failed WebSearchTool init: {e}")

        if tool_configs.get("note_taking", {}).get("enabled", True):
            try: from mindx.tools.note_taking import NoteTakingTool; notes_dir_rel = self.config.get("tools.note_taking.notes_dir_relative_to_project", f"data/bdi_notes/{self.domain.replace(' ','_')}"); notes_dir_abs = PROJECT_ROOT / notes_dir_rel; self.available_tools["note_taking"] = NoteTakingTool(notes_dir=notes_dir_abs, bdi_agent_ref=self); logger.info(f"BDI '{self.domain}': NoteTakingTool. Notes dir: {notes_dir_abs}")
            except ImportError: logger.warning(f"BDI '{self.domain}': NoteTakingTool not found.")
            except Exception as e: logger.error(f"BDI '{self.domain}': Failed NoteTakingTool init: {e}")
        
        if tool_configs.get("summarization", {}).get("enabled", True):
            try:
                from mindx.tools.summarization import SummarizationTool
                sum_provider = self.config.get("tools.summarization.llm.provider", self.llm_handler.provider_name)
                sum_model = self.config.get("tools.summarization.llm.model", self.llm_handler.model_name)
                sum_llm_handler = create_llm_handler(sum_provider, sum_model)
                self.available_tools["summarize_text"] = SummarizationTool(llm_handler=sum_llm_handler, bdi_agent_ref=self)
                logger.info(f"BDI '{self.domain}': SummarizationTool with LLM: {sum_provider}/{sum_model or 'default'}")
            except ImportError: logger.warning(f"BDI '{self.domain}': SummarizationTool not found.")
            except Exception as e: logger.error(f"BDI '{self.domain}': Failed SummarizationTool init: {e}")


    # --- Belief, Desire, Intention Management ---
    async def update_belief(self, key: str, value: Any, confidence: float = 0.9, 
                            source: BeliefSource = BeliefSource.SELF_ANALYSIS, 
                            metadata: Optional[Dict] = None, ttl_seconds: Optional[float] = None,
                            is_internal_state: bool = False): # pragma: no cover
        """Updates a belief. If is_internal_state, updates agent's private _internal_state dict."""
        if is_internal_state:
            self._internal_state[key] = value
            logger.debug(f"BDI '{self.domain}': Internal state updated '{key}' to '{str(value)[:60]}'")
            return

        namespaced_key = f"bdi.{self.domain}.beliefs.{key}"
        await self.belief_system.add_belief(namespaced_key, value, confidence, source, metadata, ttl_seconds)
        val_str = str(value); val_snippet = val_str[:60] + "..." if len(val_str)>60 else val_str
        logger.debug(f"BDI '{self.domain}': Belief updated '{key}' (NS: '{namespaced_key}') to '{val_snippet}'")

    async def get_belief(self, key: str, from_internal_state: bool = False) -> Optional[Any]: # pragma: no cover
        """Retrieves a belief's value. Checks internal state first if specified."""
        if from_internal_state and key in self._internal_state:
            return self._internal_state[key]
        
        namespaced_key = f"bdi.{self.domain}.beliefs.{key}"
        belief_obj = await self.belief_system.get_belief(namespaced_key)
        return belief_obj.value if belief_obj else None

    async def get_beliefs_by_prefix(self, key_prefix: str, min_confidence: float = 0.0) -> Dict[str, Any]: # pragma: no cover
        namespaced_prefix = f"bdi.{self.domain}.beliefs.{key_prefix}"
        belief_objects = await self.belief_system.query_beliefs(partial_key=namespaced_prefix, min_confidence=min_confidence)
        return {b.key.replace(namespaced_prefix, ""): b.value for b in belief_objects}

    def set_goal(self, goal_description: str, priority: int = 1, goal_id: Optional[str] = None, 
                 is_primary: bool = False, parent_goal_id: Optional[str] = None): # pragma: no cover
        # (Unchanged from previous version with enhanced intelligence)
        goal_id = goal_id or str(uuid.uuid4())[:8]; new_goal_entry = {"id": goal_id, "goal": goal_description, "priority": int(priority), "status": "pending", "added_at": time.time(), "parent_goal_id": parent_goal_id, "plan_history": []}
        existing_idx = next((i for i,g in enumerate(self.desires["priority_queue"]) if g["id"] == goal_id), -1)
        if existing_idx != -1: self.desires["priority_queue"][existing_idx].update(new_goal_entry); logger.info(f"BDI '{self.domain}': Updated goal ID '{goal_id}' to '{goal_description}' (Prio: {priority})")
        else: self.desires["priority_queue"].append(new_goal_entry); logger.info(f"BDI '{self.domain}': Added goal ID '{goal_id}': '{goal_description}' (Prio: {priority})")
        if is_primary: self.desires["primary_goal_description"] = goal_description; self.desires["primary_goal_id"] = goal_id; logger.info(f"BDI '{self.domain}': Primary goal set to ID '{goal_id}'")
        self.desires["priority_queue"].sort(key=lambda x: (-x["priority"], x["added_at"]))
        if self._internal_state["status"] in ["INITIALIZED", "COMPLETED_IDLE"]: self._internal_state["status"] = "PENDING_GOAL_PROCESSING"

    def get_current_goal_entry(self) -> Optional[Dict[str, Any]]: # pragma: no cover
        for goal_entry in self.desires["priority_queue"]:
            if goal_entry.get("status", "pending") == "pending": return goal_entry
        return None

    def set_plan(self, plan_actions: List[Dict[str, Any]], goal_id_for_plan: str, plan_id: Optional[str] = None): # pragma: no cover
        # (Unchanged from previous version with enhanced intelligence)
        if not isinstance(plan_actions, list): logger.error(f"BDI '{self.domain}': Invalid plan for goal '{goal_id_for_plan}'."); self.intentions.update({"current_plan_actions": [], "plan_status": "INVALID_FORMAT", "current_goal_id_for_plan": goal_id_for_plan, "current_plan_id": None}); return
        self.intentions["current_plan_id"] = plan_id or str(uuid.uuid4())[:8]
        processed_actions = [];
        for action in plan_actions:
            if not isinstance(action, dict) or "type" not in action: logger.warning(f"BDI Plan {self.intentions['current_plan_id']}: Invalid action {action}, skipping."); continue
            action_copy = action.copy(); action_copy.setdefault("id", f"{self.intentions['current_plan_id']}_act{len(processed_actions)}"); action_copy.setdefault("params", {}); action_copy.setdefault("preconditions", []); action_copy.setdefault("effects", [])
            processed_actions.append(action_copy)
        self.intentions["current_plan_actions"] = processed_actions
        self.intentions["plan_status"] = "READY" if processed_actions else "EMPTY_PLAN"
        self.intentions["current_goal_id_for_plan"] = goal_id_for_plan; self.intentions["current_action_id_in_plan"] = None
        logger.info(f"BDI '{self.domain}': Set plan ID '{self.intentions['current_plan_id']}' with {len(processed_actions)} actions for goal ID '{goal_id_for_plan}'. Status: {self.intentions['plan_status']}")
        for g_entry in self.desires["priority_queue"]:
            if g_entry["id"] == goal_id_for_plan: g_entry.setdefault("plan_history", []).append(self.intentions["current_plan_id"]); break

    def get_next_action_in_plan(self) -> Optional[Dict[str, Any]]: # pragma: no cover
        # (Unchanged from previous version with enhanced intelligence)
        if self.intentions["current_plan_actions"] and self.intentions["plan_status"] == "READY": next_action = self.intentions["current_plan_actions"][0]; self.intentions["current_action_id_in_plan"] = next_action.get("id"); return next_action
        self.intentions["current_action_id_in_plan"] = None; return None

    async def action_completed(self, action_id: str, success: bool, result: Any = None): # pragma: no cover
        # (Unchanged from previous version with enhanced intelligence)
        current_plan_actions = self.intentions["current_plan_actions"]; action_idx = next((i for i, act in enumerate(current_plan_actions) if act.get("id") == action_id), -1)
        if action_idx == -1 or action_idx != 0: logger.warning(f"BDI '{self.domain}': action_completed ID '{action_id}' mismatch. Head: {current_plan_actions[0].get('id') if current_plan_actions else 'None'}."); self._internal_state["last_action_details"] = {"id":action_id, "success": success, "result": str(result)[:200]}; return
        completed_action_details = current_plan_actions.pop(0)
        self._internal_state["last_action_details"] = {**completed_action_details, "success": success, "result": result}
        await self.update_belief(f"action_history.{action_id}", self._internal_state["last_action_details"], 0.95, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600*24*7)
        current_goal_id = self.intentions.get("current_goal_id_for_plan")
        if success:
            logger.info(f"BDI '{self.domain}': Action '{completed_action_details.get('type')}' (ID {action_id}) for goal '{current_goal_id}' success. Result: {str(result)[:100]}...")
            if not current_plan_actions: self.intentions["plan_status"] = "COMPLETED"; logger.info(f"BDI '{self.domain}': Plan ID '{self.intentions.get('current_plan_id')}' for goal ID '{current_goal_id}' completed.")
            for g_entry in self.desires["priority_queue"]: 
                if g_entry["id"] == current_goal_id and not current_plan_actions : g_entry["status"] = "completed_success"; break
            else: self.intentions["plan_status"] = "READY"
        else:
            self.intentions["plan_status"] = "FAILED"; self._internal_state["current_failure_reason"] = f"Action '{completed_action_details.get('type')}' (ID {action_id}) failed: {str(result)[:200]}"
            logger.warning(f"BDI '{self.domain}': {self._internal_state['current_failure_reason']}. Plan ID '{self.intentions.get('current_plan_id')}' for goal ID '{current_goal_id}' FAILED.")
            for g_entry in self.desires["priority_queue"]: 
                if g_entry["id"] == current_goal_id: g_entry["status"] = "failed_execution"; break
            self.intentions["current_plan_actions"] = []
        self.intentions["current_action_id_in_plan"] = None


    # --- BDI Cycle Implementation with Enhanced Intelligence ---
    async def perceive(self, external_input: Optional[Dict[str, Any]] = None): # pragma: no cover
        # (Enhanced with plan validity check)
        logger.debug(f"BDI Domain '{self.domain}': Perceiving environment...")
        if external_input and isinstance(external_input, dict):
            logger.info(f"BDI '{self.domain}': Processing external input: {str(external_input)[:200]}...")
            for key, value in external_input.items():
                await self.update_belief(f"environment.{key}", value, 0.9, BeliefSource.PERCEPTION)
        elif external_input:
            await self.update_belief("environment.raw_input", external_input, 0.8, BeliefSource.PERCEPTION)
        
        if self.intentions["current_plan_actions"] and self.intentions["plan_status"] == "READY":
            if not await self._is_plan_still_valid_async(self.intentions["current_plan_actions"], self.intentions.get("current_goal_id_for_plan")):
                logger.warning(f"BDI '{self.domain}': Plan ID '{self.intentions.get('current_plan_id')}' for goal ID '{self.intentions.get('current_goal_id_for_plan')}' no longer valid. Marking INVALID.")
                self.intentions["plan_status"] = "INVALID" # This will trigger replanning

    async def _is_plan_still_valid_async(self, current_plan_actions: List[Dict[str,Any]], goal_id: Optional[str]) -> bool: # pragma: no cover
        """(LLM-assisted) Checks if the current plan is still valid given current beliefs."""
        if not current_plan_actions: return True 
        
        if goal_id: # Basic rule: if goal status changed, plan might be invalid
            current_goal_status = next((g.get("status") for g in self.desires["priority_queue"] if g["id"] == goal_id), None)
            if current_goal_status and current_goal_status not in ["pending", "in_progress_planning", "decomposed_pending_subgoals"]:
                logger.warning(f"BDI Plan Monitor: Goal {goal_id} status is {current_goal_status}. Current plan likely obsolete.")
                return False

        if self.config.get(f"bdi.{self.domain}.plan_monitoring.llm_check_enabled", False): # Default to False
            plan_summary = "".join([f"- Action: {a['type']}, Params: {str(a.get('params',{}))[:50]}\n  Preconds: {a.get('preconditions')}\n" for a in current_plan_actions[:2]]) # Check first 2 actions
            belief_summary = await self._get_belief_summary_for_prompt(max_beliefs=10, max_value_len=80)

            prompt = (f"Agent Domain: {self.domain}\nCurrent Goal ID: {goal_id}\n"
                      f"Remaining Plan (summary):\n{plan_summary}\n"
                      f"Current Key Beliefs:\n{belief_summary}\n\n"
                      "ASSESSMENT: Is this plan still valid and likely to succeed given the current beliefs? "
                      "Focus on whether critical preconditions (if any mentioned in plan) are violated by current beliefs. "
                      "Respond with ONLY 'YES' or 'NO', followed by a very brief justification if NO. Example: 'NO - Belief 'resource_X_available' is now false.'")
            try:
                response = await self.llm_handler.generate_text(prompt, max_tokens=50, temperature=0.0)
                if response and response.strip().upper().startswith("NO"):
                    logger.info(f"BDI Plan Monitor: LLM indicates plan for goal {goal_id} may be invalid. Reason: {response.strip()[3:]}")
                    return False
                if response and response.strip().upper().startswith("YES"): return True # Explicitly valid
            except Exception as e_pm: logger.warning(f"BDI Plan Monitor: LLM check failed: {e_pm}")
        return True # Default to valid if LLM check off or fails or says YES

    async def _generate_subgoals_llm(self, parent_goal_entry: Dict[str,Any]) -> List[Dict[str,Any]]: # pragma: no cover
        # (Same as previous "COMPLETE WITH ENHANCED LLM-DRIVEN INTELLIGENCE" BDI)
        parent_goal_desc = parent_goal_entry["goal"]; parent_goal_id = parent_goal_entry["id"]
        logger.info(f"BDI '{self.domain}': LLM subgoal decomposition for: '{parent_goal_desc}' (Parent ID: {parent_goal_id})")
        belief_summary = await self._get_belief_summary_for_prompt(key_prefix="knowledge",max_beliefs=5)
        prompt = (f"You are a BDI agent's planning assistant for domain '{self.domain}'.\n"
                  f"Current complex high-level goal: '{parent_goal_desc}' (ID: {parent_goal_id}).\n"
                  f"Available general action types: {list(self._action_handlers.keys())}.\nRelevant Beliefs:\n{belief_summary}\n"
                  f"Break this goal into 2-4 actionable subgoals. Each subgoal: string. Respond ONLY with a JSON list of subgoal strings. "
                  f"Example: [\"Subgoal A for main goal\", \"Subgoal B after A\", ...]")
        try:
            subgoals_str_response = await self.llm_handler.generate_text(prompt, max_tokens=self.config.get(f"bdi.{self.domain}.subgoal_gen.max_tokens", 700), temperature=0.2, json_mode=True)
            if not subgoals_str_response or subgoals_str_response.startswith("Error:"): raise ValueError(f"LLM subgoal gen error: {subgoals_str_response}")
            parsed_subgoal_descs = [] # ... (Robust JSON parsing as in previous BDI) ...
            try: parsed_subgoal_descs = json.loads(subgoals_str_response)
            except json.JSONDecodeError: match = re.search(r"\[\s*\"[\s\S]*?\"\s*\]", subgoals_str_response, re.DOTALL); 
            if match: parsed_subgoal_descs = json.loads(match.group(0))
            else: raise ValueError("LLM subgoals not valid JSON list of strings.")
            if not isinstance(parsed_subgoal_descs, list) or not all(isinstance(sg, str) and sg for sg in parsed_subgoal_descs): raise ValueError("LLM subgoals not list of non-empty strings.")
            subgoal_entries = []; base_priority = parent_goal_entry.get("priority", 1) 
            for i, sg_desc in enumerate(parsed_subgoal_descs): subgoal_entries.append({"id": str(uuid.uuid4())[:8], "goal": sg_desc, "priority": base_priority + (i + 1) * 0.1, "status": "pending", "parent_goal_id": parent_goal_id, "added_at": time.time() + (i * 0.001)})
            logger.info(f"BDI '{self.domain}': LLM decomposed '{parent_goal_desc}' into {len(subgoal_entries)} subgoals.")
            return subgoal_entries
        except Exception as e: logger.error(f"BDI '{self.domain}': LLM subgoal decomposition failed for '{parent_goal_desc}': {e}", exc_info=True); return []

    async def deliberate(self) -> Optional[Dict[str, Any]]: # pragma: no cover
        # (Same as previous "COMPLETE WITH ENHANCED LLM-DRIVEN INTELLIGENCE" BDI, with flag name fix)
        logger.debug(f"BDI Domain '{self.domain}': Deliberating...")
        current_goal_entry = self.get_current_goal_entry();
        if not current_goal_entry: logger.info(f"BDI '{self.domain}': No pending goals.");
        if not self.desires.get("primary_goal_description") and not any(g.get("status") == "pending" for g in self.desires["priority_queue"]): self._internal_state["status"] = "COMPLETED_IDLE"; return None
        if not current_goal_entry: return None # No actionable goal currently
        logger.info(f"BDI '{self.domain}': Goal for deliberation: '{current_goal_entry['goal']}' (ID: {current_goal_entry['id']})")
        plan_needed = not (self.intentions["plan_status"] == "READY" and self.intentions.get("current_goal_id_for_plan") == current_goal_entry["id"] and self.intentions["current_plan_actions"]) or self.intentions["plan_status"] == "INVALID"
        is_complex_heuristic = len(current_goal_entry["goal"]) > self.config.get(f"bdi.{self.domain}.goal_complexity_char_threshold", 100) or current_goal_entry["priority"] <= self.config.get(f"bdi.{self.domain}.goal_complexity_priority_threshold", 2)
        if plan_needed and is_complex_heuristic and self.config.get(f"bdi.{self.domain}.enable_subgoal_decomposition", True) and not current_goal_entry.get("subgoals_generated_flag"): # Use a flag like this
            logger.info(f"BDI '{self.domain}': Goal '{current_goal_entry['goal']}' considered for subgoal decomposition.")
            subgoal_entries = await self._generate_subgoals_llm(current_goal_entry)
            if subgoal_entries:
                current_goal_entry["status"] = "decomposed_pending_subgoals"; current_goal_entry["subgoals_generated_flag"] = True
                for sg_entry in subgoal_entries: self.set_goal(sg_entry["goal"], sg_entry["priority"], sg_entry["id"], parent_goal_id=sg_entry["parent_goal_id"])
                new_top_goal = self.get_current_goal_entry(); logger.info(f"BDI '{self.domain}': Decomposed. New top goal: '{new_top_goal['goal'] if new_top_goal else 'None'}'."); return new_top_goal
        return current_goal_entry


    async def plan(self, goal_entry: Dict[str, Any]) -> bool: # pragma: no cover
        # (Enhanced with failed plan history in prompt)
        goal_description = goal_entry["goal"]; goal_id = goal_entry["id"]
        logger.info(f"BDI Domain '{self.domain}': Planning for goal '{goal_description}' (ID: {goal_id})")
        if self.intentions["plan_status"] == "READY" and self.intentions.get("current_goal_id_for_plan") == goal_id and self.intentions["current_plan_actions"]: return True # Already have a plan
        
        self.intentions["plan_status"] = "PENDING_GENERATION"
        available_actions_str = ", ".join(sorted(self._action_handlers.keys()))
        belief_summary = await self._get_belief_summary_for_prompt(key_prefix=f"knowledge.{goal_description.split(' ')[0].lower()}", max_beliefs=7)
        
        failed_plan_ids_for_this_goal = [pid for pid in goal_entry.get("plan_history", []) 
                                         if (await self.get_belief(f"plan_details.{pid}.status") or "unknown") == "FAILED"]
        failed_plan_history_str = ""
        if failed_plan_ids_for_this_goal: # pragma: no cover
            failed_plan_history_str = f"\nNOTE: Previous plan attempts for this goal (IDs: {', '.join(failed_plan_ids_for_this_goal)}) have FAILED. Analyze why and propose a DIFFERENT or CORRECTED plan."
            # Could also fetch failure reasons for these plans from beliefs if _analyze_failure_reason stores them.

        prompt_for_plan = (
            f"You are a BDI planning module for an AI agent in domain '{self.domain}'.\n"
            f"The current high-priority goal is: '{goal_description}' (Goal ID: {goal_id}).\n"
            f"Available action types for the plan: {available_actions_str}.\n"
            f"Current Relevant Beliefs (summary):\n{belief_summary}{failed_plan_history_str}\n\n"
            f"Generate a sequence of actions (a plan) to achieve this goal. "
            f"Each action MUST be a dictionary with 'type' (string from available types) and 'params' (a dictionary of parameters). "
            f"Optionally, include 'preconditions': List[str] (belief keys that must hold true) and 'effects': List[str] (belief keys that will change). "
            f"The plan should be logical and aim to efficiently achieve the goal. If the goal seems trivial or already met by beliefs, a plan with a single NO_OP action is acceptable.\n"
            f"Respond ONLY with a single, valid JSON list of these action dictionaries."
        )
        try:
            logger.debug(f"BDI Plan Gen Prompt for '{goal_id}' (start): {prompt_for_plan[:300]}...")
            plan_str_response = await self.llm_handler.generate_text(
                prompt_for_plan, 
                max_tokens=self.config.get(f"bdi.{self.domain}.planning.max_tokens", 2048), 
                temperature=self.config.get(f"bdi.{self.domain}.planning.temperature", 0.05), # Very low for structured output
                json_mode=True
            )
            if not plan_str_response or plan_str_response.startswith("Error:"): raise ValueError(f"LLM plan gen error string: {plan_str_response}")
            
            new_plan_actions: List[Dict] = [] # ... (Robust JSON parsing from plan_str_response as in previous BDI) ...
            try: new_plan_actions = json.loads(plan_str_response)
            except json.JSONDecodeError: match = re.search(r"\[\s*(\{[\s\S]*?\}(?:\s*,\s*\{[\s\S]*?\})*)\s*\]", plan_str_response, re.DOTALL);
            if match: new_plan_actions = json.loads(match.group(0))
            else: raise ValueError("LLM plan response not parsable JSON list.")
            if not isinstance(new_plan_actions, list) or not all(isinstance(act, dict) and "type" in act for act in new_plan_actions): raise ValueError("LLM plan invalid structure.")

            self.set_plan(new_plan_actions, goal_id) # This adds current_plan_id to goal's plan_history
            await self.update_belief(f"plan_details.{self.intentions['current_plan_id']}.status", "READY")
            return True
        except Exception as e:
            logger.error(f"BDI '{self.domain}': Plan gen failed for '{goal_description}': {e}", exc_info=True)
            self.set_plan([], goal_id); self.intentions["plan_status"] = "FAILED_PLANNING"
            await self.update_belief(f"goal.{goal_id}.planning_failure_reason", str(e), 0.9, BeliefSource.SELF_ANALYSIS)
            return False

    async def _analyze_failure_reason(self, failed_action_details: Dict, goal_description: str, plan_id: Optional[str]) -> str: # pragma: no cover
        """Uses LLM to analyze why an action/plan failed."""
        logger.info(f"BDI '{self.domain}': Analyzing failure of action '{failed_action_details.get('type')}' for goal '{goal_description}' (Plan ID: {plan_id}).")
        
        action_str = json.dumps({"type":failed_action_details.get("type"), "params":failed_action_details.get("params")}, indent=2)
        result_str = str(failed_action_details.get("result"))[:500] # Truncate long error messages
        belief_summary = await self._get_belief_summary_for_prompt(max_beliefs=5, max_value_len=60) # Brief context
        
        prompt = (
            f"An action in a plan failed for a BDI agent in domain '{self.domain}'.\n"
            f"Current Goal: '{goal_description}' (Plan ID: {plan_id})\n"
            f"Failed Action Details: {action_str}\n"
            f"Result/Error from Action: {result_str}\n"
            f"Relevant Beliefs (summary):\n{belief_summary}\n\n"
            f"Analyze the most likely root cause of this failure. Consider: "
            f"1. Incorrect parameters for the action. "
            f"2. Failure of an external tool or API called by the action. "
            f"3. Violated preconditions or incorrect assumptions (based on beliefs). "
            f"4. Flawed logic in the plan leading to this action. "
            f"Provide a concise analysis (1-3 sentences) of the primary cause and suggest if replanning should try a different approach, fix parameters, or check beliefs. "
            f"Example: 'The SEARCH_WEB action likely failed due to a network timeout. Replanning should consider retrying or using cached information if available.' "
            f"Or: 'The ANALYZE_DATA action received malformed input (param 'source_belief_key' might point to incorrect data). Replanning should verify input data before this action.'"
        )
        try:
            analysis_config_key = f"bdi.{self.domain}.failure_analysis"
            analysis = await self.llm_handler.generate_text(
                prompt, 
                max_tokens=self.config.get(f"{analysis_config_key}.max_tokens", 250), 
                temperature=self.config.get(f"{analysis_config_key}.temperature", 0.3)
            )
            if analysis and not analysis.startswith("Error:"):
                logger.info(f"BDI '{self.domain}': Failure analysis result: {analysis}")
                belief_key_suffix = failed_action_details.get('id', 'unknown_action')
                await self.update_belief(f"failure_analysis.{belief_key_suffix}", {"analysis": analysis, "failed_action": action_str, "goal": goal_description}, 0.8, BeliefSource.SELF_ANALYSIS)
                return analysis
        except Exception as e: # pragma: no cover
            logger.error(f"BDI '{self.domain}': LLM failure analysis call failed: {e}")
        return "Failure analysis could not be performed due to an LLM error."


    # --- Action Handlers (largely same, ensure they use await for update_belief) ---
    async def _execute_search_web(self, action_dict: Dict) -> Tuple[bool, Any]: # pragma: no cover
        # ... (Full logic from previous "Production Candidate Stub v2" BDI)
        tool = self.available_tools.get("web_search");
        if not tool: return False, "Web search tool unavailable."
        try: query = action_dict.get("params", {}).get("query");
        if not query: return False, "Missing 'query' for SEARCH_WEB."
        logger.info(f"BDI '{self.domain}': Web search: '{query}'")
        results = await tool.execute(query=query); belief_key_suffix = re.sub(r'\W+', '_', query.lower())[:50];
        await self.update_belief(f"knowledge.search_results.{belief_key_suffix}", results, 0.8, BeliefSource.PERCEPTION); return True, results
        except Exception as e: logger.error(f"BDI WebSearch error: {e}", exc_info=True); return False, str(e)

    async def _execute_take_notes(self, action_dict: Dict) -> Tuple[bool, Any]: # pragma: no cover
        # ... (Full logic from previous "Production Candidate Stub v2" BDI)
        tool = self.available_tools.get("note_taking");
        if not tool: return False, "Note taking tool unavailable."
        try: params = action_dict.get("params", {}); topic = params.get("topic"); note_action = params.get("action");
        if not topic or not note_action: return False, "Missing 'topic' or 'action' for TAKE_NOTES."
        content = params.get("content"); source_belief_key = params.get("source_belief_key")
        if not content and source_belief_key: content_from_belief = await self.get_belief(source_belief_key); content = str(content_from_belief) if content_from_belief is not None else f"Content for '{source_belief_key}' not found."
        elif not content and note_action in ["add", "update"]: content = f"Placeholder note for {topic}."
        logger.info(f"BDI '{self.domain}': Note action '{note_action}' on topic '{topic}'")
        result = await tool.execute(action=note_action, topic=topic, content=content); return True, result
        except Exception as e: logger.error(f"BDI NoteTaking error: {e}", exc_info=True); return False, str(e)

    async def _execute_summarize_text(self, action_dict: Dict) -> Tuple[bool, Any]: # pragma: no cover
        # ... (Full logic from previous "Production Candidate Stub v2" BDI)
        tool = self.available_tools.get("summarize_text");
        if not tool: return False, "Summarization tool unavailable."
        try: params = action_dict.get("params", {}); topic = params.get("topic", self.domain); text_to_summarize = params.get("text"); source_belief_key = params.get("source_belief_key")
        if not text_to_summarize and source_belief_key: text_to_summarize = str(await self.get_belief(source_belief_key) or "")
        if not text_to_summarize: return False, f"No text to summarize for topic '{topic}'."
        logger.info(f"BDI '{self.domain}': Summarization for topic '{topic}'")
        summary = await tool.execute(text_to_summarize=text_to_summarize, topic_context=topic, max_length=params.get("max_length", 250));
        belief_key_suffix = topic.replace(' ','_')[:30]; await self.update_belief(f"knowledge.summary.{belief_key_suffix}", summary, 0.85, BeliefSource.DERIVED); return True, summary
        except Exception as e: logger.error(f"BDI Summarization error: {e}", exc_info=True); return False, str(e)

    async def _execute_llm_cognitive_action(self, action_dict: Dict) -> Tuple[bool, Any]: # pragma: no cover
        # (Updated to use _resolve_action_params)
        action_type = action_dict.get("type", "COGNITIVE_GENERAL").upper()
        params = action_dict.get("params", {}); topic = params.get("topic", self.domain)
        additional_instructions = params.get("instructions", "") # General instructions for the LLM task
        
        # Resolve dynamic parameters (e.g., from beliefs or last action result)
        resolved_params = await self._resolve_action_params(params)
        input_data_str = resolved_params.get("input_data_str", "No specific input data resolved.") # Default if resolution fails
        source_belief_key = params.get("source_belief_key") # Keep original for logging/prompt if needed

        logger.info(f"BDI '{self.domain}': LLM Cognitive Action '{action_type}' for topic '{topic}'. Resolved Input (start): {input_data_str[:100]}")
        
        prompt_templates = { # These should be configurable or more sophisticated
            "ANALYZE_DATA": f"Task: Analyze Data. Domain: {self.domain}. Topic: {topic}.\nInput Data:\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nProvide concise analysis:",
            "SYNTHESIZE_INFO": f"Task: Synthesize Info. Domain: {self.domain}. Topic: {topic}.\nInput Data:\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nProvide synthesis:",
            "IDENTIFY_CRITERIA": f"Task: Identify Criteria. Domain: {self.domain}. Topic: {topic}.\nContextual Data:\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nList criteria. Respond in JSON: [\"criterion1\", ...].",
            "EVALUATE_OPTIONS": f"Task: Evaluate Options. Domain: {self.domain}. Topic: {topic}.\nInput Data (options & criteria):\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nProvide evaluation. JSON if possible: [{{\"option\":\"A\", \"score\":0.8}},...].",
            "MAKE_DECISION": f"Task: Make Decision. Domain: {self.domain}. Topic: {topic}.\nSupporting Info:\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nState decision & justification. JSON if possible: {{\"decision\":\"X\", \"justification\":\"...\"}}.",
            "GENERATE_REPORT": f"Task: Generate Report. Domain: {self.domain}. Topic: {topic}.\nContent Data:\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nFormat report clearly.",
            "DECOMPOSE_GOAL": f"Task: Decompose Goal. Domain: {self.domain}. Main Goal: {topic}.\nContext:\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nBreak main goal into 2-4 subgoals. Respond ONLY in JSON list of strings: [\"subgoal1\", ...].",
            "ANALYZE_FAILURE": f"Task: Analyze Action/Plan Failure. Domain: {self.domain}. Context: {topic}.\nFailure Data/Error:\n{input_data_str}\n\nSpecific Instructions: {additional_instructions}\nProvide concise root cause analysis."
        }
        prompt = prompt_templates.get(action_type, f"Perform cognitive task '{action_type}' on topic '{topic}'. Data:\n{input_data_str}\n\n{additional_instructions}")
        
        try:
            gen_config = params.get("generation_config_override", {}) # Allow full override from plan
            max_tokens = gen_config.get("max_tokens", self.config.get(f"bdi.llm_actions.{action_type.lower()}.max_tokens", 1500))
            temperature = gen_config.get("temperature", self.config.get(f"bdi.llm_actions.{action_type.lower()}.temperature", 0.2))
            json_mode = gen_config.get("json_mode", action_type in ["DECOMPOSE_GOAL", "IDENTIFY_CRITERIA", "EVALUATE_OPTIONS", "MAKE_DECISION"])

            logger.debug(f"BDI LLM Action '{action_type}' Prompt for {self.llm_handler.model_name} (start): {prompt[:250]}...")
            result_text = await self.llm_handler.generate_text(prompt, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode)
            
            if result_text is None or (isinstance(result_text, str) and result_text.startswith("Error:")): return False, result_text or "LLM Handler returned None."
            
            final_result: Any = result_text
            if json_mode: # Attempt to parse if JSON was expected
                try: final_result = json.loads(result_text)
                except json.JSONDecodeError:
                    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", result_text) # Try to find any JSON object or array
                    if match: 
                        try: final_result = json.loads(match.group(0))
                        except json.JSONDecodeError: logger.warning(f"BDI LLM '{action_type}' requested JSON, couldn't parse extracted: {match.group(0)[:200]}..."); # Keep as text
                    else: logger.warning(f"BDI LLM '{action_type}' requested JSON, no JSON structure found: {result_text[:200]}...");
            
            belief_key_suffix = topic.replace(' ','_').replace('.','_').replace('/','_')[:40] # Sanitize further
            await self.update_belief(f"action_results.{action_type.lower()}.{belief_key_suffix}", final_result, 0.8, BeliefSource.DERIVED, metadata={"prompt_length":len(prompt)})
            return True, final_result
        except Exception as e: logger.error(f"BDI LLM Action '{action_type}' failed: {e}", exc_info=True); return False, f"LLM Action exception: {type(e).__name__}: {e}"

    async def _execute_update_belief(self, action_dict: Dict) -> Tuple[bool, Any]: # pragma: no cover
        """Action to directly update a belief from a plan."""
        params = action_dict.get("params", {})
        key = params.get("key")
        value = params.get("value") # Value can be literal or a placeholder
        confidence = float(params.get("confidence", 0.85))
        source_str = params.get("source", "DERIVED")
        
        if not key: return False, "UPDATE_BELIEF action requires 'key' in params."
        # Value is mandatory, even if None
        if "value" not in params: return False, "UPDATE_BELIEF action requires 'value' in params (can be null/None)."

        try: source = BeliefSource(source_str.lower())
        except ValueError: source = BeliefSource.DERIVED; logger.warning(f"Invalid source '{source_str}' for UPDATE_BELIEF, using DERIVED.")

        # Resolve value if it's a placeholder
        resolved_value = await self._resolve_param_value(value)

        await self.update_belief(key, resolved_value, confidence, source)
        return True, f"Belief '{key}' updated to '{str(resolved_value)[:100]}...'"


    async def _execute_no_op(self, action_dict: Dict) -> Tuple[bool, Any]: msg = action_dict.get("params", {}).get("message", "NO_OP executed intentionally."); logger.info(f"BDI '{self.domain}': Executing NO_OP. Message: {msg}"); return True, msg # pragma: no cover
    async def _execute_fail_action(self, action_dict: Dict) -> Tuple[bool, Any]: failure_message = action_dict.get("params", {}).get("message", "Intentional FAIL_ACTION executed."); logger.warning(f"BDI '{self.domain}': Executing FAIL_ACTION. Message: {failure_message}"); return False, failure_message # pragma: no cover

    async def _resolve_action_params(self, params: Dict[str, Any]) -> Dict[str, Any]: # pragma: no cover
        """Resolves placeholder values in action parameters from beliefs or last action result."""
        resolved_params = {}
        for key, value in params.items():
            resolved_params[key] = await self._resolve_param_value(value)
        return resolved_params

    async def _resolve_param_value(self, value: Any) -> Any: # pragma: no cover
        """Resolves a single parameter value if it's a placeholder string."""
        if isinstance(value, str):
            if value.startswith("$belief."): # e.g., "$belief.knowledge.search_results_topic_xyz"
                belief_key = value[len("$belief."):]
                return await self.get_belief(belief_key)
            elif value.startswith("$last_action_result"): # e.g., "$last_action_result" or "$last_action_result.field_name"
                last_res_details = self._internal_state.get("last_action_details", {})
                last_res = last_res_details.get("result") if last_res_details.get("success") else None
                
                if value == "$last_action_result": return last_res
                
                if last_res and "." in value: # Attempt to access nested field
                    field_path = value[len("$last_action_result."):]
                    try:
                        current = last_res
                        for part in field_path.split('.'):
                            if isinstance(current, dict): current = current.get(part)
                            elif hasattr(current, part): current = getattr(current, part)
                            else: current = None; break
                        return current
                    except Exception: return None # Failed to access path
                return last_res # Return whole result if no field path or not dict/obj
        elif isinstance(value, dict): # Recursively resolve dict values
            return {k: await self._resolve_param_value(v) for k, v in value.items()}
        elif isinstance(value, list): # Recursively resolve list values
            return [await self._resolve_param_value(item) for item in value]
        return value # Not a placeholder or already resolved


    # --- Main BDI Execution Cycle ---
    async def run(self, max_cycles: int = 10, external_input: Optional[Dict[str, Any]] = None) -> str: # pragma: no cover
        """Runs the BDI agent's perceive-deliberate-plan-execute cycle with enhanced failure handling."""
        self._internal_state["current_run_id"] = str(uuid.uuid4())[:8]
        run_id = self._internal_state["current_run_id"]
        logger.info(f"BDI Domain '{self.domain}': Starting run ID '{run_id}'. Max cycles: {max_cycles}.")
        await self.update_belief("agent_status", "RUNNING", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
        self._internal_state["cycle_count"] = 0
        self._internal_state["current_failure_reason"] = None
        
        if external_input: await self.perceive(external_input)

        while self._internal_state["cycle_count"] < max_cycles:
            self._internal_state["cycle_count"] += 1; cycle_num = self._internal_state["cycle_count"]
            agent_status = self._internal_state["status"]
            await self.update_belief("current_cycle", cycle_num, source=BeliefSource.SELF_ANALYSIS, is_internal_state=True) # For external observation
            logger.info(f"--- BDI Cycle {cycle_num}/{max_cycles} | Domain: '{self.domain}' | Agent Status: {agent_status} (Run ID: {run_id}) ---")

            if agent_status not in ["RUNNING", "PENDING_GOAL_PROCESSING"]:
                logger.info(f"BDI: Agent status '{agent_status}'. Ending run ID '{run_id}'.")
                break

            try:
                if cycle_num > 1 or not external_input: await self.perceive()

                current_goal_entry = await self.deliberate()
                if not current_goal_entry:
                    current_pending_goals = [g for g in self.desires["priority_queue"] if g.get("status") == "pending"]
                    if not self.desires.get("primary_goal_description") and not current_pending_goals:
                         await self.update_belief("agent_status", "COMPLETED_IDLE", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
                         logger.info(f"BDI '{self.domain}': No goals. Idling.")
                    else: logger.info(f"BDI '{self.domain}': Deliberation yielded no actionable goal this cycle. Active goals might be blocked or completed.")
                    await asyncio.sleep(self.config.get("bdi.idle_cycle_delay_seconds", 1.0)); continue 

                current_goal_id = current_goal_entry["id"]; current_goal_desc = current_goal_entry["goal"]
                await self.update_belief("current_goal_processing", {"id": current_goal_id, "description": current_goal_desc}, is_internal_state=True)

                plan_is_valid_and_for_current_goal = ( self.intentions["plan_status"] == "READY" and self.intentions.get("current_goal_id_for_plan") == current_goal_id and self.intentions["current_plan_actions"] )
                
                if not plan_is_valid_and_for_current_goal or self.intentions["plan_status"] == "INVALID":
                    logger.info(f"BDI '{self.domain}': Planning for goal '{current_goal_desc}' (ID: {current_goal_id}). Prev Plan Status: {self.intentions['plan_status']}")
                    plan_generated = await self.plan(current_goal_entry) # This is LLM-driven
                    if not plan_generated: # Planning itself failed
                        await self.update_belief("agent_status", "FAILED", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
                        self._internal_state["current_failure_reason"] = f"Planning failed for goal: {current_goal_desc}"
                        logger.error(f"BDI '{self.domain}': {self._internal_state['current_failure_reason']}. Halting run ID '{run_id}'.")
                        for g_e in self.desires["priority_queue"]:
                            if g_e["id"] == current_goal_id: g_e["status"] = "failed_planning"; break
                        # self._save_desires_state() # If desires were persisted
                        break
                
                if self.intentions["plan_status"] == "READY" and self.intentions["current_plan_actions"]:
                    await self.update_belief(f"plan_details.{self.intentions['current_plan_id']}.status", "IN_PROGRESS")
                    action_succeeded = await self.execute_current_intention() # Calls action_completed
                    if self.intentions["plan_status"] == "FAILED": # Action failure led to plan failure
                        logger.warning(f"BDI '{self.domain}': Execution resulted in FAILED plan for goal '{current_goal_desc}'.")
                        if self.config.get(f"bdi.{self.domain}.failure_analysis.enabled", True):
                            analyzed_reason = await self._analyze_failure_reason(self._internal_state["last_action_details"], current_goal_desc, self.intentions["current_plan_id"])
                            self._internal_state["current_failure_reason"] = f"Analyzed failure for action '{self._internal_state['last_action_details'].get('type')}': {analyzed_reason} (Original: {self._internal_state['current_failure_reason']})"
                            await self.update_belief(f"goal.{current_goal_id}.execution_failure_analysis", analyzed_reason)
                        await self.update_belief("agent_status", "FAILED", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
                        break 
                elif self.intentions["plan_status"] == "COMPLETED": logger.info(f"BDI '{self.domain}': Plan for goal '{current_goal_desc}' (ID: {current_goal_id}) already COMPLETED.")
                else: logger.info(f"BDI '{self.domain}': Skipping execution for '{current_goal_desc}'. Plan status: {self.intentions['plan_status']}.")
                if self.intentions["plan_status"] in ["FAILED_PLANNING", "EMPTY_PLAN", "INVALID_FORMAT"]:
                     for g_e in self.desires["priority_queue"]:
                        if g_e["id"] == current_goal_id: g_e["status"] = self.intentions["plan_status"]; break
                     self.intentions["plan_status"] = None # Reset for next goal attempt

                if self.intentions["plan_status"] == "COMPLETED":
                    goal_id_completed = self.intentions.get("current_goal_id_for_plan")
                    if goal_id_completed and goal_id_completed == self.desires.get("primary_goal_id"):
                         await self.update_belief("agent_status", "COMPLETED_GOAL_ACHIEVED", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
                         logger.info(f"BDI '{self.domain}': PRIMARY GOAL '{self.desires['primary_goal_description']}' (ID: {goal_id_completed}) ACHIEVED!")
                
                await asyncio.sleep(self.config.get("bdi.agent_cycle_delay_seconds", 0.05))
            except asyncio.CancelledError: await self.update_belief("agent_status", "CANCELLED",source=BeliefSource.SELF_ANALYSIS, is_internal_state=True); logger.info(f"BDI '{self.domain}': Run ID '{run_id}' cancelled."); break # pragma: no cover
            except Exception as e: await self.update_belief("agent_status", "FAILED",source=BeliefSource.SELF_ANALYSIS, is_internal_state=True); self._internal_state["current_failure_reason"] = f"Cycle Exception: {e}"; logger.error(f"BDI '{self.domain}': Unhandled cycle exception run ID '{run_id}': {e}", exc_info=True); break # pragma: no cover
        
        final_status = self._internal_state["status"]
        if final_status == "RUNNING": self._internal_state["status"] = "TIMED_OUT"; final_status = "TIMED_OUT"; logger.warning(f"BDI '{self.domain}': Max cycles reached for run ID '{run_id}'.")
        logger.info(f"BDI '{self.domain}': Execution finished for run ID '{run_id}'. Final agent status: {final_status}")
        
        final_result_msg = f"BDI '{self.domain}' run {final_status} for goal '{self.desires.get('primary_goal_description','N/A')}' (Run ID {run_id}). "
        if final_status == "COMPLETED_GOAL_ACHIEVED": last_res = self._internal_state.get("last_action_details",{}).get("result"); final_result_msg += f"Last result: {str(last_res)[:150]}..."
        elif final_status == "COMPLETED_IDLE": final_result_msg = f"BDI '{self.domain}' run ID '{run_id}' became idle; no further actionable goals."
        else: failure_reason = self._internal_state.get('current_failure_reason', 'Reason not specified.'); final_result_msg += f"Reason/Info: {failure_reason}"
        await self.update_belief(f"run_history.{run_id}.outcome", {"status": final_status, "message": final_result_msg, "domain": self.domain, "primary_goal": self.desires.get("primary_goal_description")}, 0.95, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600*24*30) # Keep run outcomes for a month
        return final_result_msg
    
    async def _get_belief_summary_for_prompt(self, key_prefix: str = "knowledge", max_beliefs: int = 5, max_value_len: int = 80) -> str: # pragma: no cover
        """Helper to get a concise summary of beliefs for LLM prompts."""
        beliefs = await self.get_beliefs_by_prefix(key_prefix, min_confidence=0.6)
        if not beliefs: return "No specific relevant beliefs found."
        summary_parts = []
        for i, (key_suffix, value) in enumerate(beliefs.items()):
            if i >= max_beliefs: break
            val_str = str(value)
            summary_parts.append(f"- {key_suffix}: {val_str[:max_value_len]}{'...' if len(val_str) > max_value_len else ''}")
        return "\n".join(summary_parts)


    async def shutdown(self): logger.info(f"BDI Agent '{self.agent_id}' for domain '{self.domain}' shutting down."); await self.update_belief("agent_status", "SHUTDOWN", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True) # pragma: no cover
