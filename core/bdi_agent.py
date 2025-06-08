# mindx/core/bdi_agent.py
import os
import asyncio
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable, Set
import uuid

from utils.config import Config, PROJECT_ROOT
from utils.logging_config import get_logger
from llm.llm_factory import create_llm_handler
from llm.llm_interface import LLMHandlerInterface

from .belief_system import BeliefSystem, BeliefSource
from learning.goal_management import GoalSt
# Use a forward reference for the type hint to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from learning.strategic_evolution_agent import StrategicEvolutionAgent

logger = get_logger(__name__)

class BaseTool:
    def __init__(self,
                 config: Optional[Config] = None,
                 llm_handler: Optional[LLMHandlerInterface] = None,
                 bdi_agent_ref: Optional['BDIAgent'] = None):
        self.config = config or Config()
        self.llm_handler = llm_handler
        self.bdi_agent_ref = bdi_agent_ref
        self.logger = get_logger(f"tool.{self.__class__.__name__}")

    async def execute(self, **kwargs) -> Any:
        raise NotImplementedError(f"Tool execute method not implemented for {self.__class__.__name__}.")

class BDIAgent:
    def __init__(self,
                 domain: str,
                 belief_system_instance: BeliefSystem,
                 initial_goal: Optional[str] = None,
                 config_override: Optional[Config] = None,
                 strategic_evolution_agent: Optional['StrategicEvolutionAgent'] = None,
                 test_mode: bool = False):

        if hasattr(self, '_initialized_sync_part') and self._initialized_sync_part and not test_mode:
            return

        self.domain = domain
        self.config = config_override or Config()
        self.belief_system = belief_system_instance
        self.agent_id = f"bdi_agent_{self.domain.replace(' ','_').replace('.','-')}_{str(uuid.uuid4())[:4]}"
        self.log_prefix = f"BDI ({self.agent_id} - {self.domain}):"
        
        self.strategic_evolution_agent = strategic_evolution_agent

        self._internal_state: Dict[str, Any] = {
            "status": "INITIALIZED", "last_action_details": None,
            "current_failure_reason": None, "cycle_count": 0,
            "current_run_id": None
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

        bdi_llm_provider_cfg_key = f"bdi.{self.domain}.llm.provider"
        bdi_default_llm_provider_cfg_key = "bdi.default_llm.provider"
        global_default_provider_cfg_key = "llm.default_provider"
        self._bdi_llm_provider_cfg = self.config.get(bdi_llm_provider_cfg_key,
                                      self.config.get(bdi_default_llm_provider_cfg_key,
                                                      self.config.get(global_default_provider_cfg_key)))
        bdi_llm_model_cfg_key = f"bdi.{self.domain}.llm.model"
        bdi_default_llm_model_cfg_key = "bdi.default_llm.model"
        provider_for_model_default = self._bdi_llm_provider_cfg or "unknown_provider"
        global_default_model_for_provider_cfg_key = f"llm.{provider_for_model_default}.default_model_for_reasoning"
        global_default_model_overall_cfg_key = f"llm.{provider_for_model_default}.default_model"
        self._bdi_llm_model_cfg = self.config.get(bdi_llm_model_cfg_key,
                                   self.config.get(bdi_default_llm_model_cfg_key,
                                                   self.config.get(global_default_model_for_provider_cfg_key,
                                                                   self.config.get(global_default_model_overall_cfg_key))))
        self.llm_handler: Optional[LLMHandlerInterface] = None
        self.available_tools: Dict[str, BaseTool] = {}
        
        # This dictionary maps action names to their handler methods.
        self._action_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Tuple[bool, Any]]]] = {
            "SEARCH_WEB": self._execute_search_web,
            "TAKE_NOTES": self._execute_take_notes,
            "SUMMARIZE_TEXT": self._execute_summarize_text,
            "ANALYZE_DATA": self._execute_llm_cognitive_action,
            "SYNTHESIZE_INFO": self._execute_llm_cognitive_action,
            "IDENTIFY_CRITERIA": self._execute_llm_cognitive_action,
            "EVALUATE_OPTIONS": self._execute_llm_cognitive_action,
            "MAKE_DECISION": self._execute_llm_cognitive_action,
            "GENERATE_REPORT": self._execute_llm_cognitive_action,
            "DECOMPOSE_GOAL": self._execute_llm_cognitive_action,
            "ANALYZE_FAILURE": self._execute_llm_cognitive_action,
            "UPDATE_BELIEF": self._execute_update_belief,
            "EXTRACT_PARAMETERS_FOR_ACTION_FROM_GOAL": self._execute_extract_parameters_from_goal,
            "EXECUTE_STRATEGIC_EVOLUTION_CAMPAIGN": self._execute_strategic_evolution_campaign,
            "NO_OP": self._execute_no_op,
            "FAIL_ACTION": self._execute_fail_action,
        }
        if initial_goal: self.set_goal(initial_goal, priority=1, is_primary=True)
        self._initialized_sync_part = True
        self._initialized = False
        logger.info(f"{self.log_prefix} synchronous __init__ complete. LLM and Tools require async_init_components.")

    async def async_init_components(self):
        if self._initialized: return
        logger.info(f"{self.log_prefix} Starting asynchronous component initialization...")
        try:
            self.llm_handler = await create_llm_handler(self._bdi_llm_provider_cfg, self._bdi_llm_model_cfg)
            if self.llm_handler: logger.info(f"{self.log_prefix} Internal LLM initialized: {self.llm_handler.provider_name}/{self.llm_handler.model_name_for_api or 'default'}")
            else: raise RuntimeError("LLM Handler creation returned None unexpectedly in BDI Agent.")
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to initialize LLM handler: {e}", exc_info=True)
            from llm.mock_llm_handler import MockLLMHandler
            self.llm_handler = MockLLMHandler(model_name=f"mock_for_{self.domain}_due_to_error")
            logger.warning(f"{self.log_prefix} Using MockLLMHandler due to initialization error.")
        await self._initialize_tools_async()
        self._initialized = True
        logger.info(f"{self.log_prefix} Fully initialized. LLM Ready: {self.llm_handler is not None and self.llm_handler.provider_name != 'mock'}. Tools: {list(self.available_tools.keys())}")

    async def _initialize_tools_async(self):
        logger.info(f"{self.log_prefix} Initializing tools...")
        tool_configs_domain = self.config.get(f"bdi.{self.domain}.tools", {})
        tool_configs_default = self.config.get("bdi.default_tools", {})
        tool_configs = {**tool_configs_default, **tool_configs_domain}
        if tool_configs.get("web_search", {}).get("enabled", False):
            try:
                from tools.web_search_tool import WebSearchTool
                self.available_tools["web_search"] = WebSearchTool(config=self.config, bdi_agent_ref=self)
                logger.info(f"{self.log_prefix} Initialized WebSearchTool.")
            except ImportError: logger.warning(f"{self.log_prefix} WebSearchTool not found or 'tools' package/module missing.")
            except Exception as e: logger.error(f"{self.log_prefix} Failed WebSearchTool init: {e}", exc_info=True)
        if tool_configs.get("note_taking", {}).get("enabled", True):
            try:
                from tools.note_taking_tool import NoteTakingTool
                notes_dir_rel = self.config.get(f"bdi.{self.domain}.tools.note_taking.notes_dir_relative_to_project",
                                   self.config.get("tools.note_taking.notes_dir_relative_to_project",
                                                   f"data/bdi_notes/{self.domain.replace(' ','_').replace('.','-')}"))
                notes_dir_abs = PROJECT_ROOT / notes_dir_rel
                self.available_tools["note_taking"] = NoteTakingTool(notes_dir=notes_dir_abs, config=self.config, bdi_agent_ref=self)
                logger.info(f"{self.log_prefix} Initialized NoteTakingTool. Notes dir: {notes_dir_abs}")
            except ImportError: logger.warning(f"{self.log_prefix} NoteTakingTool not found or 'tools' package/module missing.")
            except Exception as e: logger.error(f"{self.log_prefix} Failed NoteTakingTool init: {e}", exc_info=True)
        if tool_configs.get("summarization", {}).get("enabled", True):
            try:
                from tools.summarization_tool import SummarizationTool
                sum_llm_handler_to_use = self.llm_handler
                tool_llm_cfg = self.config.get(f"bdi.{self.domain}.tools.summarization.llm",
                                   self.config.get("tools.summarization.llm"))
                if tool_llm_cfg and tool_llm_cfg.get("provider"):
                    try:
                        sum_llm_handler_to_use = await create_llm_handler(tool_llm_cfg["provider"], tool_llm_cfg.get("model"))
                        if sum_llm_handler_to_use: logger.info(f"{self.log_prefix} SummarizationTool using dedicated LLM: {sum_llm_handler_to_use.provider_name}/{sum_llm_handler_to_use.model_name_for_api or 'default'}")
                        else: raise RuntimeError("Dedicated LLM creation for SummarizationTool returned None.")
                    except Exception as e_tool_llm: logger.error(f"{self.log_prefix} Failed to create dedicated LLM for SummarizationTool: {e_tool_llm}. Falling back.", exc_info=True); sum_llm_handler_to_use = self.llm_handler
                if not sum_llm_handler_to_use:
                    logger.warning(f"{self.log_prefix} No LLM available for SummarizationTool (agent's LLM also unavailable). Using Mock.")
                    from llm.mock_llm_handler import MockLLMHandler
                    sum_llm_handler_to_use = MockLLMHandler("mock_summarizer_tool")
                self.available_tools["summarize_text"] = SummarizationTool(llm_handler=sum_llm_handler_to_use, config=self.config, bdi_agent_ref=self)
                if sum_llm_handler_to_use: logger.info(f"{self.log_prefix} SummarizationTool initialized with LLM: {sum_llm_handler_to_use.provider_name}/{sum_llm_handler_to_use.model_name_for_api or 'default'}")
            except ImportError: logger.warning(f"{self.log_prefix} SummarizationTool not found or 'tools' package/module missing.")
            except Exception as e: logger.error(f"{self.log_prefix} Failed SummarizationTool init: {e}", exc_info=True)
        logger.info(f"{self.log_prefix} Tools initialization phase complete. Loaded tools: {list(self.available_tools.keys())}")

    async def update_belief(self, key: str, value: Any, confidence: float = 0.9,
                            source: BeliefSource = BeliefSource.SELF_ANALYSIS,
                            metadata: Optional[Dict] = None, ttl_seconds: Optional[float] = None,
                            is_internal_state: bool = False):
        if is_internal_state:
            self._internal_state[key] = value; logger.debug(f"{self.log_prefix} Internal state updated '{key}' to '{str(value)[:60]}'")
            return
        namespaced_key = f"bdi.{self.domain}.beliefs.{key}"
        await self.belief_system.add_belief(namespaced_key, value, confidence, source, metadata, ttl_seconds)
        val_str = str(value); val_snippet = val_str[:60] + "..." if len(val_str)>60 else val_str
        logger.debug(f"{self.log_prefix} Belief updated '{key}' (NS: '{namespaced_key}') to '{val_snippet}'")

    async def get_belief(self, key: str, from_internal_state: bool = False) -> Optional[Any]:
        if from_internal_state and key in self._internal_state: return self._internal_state[key]
        namespaced_key = f"bdi.{self.domain}.beliefs.{key}"
        belief_obj = await self.belief_system.get_belief(namespaced_key)
        return belief_obj.value if belief_obj else None

    async def get_beliefs_by_prefix(self, key_prefix: str, min_confidence: float = 0.0) -> Dict[str, Any]:
        namespaced_prefix = f"bdi.{self.domain}.beliefs.{key_prefix}"
        belief_objects = await self.belief_system.query_beliefs(partial_key=namespaced_prefix, min_confidence=min_confidence)
        results = {}
        for belief_obj in belief_objects:
            if hasattr(belief_obj, 'key') and belief_obj.key.startswith(namespaced_prefix):
                key_suffix = belief_obj.key[len(namespaced_prefix):]
                if key_suffix.startswith('.'): key_suffix = key_suffix[1:]
                results[key_suffix] = belief_obj.value
        return results

    def set_goal(self, goal_description: str, priority: int = 1, goal_id: Optional[str] = None,
                 is_primary: bool = False, parent_goal_id: Optional[str] = None):
        goal_id = goal_id or str(uuid.uuid4())[:8]
        new_goal_entry = {"id": goal_id, "goal": goal_description, "priority": int(priority), "status": "pending", "added_at": time.time(), "parent_goal_id": parent_goal_id, "plan_history": [], "subgoals_generated_flag": False }
        existing_idx = next((i for i,g in enumerate(self.desires["priority_queue"]) if g["id"] == goal_id), -1)
        if existing_idx != -1: self.desires["priority_queue"][existing_idx].update(new_goal_entry); logger.info(f"{self.log_prefix} Updated goal ID '{goal_id}' to '{goal_description}' (Prio: {priority})")
        else: self.desires["priority_queue"].append(new_goal_entry); logger.info(f"{self.log_prefix} Added goal ID '{goal_id}': '{goal_description}' (Prio: {priority})")
        if is_primary: self.desires["primary_goal_description"] = goal_description; self.desires["primary_goal_id"] = goal_id; logger.info(f"{self.log_prefix} Primary goal set to ID '{goal_id}'")
        self.desires["priority_queue"].sort(key=lambda x: (-x["priority"], x["added_at"]))
        if self._internal_state["status"] in ["INITIALIZED", "COMPLETED_IDLE"]: self._internal_state["status"] = "PENDING_GOAL_PROCESSING"

    def get_current_goal_entry(self) -> Optional[Dict[str, Any]]:
        for goal_entry in self.desires["priority_queue"]:
            if goal_entry.get("status", "pending") == "pending": return goal_entry
        return None

    def set_plan(self, plan_actions: List[Dict[str, Any]], goal_id_for_plan: str, plan_id: Optional[str] = None):
        if not isinstance(plan_actions, list):
            logger.error(f"{self.log_prefix} Invalid plan for goal '{goal_id_for_plan}'. Expected list of actions.")
            self.intentions.update({ "current_plan_actions": [], "plan_status": "INVALID_FORMAT", "current_goal_id_for_plan": goal_id_for_plan, "current_plan_id": None })
            return
        self.intentions["current_plan_id"] = plan_id or str(uuid.uuid4())[:8]
        processed_actions = []
        for i, action in enumerate(plan_actions):
            if not isinstance(action, dict) or "type" not in action: logger.warning(f"{self.log_prefix} Plan {self.intentions['current_plan_id']}: Invalid action structure {action}, skipping."); continue
            action_copy = action.copy(); action_copy.setdefault("id", f"{self.intentions['current_plan_id']}_act{i}"); action_copy.setdefault("params", {}); action_copy.setdefault("preconditions", []); action_copy.setdefault("effects", []); processed_actions.append(action_copy)
        self.intentions["current_plan_actions"] = processed_actions
        self.intentions["plan_status"] = "READY" if processed_actions else "EMPTY_PLAN"
        self.intentions["current_goal_id_for_plan"] = goal_id_for_plan
        self.intentions["current_action_id_in_plan"] = None
        logger.info(f"{self.log_prefix} Set plan ID '{self.intentions['current_plan_id']}' with {len(processed_actions)} actions for goal ID '{goal_id_for_plan}'. Status: {self.intentions['plan_status']}")
        logger.debug(f"{self.log_prefix} Generated plan for goal '{goal_id_for_plan}': {json.dumps(processed_actions, indent=2)}")
        for g_entry in self.desires["priority_queue"]:
            if g_entry["id"] == goal_id_for_plan: g_entry.setdefault("plan_history", []).append(self.intentions["current_plan_id"]); break

    def get_next_action_in_plan(self) -> Optional[Dict[str, Any]]:
        if self.intentions["current_plan_actions"] and self.intentions["plan_status"] == "READY":
            next_action = self.intentions["current_plan_actions"][0]; self.intentions["current_action_id_in_plan"] = next_action.get("id"); return next_action
        self.intentions["current_action_id_in_plan"] = None; return None

    async def action_completed(self, action_id: str, success: bool, result: Any = None):
        current_plan_actions = self.intentions["current_plan_actions"]
        action_idx = next((i for i, act in enumerate(current_plan_actions) if act.get("id") == action_id), -1)
        if action_idx == -1 or (current_plan_actions and action_idx != 0) :
            logger.warning(f"{self.log_prefix} action_completed ID '{action_id}' mismatch or out of order. Head: {current_plan_actions[0].get('id') if current_plan_actions else 'None'}.")
            self._internal_state["last_action_details"] = {"id":action_id, "success": success, "result": str(result)[:200]}; return
        completed_action_details = current_plan_actions.pop(0)
        self._internal_state["last_action_details"] = {**completed_action_details, "success": success, "result": result}
        await self.update_belief(f"action_history.{action_id}", self._internal_state["last_action_details"], 0.95, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600*24*7)
        current_goal_id = self.intentions.get("current_goal_id_for_plan")
        if success:
            logger.info(f"{self.log_prefix} Action '{completed_action_details.get('type')}' (ID {action_id}) for goal '{current_goal_id}' success. Result: {str(result)[:100]}...")
            if not current_plan_actions:
                self.intentions["plan_status"] = "COMPLETED"; logger.info(f"{self.log_prefix} Plan ID '{self.intentions.get('current_plan_id')}' for goal ID '{current_goal_id}' completed.")
                for g_entry in self.desires["priority_queue"]:
                    if g_entry["id"] == current_goal_id: g_entry["status"] = "completed_success"; break
            else: self.intentions["plan_status"] = "READY"
        else:
            self.intentions["plan_status"] = "FAILED"; self._internal_state["current_failure_reason"] = f"Action '{completed_action_details.get('type')}' (ID {action_id}) failed: {str(result)[:200]}"
            logger.warning(f"{self.log_prefix} {self._internal_state['current_failure_reason']}. Plan ID '{self.intentions.get('current_plan_id')}' for goal ID '{current_goal_id}' FAILED.")
            for g_entry in self.desires["priority_queue"]:
                if g_entry["id"] == current_goal_id: g_entry["status"] = "failed_execution"; break
            self.intentions["current_plan_actions"] = []
        self.intentions["current_action_id_in_plan"] = None

    async def perceive(self, external_input: Optional[Dict[str, Any]] = None):
        if not self._initialized: await self.async_init_components()
        logger.debug(f"{self.log_prefix} Perceiving environment...")
        if external_input and isinstance(external_input, dict):
            logger.info(f"{self.log_prefix} Processing external input: {str(external_input)[:200]}...")
            for key, value in external_input.items():
                await self.update_belief(f"environment.{key}", value, 0.9, BeliefSource.PERCEPTION)
        elif external_input:
            await self.update_belief("environment.raw_input", external_input, 0.8, BeliefSource.PERCEPTION)
        if self.intentions["current_plan_actions"] and self.intentions["plan_status"] == "READY":
            if not await self._is_plan_still_valid_async(self.intentions["current_plan_actions"], self.intentions.get("current_goal_id_for_plan")):
                logger.warning(f"{self.log_prefix} Plan ID '{self.intentions.get('current_plan_id')}' for goal ID '{self.intentions.get('current_goal_id_for_plan')}' no longer valid due to perception. Marking INVALID.")
                self.intentions["plan_status"] = "INVALID"

    async def _is_plan_still_valid_async(self, current_plan_actions: List[Dict[str,Any]], goal_id: Optional[str]) -> bool:
        if not self._initialized: await self.async_init_components()
        if not self.llm_handler: return True
        if not current_plan_actions: return True
        if goal_id:
            current_goal_status_str = next((g.get("status") for g in self.desires["priority_queue"] if g["id"] == goal_id), None)
            if current_goal_status_str:
                try:
                    status_enum = GoalSt(current_goal_status_str)
                    if status_enum not in [GoalSt.PENDING, GoalSt.ACTIVE]:
                        logger.warning(f"{self.log_prefix} Plan Monitor: Goal {goal_id} status is {status_enum.name}. Current plan likely obsolete.")
                        return False
                except ValueError:
                     if current_goal_status_str not in ["pending", "in_progress_planning", "decomposed_pending_subgoals"]:
                        logger.warning(f"{self.log_prefix} Plan Monitor: Goal {goal_id} internal status is {current_goal_status_str}. Current plan likely obsolete.")
                        return False
        if self.config.get(f"bdi.{self.domain}.plan_monitoring.llm_check_enabled", False):
            plan_summary = "".join([f"- Action: {a['type']}, Params: {str(a.get('params',{}))[:50]}\n  Preconds: {a.get('preconditions')}\n" for a in current_plan_actions[:2]])
            belief_summary = await self._get_belief_summary_for_prompt(max_beliefs=10, max_value_len=80)
            prompt = (f"Agent Domain: {self.domain}\nCurrent Goal ID: {goal_id}\n"
                      f"Remaining Plan (summary):\n{plan_summary}\n"
                      f"Current Key Beliefs:\n{belief_summary}\n\n"
                      "ASSESSMENT: Is this plan still valid and likely to succeed given the current beliefs? "
                      "Respond with ONLY 'YES' or 'NO', followed by a very brief justification if NO. Example: 'NO - Belief 'resource_X_available' is now false.'")
            try:
                response = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=50, temperature=0.0)
                if response and response.strip().upper().startswith("NO"): logger.info(f"{self.log_prefix} Plan Monitor: LLM indicates plan for goal {goal_id} may be invalid. Reason: {response.strip()[3:]}"); return False
                if response and response.strip().upper().startswith("YES"): return True
            except Exception as e_pm: logger.warning(f"{self.log_prefix} Plan Monitor: LLM check failed: {e_pm}")
        return True

    async def _generate_subgoals_llm(self, parent_goal_entry: Dict[str,Any]) -> List[Dict[str,Any]]:
        if not self._initialized: await self.async_init_components()
        if not self.llm_handler: return []
        parent_goal_desc = parent_goal_entry["goal"]; parent_goal_id = parent_goal_entry["id"]
        logger.info(f"{self.log_prefix} LLM subgoal decomposition for: '{parent_goal_desc}' (Parent ID: {parent_goal_id})")
        belief_summary = await self._get_belief_summary_for_prompt(key_prefix="knowledge",max_beliefs=5)
        prompt = (f"You are a BDI agent's planning assistant for domain '{self.domain}'.\n"
                  f"Current complex high-level goal: '{parent_goal_desc}' (ID: {parent_goal_id}).\n"
                  f"Available general action types: {list(self._action_handlers.keys())}.\nRelevant Beliefs:\n{belief_summary}\n"
                  f"Break this goal into 2-4 actionable subgoals. Each subgoal: string. Respond ONLY with a JSON list of subgoal strings. "
                  f"Example: [\"Subgoal A for main goal\", \"Subgoal B after A\", ...]")
        try:
            subgoals_str_response = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=self.config.get(f"bdi.{self.domain}.subgoal_gen.max_tokens", 700), temperature=0.2, json_mode=True)
            if not subgoals_str_response or (isinstance(subgoals_str_response, str) and subgoals_str_response.startswith("Error:")):
                raise ValueError(f"LLM subgoal gen error: {subgoals_str_response}")
            parsed_subgoal_descs = []
            try: parsed_subgoal_descs = json.loads(subgoals_str_response)
            except json.JSONDecodeError:
                match = re.search(r"\[\s*\"[\s\S]*?\"\s*\]", subgoals_str_response, re.DOTALL)
                if match: parsed_subgoal_descs = json.loads(match.group(0))
                else: logger.error(f"{self.log_prefix} LLM subgoals not valid JSON list and regex found no match. Raw: {subgoals_str_response[:300]}"); raise ValueError("LLM subgoals not valid JSON list of strings and no fallback match.")
            if not isinstance(parsed_subgoal_descs, list) or not all(isinstance(sg, str) and sg for sg in parsed_subgoal_descs):
                raise ValueError("LLM subgoals not list of non-empty strings.")
            subgoal_entries = []
            base_priority = parent_goal_entry.get("priority", 1)
            for i, sg_desc in enumerate(parsed_subgoal_descs):
                subgoal_entries.append({ "id": str(uuid.uuid4())[:8], "goal": sg_desc, "priority": base_priority + (i + 1) * 0.1, "status": "pending", "parent_goal_id": parent_goal_id, "added_at": time.time() + (i * 0.001)})
            logger.info(f"{self.log_prefix} LLM decomposed '{parent_goal_desc}' into {len(subgoal_entries)} subgoals.")
            return subgoal_entries
        except Exception as e:
            logger.error(f"{self.log_prefix} LLM subgoal decomposition failed for '{parent_goal_desc}': {e}", exc_info=True)
            return []

    async def deliberate(self) -> Optional[Dict[str, Any]]:
        if not self._initialized: await self.async_init_components()
        logger.debug(f"{self.log_prefix} Deliberating...")
        current_goal_entry = self.get_current_goal_entry()
        if not current_goal_entry: logger.info(f"{self.log_prefix} No pending goals.")
        if not self.desires.get("primary_goal_description") and not any(g.get("status") == "pending" for g in self.desires["priority_queue"]):
            self._internal_state["status"] = "COMPLETED_IDLE"; return None
        if not current_goal_entry: return None
        logger.info(f"{self.log_prefix} Goal for deliberation: '{current_goal_entry['goal']}' (ID: {current_goal_entry['id']})")
        plan_is_stale_or_missing = not ( self.intentions["plan_status"] == "READY" and self.intentions.get("current_goal_id_for_plan") == current_goal_entry["id"] and self.intentions["current_plan_actions"] ) or self.intentions["plan_status"] == "INVALID"
        is_complex_heuristic = len(current_goal_entry["goal"]) > self.config.get(f"bdi.{self.domain}.goal_complexity_char_threshold", 100) or current_goal_entry["priority"] <= self.config.get(f"bdi.{self.domain}.goal_complexity_priority_threshold", 2)
        if plan_is_stale_or_missing and is_complex_heuristic and self.config.get(f"bdi.{self.domain}.enable_subgoal_decomposition", True) and not current_goal_entry.get("subgoals_generated_flag"):
            logger.info(f"{self.log_prefix} Goal '{current_goal_entry['goal']}' considered for subgoal decomposition.")
            subgoal_entries = await self._generate_subgoals_llm(current_goal_entry)
            if subgoal_entries:
                current_goal_entry["status"] = "decomposed_pending_subgoals"; current_goal_entry["subgoals_generated_flag"] = True
                for sg_entry in subgoal_entries: self.set_goal(sg_entry["goal"], sg_entry["priority"], sg_entry["id"], parent_goal_id=sg_entry["parent_goal_id"])
                new_top_goal = self.get_current_goal_entry(); logger.info(f"{self.log_prefix} Decomposed. New top goal: '{new_top_goal['goal'] if new_top_goal else 'None'}'."); return new_top_goal
        return current_goal_entry

    async def plan(self, goal_entry: Dict[str, Any]) -> bool:
        if not self._initialized: await self.async_init_components()
        if not self.llm_handler:
            logger.error(f"{self.log_prefix} LLM handler not initialized. Cannot generate plan.")
            return False
        goal_description = goal_entry["goal"]; goal_id = goal_entry["id"]
        logger.info(f"{self.log_prefix} Planning for goal '{goal_description}' (ID: {goal_id})")
        if self.intentions["plan_status"] == "READY" and self.intentions.get("current_goal_id_for_plan") == goal_id and self.intentions["current_plan_actions"]:
            logger.debug(f"{self.log_prefix} Using existing valid plan for goal '{goal_id}'."); return True
        
        self.intentions["plan_status"] = "PENDING_GENERATION"
        available_actions_str = ", ".join(sorted(self._action_handlers.keys()))
        
        goal_prefix_hint = "_".join(goal_description.lower().split()[:3])
        belief_summary = await self._get_belief_summary_for_prompt(key_prefix=f"knowledge.{goal_prefix_hint}", max_beliefs=7, max_value_len=100)
        if belief_summary == "No specific relevant beliefs found.": belief_summary = await self._get_belief_summary_for_prompt(max_beliefs=5, max_value_len=80)
        failed_plan_ids_for_this_goal = [pid for pid in goal_entry.get("plan_history", []) if (await self.get_belief(f"plan_details.{pid}.status") or "unknown") == "FAILED"]
        failed_plan_history_str = ""
        if failed_plan_ids_for_this_goal: failed_plan_history_str = f"\nNOTE: Previous plan attempts for this goal (IDs: {', '.join(failed_plan_ids_for_this_goal)}) have FAILED. Analyze why and propose a DIFFERENT or CORRECTED plan."
        
        prompt_for_plan = (
            f"You are a BDI planning module for an AI agent in domain '{self.domain}'.\n"
            f"The current high-priority goal is: '{goal_description}' (Goal ID: {goal_id}).\n"
            f"Available action types for the plan: {available_actions_str}.\n"
            f"Current Relevant Beliefs (summary):\n{belief_summary}{failed_plan_history_str}\n\n"
            f"Generate a sequence of actions (a plan) to achieve this goal. "
            f"Each action MUST be a dictionary with 'type' (string from available types) and 'params' (a dictionary of parameters).\n\n"
            f"CRITICAL DOCTRINE:\n"
            f"1. For any high-level, abstract goal related to system improvement, evolution, bug fixing, or new component development "
            f"(e.g., 'Evolve the system', 'Improve logging', 'Develop a new tool'), you MUST use the "
            f"'EXECUTE_STRATEGIC_EVOLUTION_CAMPAIGN' action. This is the primary mechanism for strategic change.\n"
            f"   - The `params` for this action MUST include 'campaign_goal_description', which should be the original goal text.\n"
            f"   - Example: {{\"type\": \"EXECUTE_STRATEGIC_EVOLUTION_CAMPAIGN\", \"params\": {{\"campaign_goal_description\": \"{goal_description}\"}} }}\n"
            f"2. Use other actions like 'SEARCH_WEB' or 'TAKE_NOTES' for information gathering if needed *before* launching a campaign.\n\n"
            f"CRITICAL INSTRUCTIONS FOR OTHER ACTIONS:\n"
            f"1. For 'UPDATE_BELIEF' action: `params` MUST include 'key' and 'value'.\n"
            f"2. For 'SEARCH_WEB': `params` MUST include 'query'.\n"
            f"3. For 'EXTRACT_PARAMETERS_FOR_ACTION_FROM_GOAL' action: `params` MUST include 'goal_desc' and 'action_to_parameterize'.\n"
            f"If the goal is trivial, a single NO_OP action is acceptable.\n"
            f"Respond ONLY with a single, valid JSON list of these action dictionaries."
        )
        try:
            logger.debug(f"{self.log_prefix} Plan Gen Prompt for '{goal_id}' (start): {prompt_for_plan[:600]}...")
            plan_str_response = await self.llm_handler.generate_text(prompt_for_plan, model=self.llm_handler.model_name_for_api, max_tokens=self.config.get(f"bdi.{self.domain}.planning.max_tokens", 2560), temperature=self.config.get(f"bdi.{self.domain}.planning.temperature", 0.1), json_mode=True)
            if not plan_str_response or (isinstance(plan_str_response, str) and plan_str_response.startswith("Error:")): raise ValueError(f"LLM plan gen error string: {plan_str_response}")
            new_plan_actions: List[Dict] = []
            try: new_plan_actions = json.loads(plan_str_response)
            except json.JSONDecodeError:
                match = re.search(r"\[\s*(\{[\s\S]*?\}(?:\s*,\s*\{[\s\S]*?\})*)\s*\]", plan_str_response, re.DOTALL)
                if match: new_plan_actions = json.loads(match.group(0))
                else: logger.error(f"{self.log_prefix} LLM plan response for goal '{goal_id}' not parsable JSON list. Raw: {plan_str_response[:500]}"); raise ValueError("LLM plan response not parsable JSON list.")
            if not isinstance(new_plan_actions, list): logger.error(f"{self.log_prefix} LLM plan response for goal '{goal_id}' is not a list. Response: {str(new_plan_actions)[:200]}"); raise ValueError("LLM plan must be a list of action dictionaries.")
            if new_plan_actions and not all(isinstance(act, dict) and "type" in act for act in new_plan_actions): logger.error(f"{self.log_prefix} LLM plan for goal '{goal_id}' contains invalid action structures. Plan: {str(new_plan_actions)[:500]}"); raise ValueError("LLM plan invalid structure: not all actions are dicts with a 'type'.")
            self.set_plan(new_plan_actions, goal_id)
            await self.update_belief(f"plan_details.{self.intentions['current_plan_id']}.status", "READY")
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Plan gen failed for '{goal_description}': {e}", exc_info=True)
            self.set_plan([], goal_id); self.intentions["plan_status"] = "FAILED_PLANNING"
            await self.update_belief(f"goal.{goal_id}.planning_failure_reason", str(e), 0.9, BeliefSource.SELF_ANALYSIS)
            return False
            
    async def _analyze_failure_reason(self, failed_action_details: Dict, goal_description: str, plan_id: Optional[str]) -> str:
        if not self._initialized: await self.async_init_components()
        if not self.llm_handler: return "LLM handler not available for failure analysis."
        logger.info(f"{self.log_prefix} Analyzing failure of action '{failed_action_details.get('type')}' for goal '{goal_description}' (Plan ID: {plan_id}).")
        action_str = json.dumps({"type":failed_action_details.get("type"), "params":failed_action_details.get("params")}, indent=2)
        result_str = str(failed_action_details.get("result"))[:500]
        belief_summary = await self._get_belief_summary_for_prompt(max_beliefs=5, max_value_len=60)
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
        )
        try:
            analysis_config_key = f"bdi.{self.domain}.failure_analysis"
            analysis = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=self.config.get(f"{analysis_config_key}.max_tokens", 250), temperature=self.config.get(f"{analysis_config_key}.temperature", 0.3))
            if analysis and not (isinstance(analysis, str) and analysis.startswith("Error:")):
                logger.info(f"{self.log_prefix} Failure analysis result: {analysis}")
                belief_key_suffix = failed_action_details.get('id', 'unknown_action')
                await self.update_belief(f"failure_analysis.{belief_key_suffix}", {"analysis": analysis, "failed_action": action_str, "goal": goal_description}, 0.8, BeliefSource.SELF_ANALYSIS)
                return analysis
        except Exception as e: logger.error(f"{self.log_prefix} LLM failure analysis call failed: {e}")
        return "Failure analysis could not be performed due to an LLM error."

    # --- Start: Added/Implemented Action Handlers ---

    async def _execute_search_web(self, action_dict: Dict) -> Tuple[bool, Any]:
        """Delegates web search to the WebSearchTool."""
        tool = self.available_tools.get("web_search")
        if not tool:
            return False, "SEARCH_WEB action failed: WebSearchTool is not available or enabled."
        
        params = action_dict.get("params", {})
        try:
            result = await tool.execute(**params)
            return True, result
        except Exception as e:
            logger.error(f"{self.log_prefix} WebSearchTool execution failed: {e}", exc_info=True)
            return False, f"Exception in WebSearchTool: {e}"

    async def _execute_take_notes(self, action_dict: Dict) -> Tuple[bool, Any]:
        """Delegates note taking to the NoteTakingTool."""
        tool = self.available_tools.get("note_taking")
        if not tool:
            return False, "TAKE_NOTES action failed: NoteTakingTool is not available or enabled."
        
        params = action_dict.get("params", {})
        try:
            result = await tool.execute(**params)
            # The tool itself may update beliefs, but we return its direct result
            return True, result
        except Exception as e:
            logger.error(f"{self.log_prefix} NoteTakingTool execution failed: {e}", exc_info=True)
            return False, f"Exception in NoteTakingTool: {e}"

    async def _execute_summarize_text(self, action_dict: Dict) -> Tuple[bool, Any]:
        """Delegates text summarization to the SummarizationTool."""
        tool = self.available_tools.get("summarize_text")
        if not tool:
            return False, "SUMMARIZE_TEXT action failed: SummarizationTool is not available or enabled."
            
        params = action_dict.get("params", {})
        try:
            result = await tool.execute(**params)
            return True, result
        except Exception as e:
            logger.error(f"{self.log_prefix} SummarizationTool execution failed: {e}", exc_info=True)
            return False, f"Exception in SummarizationTool: {e}"

    async def _execute_strategic_evolution_campaign(self, action_dict: Dict) -> Tuple[bool, Any]:
        """Hands off a high-level goal to the StrategicEvolutionAgent."""
        if not self.strategic_evolution_agent:
            msg = "Action failed: This BDI agent is not configured with a StrategicEvolutionAgent."
            logger.error(f"{self.log_prefix} {msg}")
            return False, msg

        params = action_dict.get("params", {})
        objective = params.get("campaign_goal_description")
        if not objective:
            return False, "EXECUTE_STRATEGIC_EVOLUTION_CAMPAIGN requires 'campaign_goal_description' in params."

        logger.info(f"{self.log_prefix} Handing off objective to StrategicEvolutionAgent: '{objective}'")
        try:
            # The campaign is a complex, multi-step process handled entirely by the other agent.
            result = await self.strategic_evolution_agent.run_campaign(objective=objective)
            logger.info(f"{self.log_prefix} Strategic evolution campaign completed. Result: {result}")
            return True, result
        except Exception as e:
            error_msg = f"Strategic evolution campaign failed with an exception: {e}"
            logger.exception(f"{self.log_prefix} {error_msg}")
            return False, error_msg
            
    # --- End: Added/Implemented Action Handlers ---

    async def _execute_extract_parameters_from_goal(self, action_dict: Dict) -> Tuple[bool, Any]:
        if not self._initialized: await self.async_init_components()
        if not self.llm_handler: return False, "LLM handler not available for parameter extraction."
        params = action_dict.get("params", {})
        goal_desc_to_parse = params.get("goal_desc")
        action_to_parameterize = params.get("action_to_parameterize")
        output_belief_key = params.get("output_belief_key")
        expected_params_list = params.get("expected_params", [])
        if not all([goal_desc_to_parse, action_to_parameterize, output_belief_key]):
            return False, "EXTRACT_PARAMETERS_FOR_ACTION_FROM_GOAL action requires 'goal_desc', 'action_to_parameterize', and 'output_belief_key' in params."
        logger.info(f"{self.log_prefix} Extracting params for action '{action_to_parameterize}' from goal: '{goal_desc_to_parse[:100]}...'")
        prompt = (
            f"You are an intelligent parameter extraction assistant for a BDI agent.\n"
            f"The current goal is: \"{goal_desc_to_parse}\"\n"
            f"The target action for which parameters need to be extracted is: \"{action_to_parameterize}\"\n"
            f"Respond ONLY with a single JSON dictionary of the extracted parameters. "
            f"If no relevant parameters can be extracted, return an empty JSON object {{}}."
        )
        try:
            extracted_params_str = await self.llm_handler.generate_text(
                prompt, model=self.llm_handler.model_name_for_api,
                max_tokens=self.config.get(f"bdi.llm_actions.extract_parameters.max_tokens", 512),
                temperature=self.config.get(f"bdi.llm_actions.extract_parameters.temperature", 0.0),
                json_mode=True
            )
            if not extracted_params_str or (isinstance(extracted_params_str, str) and extracted_params_str.startswith("Error:")):
                raise ValueError(f"LLM parameter extraction error: {extracted_params_str}")
            extracted_params_dict = json.loads(extracted_params_str)
            if not isinstance(extracted_params_dict, dict):
                extracted_params_dict = {"raw_llm_output": extracted_params_str}
            await self.update_belief(output_belief_key, extracted_params_dict, 0.9, BeliefSource.INFERENCE)
            logger.info(f"{self.log_prefix} Extracted parameters stored in belief '{output_belief_key}': {extracted_params_dict}")
            return True, extracted_params_dict
        except Exception as e:
            logger.error(f"{self.log_prefix} Parameter extraction for action '{action_to_parameterize}' failed: {e}", exc_info=True)
            return False, f"Parameter extraction LLM error: {e}"

    async def _execute_llm_cognitive_action(self, action_dict: Dict) -> Tuple[bool, Any]:
        if not self._initialized: await self.async_init_components()
        if not self.llm_handler: return False, "LLM handler not available for cognitive action."
        action_type = action_dict.get("type", "COGNITIVE_GENERAL").upper()
        resolved_params = action_dict.get("params", {})
        topic = resolved_params.get("topic", self.domain)
        additional_instructions = resolved_params.get("instructions", "")
        input_data_str = resolved_params.get("input_data_str", "No specific input data resolved.")
        logger.info(f"{self.log_prefix} LLM Cognitive Action '{action_type}' for topic '{topic}'. Input (start): {input_data_str[:100]}")
        prompt_templates = {
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
            gen_config = resolved_params.get("generation_config_override", {})
            max_tokens = gen_config.get("max_tokens", self.config.get(f"bdi.llm_actions.{action_type.lower()}.max_tokens", 1500))
            temperature = gen_config.get("temperature", self.config.get(f"bdi.llm_actions.{action_type.lower()}.temperature", 0.2))
            json_mode_requested = gen_config.get("json_mode", action_type in ["DECOMPOSE_GOAL", "IDENTIFY_CRITERIA", "EVALUATE_OPTIONS", "MAKE_DECISION"])
            logger.debug(f"{self.log_prefix} LLM Action '{action_type}' Prompt for {self.llm_handler.model_name_for_api} (start): {prompt[:250]}...")
            result_text = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=max_tokens, temperature=temperature, json_mode=json_mode_requested)
            if result_text is None or (isinstance(result_text, str) and result_text.startswith("Error:")): return False, result_text or "LLM Handler returned None."
            final_result: Any = result_text
            if json_mode_requested:
                try: final_result = json.loads(result_text)
                except json.JSONDecodeError:
                    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", result_text)
                    if match:
                        try: final_result = json.loads(match.group(0))
                        except json.JSONDecodeError: logger.warning(f"{self.log_prefix} LLM '{action_type}' requested JSON, couldn't parse extracted: {match.group(0)[:200]}...");
                    else: logger.warning(f"{self.log_prefix} LLM '{action_type}' requested JSON, no JSON structure found: {result_text[:200]}...");
            belief_key_suffix = topic.replace(' ','_').replace('.','_').replace('/','_')[:40]
            await self.update_belief(f"action_results.{action_type.lower()}.{belief_key_suffix}", final_result, 0.8, BeliefSource.DERIVED, metadata={"prompt_length":len(prompt)})
            return True, final_result
        except Exception as e: logger.error(f"{self.log_prefix} LLM Action '{action_type}' failed: {e}", exc_info=True); return False, f"LLM Action exception: {type(e).__name__}: {e}"

    async def _execute_update_belief(self, action_dict: Dict) -> Tuple[bool, Any]:
        resolved_params = action_dict.get("params", {})
        key = resolved_params.get("key")
        value = resolved_params.get("value")
        confidence = float(resolved_params.get("confidence", 0.85))
        source_str = resolved_params.get("source", "DERIVED")
        if not key: return False, "UPDATE_BELIEF action requires 'key' in resolved params."
        if "value" not in resolved_params: return False, "UPDATE_BELIEF action requires 'value' in resolved params."
        try: source = BeliefSource(source_str.lower())
        except ValueError: source = BeliefSource.DERIVED; logger.warning(f"Invalid source '{source_str}' for UPDATE_BELIEF, using DERIVED.")
        await self.update_belief(key, value, confidence, source)
        return True, f"Belief '{key}' updated to '{str(value)[:100]}...'"

    async def _execute_no_op(self, action_dict: Dict) -> Tuple[bool, Any]: msg = action_dict.get("params", {}).get("message", "NO_OP executed intentionally."); logger.info(f"{self.log_prefix} Executing NO_OP. Message: {msg}"); return True, msg
    async def _execute_fail_action(self, action_dict: Dict) -> Tuple[bool, Any]: failure_message = action_dict.get("params", {}).get("message", "Intentional FAIL_ACTION executed."); logger.warning(f"{self.log_prefix} Executing FAIL_ACTION. Message: {failure_message}"); return False, failure_message

    async def _resolve_action_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if params is None: return {}
        if not isinstance(params, dict):
            resolved_value = await self._resolve_param_value(params)
            if isinstance(resolved_value, dict): return resolved_value
            else: logger.warning(f"{self.log_prefix} Action params placeholder '{params}' did not resolve to a dict. Got {type(resolved_value)}. Using empty params."); return {}
        resolved_params = {}
        for key, value in params.items():
            resolved_params[key] = await self._resolve_param_value(value)
        return resolved_params

    async def _resolve_param_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("$belief."):
                belief_key_full = value[len("$belief."):]
                key_parts = belief_key_full.split('.')
                base_belief_key = key_parts[0]
                belief_val = await self.get_belief(base_belief_key)
                if belief_val is not None and len(key_parts) > 1:
                    current = belief_val
                    for part in key_parts[1:]:
                        if isinstance(current, dict): current = current.get(part)
                        elif hasattr(current, part): current = getattr(current, part)
                        else: current = None; break
                    return current
                return belief_val
            elif value.startswith("$last_action_result"):
                last_res_details = self._internal_state.get("last_action_details", {})
                last_res = last_res_details.get("result") if last_res_details.get("success") else None
                if value == "$last_action_result": return last_res
                if last_res and "." in value:
                    field_path = value[len("$last_action_result."):]
                    try:
                        current = last_res
                        for part in field_path.split('.'):
                            if isinstance(current, dict): current = current.get(part)
                            elif hasattr(current, part): current = getattr(current, part)
                            else: current = None; break
                        return current
                    except Exception: return None
                return last_res
        elif isinstance(value, dict):
            return {k: await self._resolve_param_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [await self._resolve_param_value(item) for item in value]
        return value

    async def execute_current_intention(self) -> bool:
        if not self._initialized: await self.async_init_components()
        action_to_execute = self.get_next_action_in_plan()
        if not action_to_execute: logger.info(f"{self.log_prefix} No action to execute or plan not ready/valid."); return False
        action_type = action_to_execute.get("type", "UNKNOWN_ACTION").upper()
        action_id = action_to_execute.get("id", "unidentified_action")
        logger.info(f"{self.log_prefix} Attempting to execute action '{action_type}' (ID: {action_id})")
        handler = self._action_handlers.get(action_type)
        if not handler:
            logger.error(f"{self.log_prefix} No handler for action type '{action_type}' (ID: {action_id})")
            await self.action_completed(action_id, False, f"Unknown action type: {action_type}"); return False
        try:
            resolved_params = await self._resolve_action_params(action_to_execute.get("params", {}))
            action_dict_for_handler = action_to_execute.copy()
            action_dict_for_handler["params"] = resolved_params
            logger.debug(f"{self.log_prefix} Executing '{action_type}' with resolved params: {str(resolved_params)[:200]}")
            success, result = await handler(action_dict_for_handler)
            await self.action_completed(action_id, success, result); return success
        except Exception as e:
            logger.error(f"{self.log_prefix} Exception during execution of action '{action_type}' (ID: {action_id}): {e}", exc_info=True)
            await self.action_completed(action_id, False, f"Exception during action execution: {type(e).__name__}: {e}"); return False

    async def run(self, max_cycles: int = 10, external_input: Optional[Dict[str, Any]] = None) -> str:
        if not self._initialized: await self.async_init_components()
        self._internal_state["current_run_id"] = str(uuid.uuid4())[:8]; run_id = self._internal_state["current_run_id"]
        logger.info(f"{self.log_prefix} Starting run ID '{run_id}'. Max cycles: {max_cycles}.")
        await self.update_belief("agent_status", "RUNNING", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
        self._internal_state["cycle_count"] = 0; self._internal_state["current_failure_reason"] = None
        if external_input: await self.perceive(external_input)
        while self._internal_state["cycle_count"] < max_cycles:
            self._internal_state["cycle_count"] += 1; cycle_num = self._internal_state["cycle_count"]
            agent_status = self._internal_state["status"]
            await self.update_belief("current_cycle", cycle_num, source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
            logger.info(f"--- {self.log_prefix} Cycle {cycle_num}/{max_cycles} | Status: {agent_status} (Run ID: {run_id}) ---")
            if agent_status not in ["RUNNING", "PENDING_GOAL_PROCESSING"]: logger.info(f"{self.log_prefix} Agent status '{agent_status}'. Ending run ID '{run_id}'."); break
            try:
                if cycle_num > 1 or not external_input: await self.perceive()
                current_goal_entry = await self.deliberate()
                if not current_goal_entry:
                    current_pending_goals = [g for g in self.desires["priority_queue"] if g.get("status") == "pending"]
                    if not self.desires.get("primary_goal_description") and not current_pending_goals:
                         await self.update_belief("agent_status", "COMPLETED_IDLE", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True); logger.info(f"{self.log_prefix} No goals. Idling.")
                    else: logger.info(f"{self.log_prefix} Deliberation yielded no actionable goal this cycle.")
                    await asyncio.sleep(self.config.get("bdi.idle_cycle_delay_seconds", 1.0)); continue
                current_goal_id = current_goal_entry["id"]; current_goal_desc = current_goal_entry["goal"]
                await self.update_belief("current_goal_processing", {"id": current_goal_id, "description": current_goal_desc}, is_internal_state=True)
                plan_is_valid_and_for_current_goal = ( self.intentions["plan_status"] == "READY" and self.intentions.get("current_goal_id_for_plan") == current_goal_id and self.intentions["current_plan_actions"] )
                if not plan_is_valid_and_for_current_goal or self.intentions["plan_status"] == "INVALID":
                    logger.info(f"{self.log_prefix} Planning for goal '{current_goal_desc}' (ID: {current_goal_id}). Prev Plan Status: {self.intentions['plan_status']}")
                    plan_generated = await self.plan(current_goal_entry)
                    if not plan_generated:
                        await self.update_belief("agent_status", "FAILED", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True); self._internal_state["current_failure_reason"] = f"Planning failed for goal: {current_goal_desc}"
                        logger.error(f"{self.log_prefix} {self._internal_state['current_failure_reason']}. Halting run ID '{run_id}'.")
                        for g_e in self.desires["priority_queue"]:
                            if g_e["id"] == current_goal_id: g_e["status"] = "failed_planning"; break
                        break
                if self.intentions["plan_status"] == "READY" and self.intentions["current_plan_actions"]:
                    await self.update_belief(f"plan_details.{self.intentions['current_plan_id']}.status", "IN_PROGRESS")
                    action_succeeded = await self.execute_current_intention()
                    if self.intentions["plan_status"] == "FAILED":
                        logger.warning(f"{self.log_prefix} Execution resulted in FAILED plan for goal '{current_goal_desc}'.")
                        if self.config.get(f"bdi.{self.domain}.failure_analysis.enabled", True):
                            analyzed_reason = await self._analyze_failure_reason(self._internal_state["last_action_details"], current_goal_desc, self.intentions["current_plan_id"])
                            self._internal_state["current_failure_reason"] = f"Analyzed failure for action '{self._internal_state['last_action_details'].get('type')}': {analyzed_reason} (Original: {self._internal_state['current_failure_reason']})"
                            await self.update_belief(f"goal.{current_goal_id}.execution_failure_analysis", analyzed_reason)
                        await self.update_belief("agent_status", "FAILED", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True); break
                elif self.intentions["plan_status"] == "COMPLETED": logger.info(f"{self.log_prefix} Plan for goal '{current_goal_desc}' (ID: {current_goal_id}) already COMPLETED.")
                else: logger.info(f"{self.log_prefix} Skipping execution for '{current_goal_desc}'. Plan status: {self.intentions['plan_status']}.")
                if self.intentions["plan_status"] in ["FAILED_PLANNING", "EMPTY_PLAN", "INVALID_FORMAT"]:
                     for g_e in self.desires["priority_queue"]:
                        if g_e["id"] == current_goal_id: g_e["status"] = self.intentions["plan_status"]; break
                     self.intentions["plan_status"] = None
                if self.intentions["plan_status"] == "COMPLETED":
                    goal_id_completed = self.intentions.get("current_goal_id_for_plan")
                    if goal_id_completed and goal_id_completed == self.desires.get("primary_goal_id"):
                         primary_goal_entry = next((g for g in self.desires["priority_queue"] if g["id"] == goal_id_completed), None)
                         if primary_goal_entry and primary_goal_entry["status"] == "completed_success":
                             await self.update_belief("agent_status", "COMPLETED_GOAL_ACHIEVED", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
                             logger.info(f"{self.log_prefix} PRIMARY GOAL '{self.desires['primary_goal_description']}' (ID: {goal_id_completed}) ACHIEVED!")
                await asyncio.sleep(self.config.get("bdi.agent_cycle_delay_seconds", 0.05))
            except asyncio.CancelledError: await self.update_belief("agent_status", "CANCELLED",source=BeliefSource.SELF_ANALYSIS, is_internal_state=True); logger.info(f"{self.log_prefix} Run ID '{run_id}' cancelled."); break
            except Exception as e: await self.update_belief("agent_status", "FAILED",source=BeliefSource.SELF_ANALYSIS, is_internal_state=True); self._internal_state["current_failure_reason"] = f"Cycle Exception: {type(e).__name__}: {e}"; logger.error(f"{self.log_prefix} Unhandled cycle exception run ID '{run_id}': {e}", exc_info=True); break
        final_status = self._internal_state["status"]
        if final_status == "RUNNING": self._internal_state["status"] = "TIMED_OUT"; final_status = "TIMED_OUT"; logger.warning(f"{self.log_prefix} Max cycles reached for run ID '{run_id}'.")
        logger.info(f"{self.log_prefix} Execution finished for run ID '{run_id}'. Final agent status: {final_status}")
        final_result_msg = f"BDI '{self.domain}' run {final_status} for goal '{self.desires.get('primary_goal_description','N/A')}' (Run ID {run_id}). "
        if final_status == "COMPLETED_GOAL_ACHIEVED": last_res = self._internal_state.get("last_action_details",{}).get("result"); final_result_msg += f"Last result: {str(last_res)[:150]}..."
        elif final_status == "COMPLETED_IDLE": final_result_msg = f"BDI '{self.domain}' run ID '{run_id}' became idle; no further actionable goals."
        else: failure_reason = self._internal_state.get('current_failure_reason', 'Reason not specified.'); final_result_msg += f"Reason/Info: {failure_reason}"
        await self.update_belief(f"run_history.{run_id}.outcome", {"status": final_status, "message": final_result_msg, "domain": self.domain, "primary_goal": self.desires.get("primary_goal_description")}, 0.95, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600*24*30)
        return final_result_msg
    
    async def shutdown(self):
        logger.info(f"{self.log_prefix} Shutting down.")
        await self.update_belief("agent_status", "SHUTDOWN", source=BeliefSource.SELF_ANALYSIS, is_internal_state=True)
    
    async def _get_belief_summary_for_prompt(self, key_prefix: str = "knowledge", max_beliefs: int = 5, max_value_len: int = 80) -> str:
        beliefs = await self.get_beliefs_by_prefix(key_prefix, min_confidence=0.6)
        if not beliefs: return "No specific relevant beliefs found."
        summary_parts = []
        sorted_belief_keys = sorted(beliefs.keys())
        for i, key_suffix in enumerate(sorted_belief_keys):
            if i >= max_beliefs: break
            value = beliefs[key_suffix]
            val_str = str(value)
            summary_parts.append(f"- {key_suffix}: {val_str[:max_value_len]}{'...' if len(val_str) > max_value_len else ''}")
        return "\n".join(summary_parts) if summary_parts else "No relevant beliefs to summarize."
