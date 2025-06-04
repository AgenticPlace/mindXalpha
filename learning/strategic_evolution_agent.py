# mindx/learning/strategic_evolution_agent.py
"""
StrategicEvolutionAgent (SEA) for MindX

This agent is responsible for high-level strategic planning and execution
of self-improvement campaigns for the MindX system. It operates on broad
improvement objectives, using internal GoalManager and PlanManager,
SystemAnalyzerTool for identifying opportunities, and then requests tactical
code modifications via the CoordinatorAgent.
"""

import asyncio
import logging
import os
import time
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from mindx.core.belief_system import BeliefSystem, BeliefSource
from mindx.llm.llm_factory import create_llm_handler, LLMHandler
from mindx.orchestration.coordinator_agent import CoordinatorAgent, InteractionType, InteractionStatus # To call Coordinator
from .goal_management import GoalManager, Goal, GoalSt, SimplePriorityThenTimeStrategy # Using new GoalSt
from .plan_management import PlanManager, Plan, Action, PlanSt as SEA_PlanSt, ActionSt as SEA_ActionSt # Using new PlanSt, ActionSt

logger = get_logger(__name__)

# SystemAnalyzerTool (as defined in previous response, could be a separate import)
class SystemAnalyzerTool: # pragma: no cover # Assuming it's tested elsewhere
    def __init__(self, belief_system: BeliefSystem, llm_handler: LLMHandler, config: Config):
        self.belief_system = belief_system; self.llm_handler = llm_handler; self.config = config
        self.system_capabilities_cache: Optional[Dict[str, Any]] = None
        logger.info(f"SystemAnalyzerTool initialized for {self.__class__.__name__}.")

    async def scan_system_capabilities(self) -> Dict[str, Any]:
        # (Full _scan_codebase_capabilities logic from previous CoordinatorAgent or AGISelfImprovement)
        src_dir = PROJECT_ROOT / "mindx"; capabilities: Dict[str, Any] = {}
        if not src_dir.is_dir(): logger.error(f"SysAnalyzer: Scan dir {src_dir} not found."); return {}
        for item in src_dir.rglob("*.py"):
            if item.name.startswith("__"): continue
            try: rel_path = item.relative_to(PROJECT_ROOT); mod_name = ".".join(rel_path.parts[:-1] + (rel_path.stem,))
            except ValueError: mod_name = item.stem
            try:
                with item.open("r", encoding="utf-8") as f_handle: tree = ast.parse(f_handle.read())
                for node in ast.walk(tree):
                    n_name = getattr(node, 'name', None)
                    if not n_name or n_name.startswith("_"): continue
                    cap_k: Optional[str] = None; cap_t: Optional[str] = None
                    if isinstance(node, ast.ClassDef): cap_k=f"{mod_name}.{n_name}"; cap_t="class"
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if node in getattr(p,'body',[])):
                        cap_k=f"{mod_name}.{n_name}"; cap_t="function"
                    if cap_k and cap_t: capabilities[cap_k] = {"type": cap_t, "name": n_name, "module": mod_name, "path": str(item)}
            except Exception as e_ast: logger.warning(f"SysAnalyzer: AST scan error {item}: {e_ast}")
        self.system_capabilities_cache = capabilities; logger.info(f"SysAnalyzer: Scanned {len(capabilities)} capabilities.")
        return capabilities

    async def analyze_system_for_improvements(self, analysis_focus_hint: str = "overall robustness and maintainability") -> Dict[str, Any]:
        # (Full LLM-driven system analysis logic from previous SystemAnalyzerTool/CoordinatorAgent)
        logger.info(f"SysAnalyzer: LLM analysis. Focus: {analysis_focus_hint}")
        if self.system_capabilities_cache is None: await self.scan_system_capabilities()
        if not self.system_capabilities_cache: return {"error": "SysAnalyzer: No capabilities scanned.", "improvement_suggestions": []}
        modules = sorted(list(set(c.get("module", "unknown") for c in self.system_capabilities_cache.values()))); example_caps = list(self.system_capabilities_cache.keys())[:3]
        system_summary = (f"MindX System: ~{len(modules)} modules, {len(self.system_capabilities_cache)} capabilities. Ex: {', '.join(example_caps)}.")
        prompt = (f"You are an AI System Architect for MindX. Analyze based on system structure. Focus: {analysis_focus_hint}\n"
                  f"Structure: {system_summary}\nIdentify 1-3 high-impact improvement areas. For each: 'target_component_path' (e.g., 'mindx.core.belief_system'), "
                  f"'suggestion' (what & why), 'priority' (1-10), 'is_critical_target' (boolean). Respond ONLY in JSON: {{\"improvement_suggestions\": [{{...}}]}}")
        try:
            max_tokens = self.config.get("coordinator.system_analysis.max_tokens", 1800) # Using existing config key
            temp = self.config.get("coordinator.system_analysis.temperature", 0.1)
            response_str = await self.llm_handler.generate_text(prompt, max_tokens=max_tokens, temperature=temp, json_mode=True)
            analysis_result = {} 
            if response_str and not response_str.startswith("Error:"):
                try: analysis_result = json.loads(response_str)
                except json.JSONDecodeError: match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", response_str, re.DOTALL);
                if match: try: analysis_result = json.loads(match.group(1))
                except json.JSONDecodeError: pass
                if not analysis_result: json_start = response_str.find('{'); json_end = response_str.rfind('}') + 1;
                if json_start != -1 and json_end > json_start: analysis_result = json.loads(response_str[json_start:json_end])
                else: raise ValueError(f"No JSON in SysAnalyzer LLM response. Raw: {response_str[:300]}")
            else: raise ValueError(f"SysAnalyzer LLM error/empty: {response_str}")
            if "improvement_suggestions" not in analysis_result or not isinstance(analysis_result["improvement_suggestions"], list): analysis_result["improvement_suggestions"] = []
            valid_suggs = [];
            for s_idx, s_item in enumerate(analysis_result.get("improvement_suggestions",[])):
                if isinstance(s_item, dict) and all(k in s_item for k in ["target_component_path", "suggestion", "priority", "is_critical_target"]): valid_suggs.append(s_item)
                else: logger.warning(f"SysAnalyzer: Discarding malformed sugg #{s_idx+1}: {str(s_item)[:100]}")
            analysis_result["improvement_suggestions"] = valid_suggs
            return analysis_result
        except Exception as e: logger.error(f"SysAnalyzer: LLM analysis error: {e}", exc_info=True); return {"error": str(e), "improvement_suggestions": []}


class StrategicEvolutionAgent: # pragma: no cover
    """
    Manages strategic self-improvement campaigns for the MindX system.
    """
    def __init__(
        self,
        agent_id: str, 
        belief_system: BeliefSystem,
        coordinator_agent: CoordinatorAgent,
        config_override: Optional[Config] = None,
        test_mode: bool = False
    ):
        if hasattr(self, '_initialized') and self._initialized and not test_mode:
            return

        self.agent_id = agent_id
        self.belief_system = belief_system
        self.coordinator_agent = coordinator_agent # To make tactical SIA requests
        self.config = config_override or Config()
        self.log_prefix = f"SEA ({self.agent_id}):"

        # LLM for this SEA's own high-level reasoning & strategic plan/param generation
        sea_llm_provider = self.config.get(f"strategic_evolution_agent.{self.agent_id}.llm.provider", self.config.get("llm.default_provider"))
        sea_llm_model = self.config.get(f"strategic_evolution_agent.{self.agent_id}.llm.model", self.config.get(f"llm.{sea_llm_provider}.default_model_for_reasoning"))
        self.llm_handler = create_llm_handler(sea_llm_provider, sea_llm_model)
        logger.info(f"{self.log_prefix} Internal LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default'}")
        
        self.system_analyzer = SystemAnalyzerTool(self.belief_system, self.llm_handler, self.config) # Analyzer uses SEA's LLM

        self.goal_manager = GoalManager(agent_id=self.agent_id) # Manages strategic goals for this SEA
        # The action executor for PlanManager will be self._dispatch_strategic_action
        self.plan_manager = PlanManager(agent_id=self.agent_id, action_executor=self._dispatch_strategic_action, config_override=self.config)
        
        self.campaign_history: List[Dict[str, Any]] = self._load_campaign_history()
        
        self._current_campaign_run_id: Optional[str] = None
        self._initialized = True
        logger.info(f"StrategicEvolutionAgent '{self.agent_id}' initialized.")

    def _get_history_file_path(self) -> Path:
        safe_agent_id_stem = re.sub(r'\W+', '_', self.agent_id)
        return PROJECT_ROOT / "data" / f"sea_campaign_history_{safe_agent_id_stem}.json"

    def _load_campaign_history(self) -> List[Dict[str, Any]]:
        history_file = self._get_history_file_path()
        if history_file.exists():
            try:
                with history_file.open("r", encoding="utf-8") as f: history = json.load(f)
                logger.info(f"{self.log_prefix} Loaded {len(history)} campaign records.")
                return history
            except Exception as e: logger.error(f"{self.log_prefix} Error loading campaign history: {e}")
        return []

    def _save_campaign_history(self):
        history_file = self._get_history_file_path()
        try:
            history_file.parent.mkdir(parents=True, exist_ok=True)
            with history_file.open("w", encoding="utf-8") as f: json.dump(self.campaign_history, f, indent=2)
            logger.debug(f"{self.log_prefix} Saved {len(self.campaign_history)} campaign records.")
        except Exception as e: logger.error(f"{self.log_prefix} Error saving campaign history: {e}")

    async def _dispatch_strategic_action(self, action: Action) -> Tuple[bool, Any]:
        """Dispatcher for actions within the SEA's own strategic plans."""
        action_type = action.type.upper() # Ensure uppercase
        logger.info(f"{self.log_prefix} Dispatching strategic action: {action_type} (ID: {action.id}), Params: {action.params}")
        
        handler = getattr(self, f"_sea_action_{action_type.lower()}", None)
        if handler and callable(handler):
            try:
                return await handler(action.params, current_plan_id=self.plan_manager.get_plan(action.id.split('_act')[0])?.id if action.id else None) # Pass plan_id for context
            except Exception as e:
                logger.error(f"{self.log_prefix} Error executing strategic action {action_type} (ID: {action.id}): {e}", exc_info=True)
                return False, f"Exception in handler for {action_type}: {e}"
        else:
            logger.warning(f"{self.log_prefix} No handler found for strategic action type: {action_type}")
            return False, f"Unknown strategic action type: {action_type}"

    # --- Strategic Action Handlers for SEA's PlanManager ---
    async def _sea_action_request_system_analysis(self, params: Dict[str, Any], current_plan_id: Optional[str]) -> Tuple[bool, Any]:
        focus_hint = params.get("focus_hint", f"Improvements relevant to campaign run {self._current_campaign_run_id or 'general'}")
        analysis_result = await self.system_analyzer.analyze_system_for_improvements(analysis_focus_hint=focus_hint)
        suggestions = analysis_result.get("improvement_suggestions", [])
        if analysis_result.get("error") or not suggestions:
            return False, {"message": f"Analysis failed or no suggestions. Error: {analysis_result.get('error')}", "num_suggestions": 0}
        
        # Store suggestions in a belief linked to this SEA and plan/campaign
        belief_key = f"strategic_evolution.{self.agent_id}.plan.{current_plan_id or 'adhoc'}.analysis_suggestions"
        await self.belief_system.add_belief(belief_key, suggestions, 0.9, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600) # Cache for 1hr
        return True, {"num_suggestions": len(suggestions), "suggestions_belief_key": belief_key}

    async def _sea_action_select_improvement_target(self, params: Dict[str, Any], current_plan_id: Optional[str]) -> Tuple[bool, Any]:
        source_belief_key = params.get("suggestions_belief_key", f"strategic_evolution.{self.agent_id}.plan.{current_plan_id or 'adhoc'}.analysis_suggestions")
        num_to_select = params.get("num_to_select", 1)

        suggestions_belief = await self.belief_system.get_belief(source_belief_key)
        suggestions = suggestions_belief.value if suggestions_belief and isinstance(suggestions_belief.value, list) else []
        
        if not suggestions: return False, {"message": f"No suggestions found at {source_belief_key}."}
        
        # Prioritization logic (can be LLM call or rule-based)
        # For now, simple priority sort from suggestions themselves
        suggestions.sort(key=lambda x: x.get("priority", 0), reverse=True)
        selected = suggestions[:num_to_select]
        
        if not selected: return False, {"message": "No suggestions met selection criteria."}

        selected_target_belief_key = f"strategic_evolution.{self.agent_id}.plan.{current_plan_id or 'adhoc'}.selected_targets"
        await self.belief_system.add_belief(selected_target_belief_key, selected, 0.95, BeliefSource.SELF_ANALYSIS)
        logger.info(f"{self.log_prefix} Selected {len(selected)} targets for improvement. Stored in: {selected_target_belief_key}")
        return True, {"selected_targets": selected, "selected_targets_belief_key": selected_target_belief_key}

    async def _sea_action_formulate_sia_task_goal(self, params: Dict[str, Any], current_plan_id: Optional[str]) -> Tuple[bool, Any]:
        """Takes one selected_target and formulates a precise goal for SIA."""
        selected_target_item: Optional[Dict] = params.get("selected_target_item") # This should be one item from the list
        if not selected_target_item or not isinstance(selected_target_item, dict):
            return False, {"message": "Missing or invalid 'selected_target_item' in params."}

        target_path = selected_target_item["target_component_path"]
        original_suggestion = selected_target_item["suggestion"]
        priority = selected_target_item["priority"]
        is_critical = selected_target_item.get("is_critical_target", False)

        # Could use LLM to refine 'original_suggestion' into a more precise SIA goal if needed
        # For now, assume original_suggestion is good enough for SIA's context.
        sia_task_goal = original_suggestion 
        
        formulated_task_details = {
            "target_component_path": target_path,
            "improvement_description_for_sia": sia_task_goal,
            "original_priority": priority,
            "is_critical_target_from_analysis": is_critical
        }
        logger.info(f"{self.log_prefix} Formulated SIA task for '{target_path}': '{sia_task_goal[:80]}...'")
        return True, {"formulated_sia_task_details": formulated_task_details}


    async def _sea_action_request_coordinator_for_sia_execution(self, params: Dict[str, Any], current_plan_id: Optional[str]) -> Tuple[bool, Any]:
        sia_task_details = params.get("formulated_sia_task_details") # Output from previous action
        if not sia_task_details or not isinstance(sia_task_details, dict):
            return False, {"message": "Missing 'formulated_sia_task_details' from previous step."}

        target_path = sia_task_details["target_component_path"]
        improvement_desc_for_sia = sia_task_details["improvement_description_for_sia"]
        
        logger.info(f"{self.log_prefix} Requesting Coordinator to execute SIA for '{target_path}'.")
        interaction_metadata = {
            "target_component": target_path,
            "analysis_context": improvement_desc_for_sia,
            "source": f"sea_agent_{self.agent_id}_plan_{current_plan_id or 'adhoc'}",
            "max_cycles": params.get("max_sia_cycles", self.config.get("self_improvement_agent.default_max_cycles",1)),
            # Pass LLM overrides for SIA if specified in this SEA's plan action params
            "sia_llm_provider": params.get("sia_llm_provider"),
            "sia_llm_model": params.get("sia_llm_model"),
        }
        interaction_content = f"SEA '{self.agent_id}' requests SIA modification for '{target_path}'. Goal: {improvement_desc_for_sia[:100]}"
        
        coordinator_response_dict = await self.coordinator_agent.handle_user_input(
            content=interaction_content, agent_id=self.agent_id,
            interaction_type=InteractionType.COMPONENT_IMPROVEMENT, metadata=interaction_metadata )
        
        # Store Coordinator's response (which includes SIA's full JSON output in `response.data`)
        # This is crucial for the EVALUATE_SIA_OUTCOME action.
        belief_key_for_outcome = f"sea.{self.agent_id}.plan.{current_plan_id or 'adhoc'}.sia_results.{target_path.replace('.','_')}"
        await self.belief_system.add_belief(belief_key_for_outcome, coordinator_response_dict, 0.9, BeliefSource.COMMUNICATION)
        
        sia_success = coordinator_response_dict.get("response", {}).get("status") == "SUCCESS" and \
                      coordinator_response_dict.get("response", {}).get("data",{}).get("final_status","").startswith("SUCCESS")

        return sia_success, {"coordinator_response": coordinator_response_dict, "sia_outcome_belief_key": belief_key_for_outcome}

    async def _sea_action_evaluate_sia_outcome(self, params: Dict[str, Any], current_plan_id: Optional[str]) -> Tuple[bool, Any]:
        sia_outcome_belief_key = params.get("sia_outcome_belief_key")
        if not sia_outcome_belief_key: return False, {"message": "Missing 'sia_outcome_belief_key'."}

        coordinator_response_belief = await self.belief_system.get_belief(sia_outcome_belief_key)
        if not coordinator_response_belief or not isinstance(coordinator_response_belief.value, dict):
            return False, {"message": f"No SIA outcome data at '{sia_outcome_belief_key}'."}
        
        coordinator_response = coordinator_response_belief.value
        sia_full_result = coordinator_response.get("response", {}) # This is the JSON from SIA CLI
        sia_data = sia_full_result.get("data", {}) # This is SIA's internal "data" field
        target_path = sia_data.get("target_file_path_conceptual", params.get("target_component_path", "unknown_target"))
        original_suggestion = sia_data.get("cycle_results", [{}])[-1].get("improvement_description", "N/A") if sia_data.get("cycle_results") else "N/A"
        diff_patch = sia_data.get("cycle_results", [{}])[-1].get("diff_patch", "No diff available.") if sia_data.get("cycle_results") else "No diff."

        logger.info(f"{self.log_prefix} Evaluating SIA outcome for '{target_path}'. Original Suggestion: {original_suggestion[:80]}")
        
        # LLM-based critique of the change (more abstract than SIA's internal eval)
        prompt = (f"Review outcome of an automated code change by MindX's SelfImprovementAgent (SIA).\n"
                  f"Target: {target_path}\nOriginal Goal for SIA: {original_suggestion}\n"
                  f"SIA Reported Status: {sia_data.get('final_status', 'N/A')}\n"
                  f"SIA Message: {sia_full_result.get('message', 'N/A')}\n"
                  f"Diff Patch (if change made):\n```diff\n{diff_patch[:2000]}\n```\n" # Truncate long diffs
                  f"Based on this, provide a strategic assessment: \n"
                  f"1. `overall_assessment`: string - 'Highly Positive', 'Positive', 'Neutral', 'Negative', 'Highly Negative'.\n"
                  f"2. `alignment_with_goal`: float (0.0-1.0) - How well did the change align with original goal for SIA?\n"
                  f"3. `potential_impact`: string - Brief note on likely impact (e.g., 'Improved readability', 'Minor perf gain', 'Risk of regression in Y').\n"
                  f"4. `next_step_suggestion`: string - 'Monitor closely', 'Consider rollback if issues arise', 'Proceed with further related improvements', 'Mark campaign goal as partially met'.\n"
                  f"Respond ONLY in JSON." )
        try:
            eval_str = await self.llm_handler.generate_text(prompt, max_tokens=500, temperature=0.1, json_mode=True)
            eval_result = {} # ... (Robust JSON parsing) ...
            try: eval_result = json.loads(eval_str)
            except json.JSONDecodeError: match = re.search(r"\{[\s\S]*\}", eval_str);
            if match: eval_result = json.loads(match.group(0))
            else: raise ValueError("LLM eval response not JSON.")
            if not isinstance(eval_result, dict) or not all(k in eval_result for k in ["overall_assessment", "alignment_with_goal"]): raise ValueError("LLM eval missing keys.")
            
            await self.belief_system.add_belief(f"sea.{self.agent_id}.evaluation.{target_path.replace('.','_')}", eval_result, 0.8, BeliefSource.SELF_ANALYSIS)
            return True, eval_result
        except Exception as e: return False, {"message": f"LLM evaluation failed: {e}"}

    async def _sea_action_update_campaign_goal_status(self, params: Dict[str, Any], current_plan_id: Optional[str]) -> Tuple[bool, Any]: # pragma: no cover
        """BDI Action: Assesses and updates the status of the current campaign goal."""
        campaign_goal_id = self.intentions.get("current_goal_id_for_plan") # Assuming current plan is for campaign goal
        if not campaign_goal_id: return False, {"message": "No campaign goal ID associated with current plan."}

        # Logic to determine if campaign_goal_id is met (e.g., based on evaluations of SIA changes, etc.)
        # This is complex and would involve querying beliefs about multiple SIA outcomes.
        # For this stub, assume it's met if this action is called after successful steps.
        # In a real system, this action would receive inputs (e.g. list of evaluation belief keys)
        # and use LLM to make a final judgement on overall campaign goal.
        
        new_status_str = params.get("new_status", "COMPLETED_SUCCESS").upper() # e.g., plan passes "COMPLETED_SUCCESS"
        try: new_status = GoalSt[new_status_str]
        except KeyError: return False, {"message": f"Invalid GoalSt value '{new_status_str}' provided."}

        goal_updated = False
        for g_entry in self.desires["priority_queue"]:
            if g_entry["id"] == campaign_goal_id:
                g_entry["status"] = new_status.value # Update status of the goal in the BDI desires queue
                goal_updated = True; break
        
        if goal_updated:
            logger.info(f"{self.log_prefix} Campaign goal ID '{campaign_goal_id}' status updated to {new_status.name}.")
            if new_status == GoalSt.COMPLETED_SUCCESS and campaign_goal_id == self.desires.get("primary_goal_id"):
                self._internal_state["status"] = "COMPLETED_GOAL_ACHIEVED" # Mark BDI agent itself as completed its primary
            return True, {"message": f"Campaign goal {campaign_goal_id} status set to {new_status.name}."}
        else: # pragma: no cover
            return False, {"message": f"Campaign goal ID '{campaign_goal_id}' not found in desires queue to update status."}


    # --- Main Campaign Management Method ---
    async def manage_improvement_campaign(self, campaign_goal_description: str, max_bdi_run_cycles: int = 20) -> Dict[str, Any]: # pragma: no cover
        """
        Manages a self-improvement campaign for a given high-level goal.
        This is the main entry point for this agent when called by the Coordinator.
        """
        self._current_campaign_run_id = str(uuid.uuid4())[:8]
        logger.info(f"{self.log_prefix} Starting new improvement campaign (Run ID: {self._current_campaign_run_id}). Goal: '{campaign_goal_description}'")
        
        await self.update_belief(f"campaign.{self._current_campaign_run_id}.objective", campaign_goal_description, 0.95, BeliefSource.COMMUNICATION)

        # Set the campaign goal for the internal BDI agent
        # This BDI run will attempt to achieve this single strategic goal.
        campaign_goal_id = str(uuid.uuid4())[:8]
        self.bdi_agent.set_goal(
            goal_description=f"Successfully execute improvement campaign: {campaign_goal_description}",
            priority=1, is_primary=True, goal_id=campaign_goal_id
        )
        
        # Run the BDI agent. Its plan will involve calling this SEA's _sea_action_... methods.
        bdi_final_outcome_message = await self.bdi_agent.run(max_cycles=max_bdi_run_cycles)
        
        bdi_final_status = self.bdi_agent._internal_state["status"] # Get BDI's final status
        campaign_result_status = "SUCCESS" if bdi_final_status == "COMPLETED_GOAL_ACHIEVED" else "FAILURE"

        campaign_summary = {
            "campaign_run_id": self._current_campaign_run_id, "agent_id": self.agent_id,
            "campaign_goal": campaign_goal_description, "bdi_outcome_message": bdi_final_outcome_message,
            "bdi_final_status": bdi_final_status, "overall_campaign_status": campaign_result_status,
            # Optionally, gather key results from beliefs populated by BDI actions
            "final_report_snippet": await self.bdi_agent.get_belief(f"action_results.GENERATE_REPORT.{self.bdi_agent.domain.replace(' ','_')[:30]}") or "No final report generated by BDI."
        }
        
        self.campaign_history.append({ "timestamp": time.time(), **campaign_summary }); self._save_campaign_history()
        await self.update_belief(f"campaign.{self._current_campaign_run_id}.outcome", campaign_summary, 0.9, BeliefSource.SELF_ANALYSIS)

        logger.info(f"{self.log_prefix} Improvement campaign (Run ID: {self._current_campaign_run_id}) finished. BDI Status: {bdi_final_status}. Overall: {campaign_result_status}")
        return campaign_summary
    
    async def shutdown(self): # pragma: no cover
        logger.info(f"StrategicEvolutionAgent '{self.agent_id}' shutting down...")
        if self.bdi_agent: await self.bdi_agent.shutdown()
        self._save_campaign_history()
        logger.info(f"StrategicEvolutionAgent '{self.agent_id}' shutdown complete.")
