# mindx/orchestration/mastermind_agent.py
"""
MastermindAgent for MindX: The Apex Orchestrator and Evolutionary Director.

This agent sits at the highest level of the MindX operational hierarchy.
It is responsible for:
- Setting long-term, overarching evolutionary goals for the MindX system.
- Monitoring the overall health and progress of MindX via the CoordinatorAgent.
- Initiating strategic improvement or development campaigns by tasking the
  StrategicEvolutionAgent (SEA) (conceptually, or directly tasking Coordinator
  with high-level analysis that SEA or Coordinator's loop would pick up).
- Managing secure identities for newly conceptualized agents or tools via an
  IDManagerAgent.
- Operating autonomously based on its own BDI-driven reasoning to guide the
  meta-evolution of the MindX system.
"""

import os
import logging
import asyncio
import json
import time
import uuid
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from mindx.core.belief_system import BeliefSystem, BeliefSource
from mindx.llm.llm_factory import create_llm_handler, LLMHandler
# Mastermind uses its own BDI agent for its strategic execution loop
from mindx.core.bdi_agent import BDIAgent, BaseTool as BDIBaseTool, Goal as BDIGoal, GoalSt as BDIGoalStatus
# Mastermind uses IDManagerAgent for identity provisioning
from mindx.core.id_manager_agent import IDManagerAgent
# Mastermind interacts with the CoordinatorAgent
from mindx.orchestration.coordinator_agent import CoordinatorAgent, InteractionType, InteractionStatus, Interaction

logger = get_logger(__name__)

class MastermindAgent:
    """
    The MastermindAgent: Oversees the evolution and high-level strategy of the MindX system.
    It can initiate the creation of new capabilities or agents by tasking subordinate systems.
    """
    _instance = None
    _lock = asyncio.Lock() # Class-level lock for async-safe singleton factory

    @classmethod
    async def get_instance(cls, # pragma: no cover
                           agent_id: Optional[str] = None, 
                           config_override: Optional[Config] = None,
                           belief_system_override: Optional[BeliefSystem] = None,
                           coordinator_agent_override: Optional[CoordinatorAgent] = None,
                           test_mode: bool = False) -> 'MastermindAgent':
        """Factory method to get or create the singleton MastermindAgent instance."""
        async with cls._lock:
            # agent_id defaults here if not provided, using config
            temp_config = config_override or Config(test_mode=test_mode) # Ensure config is available for default ID
            effective_agent_id = agent_id or temp_config.get("mastermind_agent.default_agent_id", "mastermind_ overseeing_prime")


            if cls._instance is None or test_mode or cls._instance.agent_id != effective_agent_id:
                if test_mode and cls._instance is not None and cls._instance.agent_id == effective_agent_id:
                    logger.debug(f"MastermindAgent Factory: Resetting existing test instance for '{effective_agent_id}'.")
                    await cls._instance.shutdown() 
                    cls._instance = None 
                elif test_mode and cls._instance is not None and cls._instance.agent_id != effective_agent_id: # pragma: no cover
                    logger.debug(f"MastermindAgent Factory: Test mode switching agent ID from '{cls._instance.agent_id}' to '{effective_agent_id}'. Shutting down old.")
                    await cls._instance.shutdown()
                    cls._instance = None


                logger.info(f"MastermindAgent Factory: Creating new instance for ID '{effective_agent_id}'.")
                # Dependencies must be ready for Mastermind's __init__
                cfg = config_override or Config(test_mode=test_mode) # Ensure config respects test_mode again if it was reset
                bs = belief_system_override or BeliefSystem(test_mode=test_mode)
                
                coord = coordinator_agent_override
                if not coord and not test_mode: # pragma: no cover 
                    from mindx.orchestration.coordinator_agent import get_coordinator_agent_mindx_async
                    logger.warning(f"MastermindAgent Factory: CoordinatorAgent not provided for '{effective_agent_id}', getting/creating default instance.")
                    coord = await get_coordinator_agent_mindx_async(config_override=cfg, test_mode=test_mode)
                elif not coord and test_mode: 
                    logger.warning(f"MastermindAgent Factory (test_mode): CoordinatorAgent not provided for '{effective_agent_id}'. Operations requiring it may fail if not mocked externally.")

                if not coord: # Still no coordinator, this is a problem unless specifically in a test that mocks it out entirely
                     logger.error(f"MastermindAgent Factory: CRITICAL - CoordinatorAgent could not be obtained for '{effective_agent_id}'. Mastermind will be impaired.")
                     # Depending on strictness, could raise an error here.

                instance = cls(
                    agent_id=effective_agent_id, 
                    belief_system_instance=bs,
                    coordinator_agent_instance=coord, # This can be None if test_mode and not provided
                    config_override=cfg,
                    _is_factory_called=True,
                    test_mode=test_mode
                )
                cls._instance = instance
            elif cls._instance.agent_id != effective_agent_id: # pragma: no cover 
                logger.error(f"MastermindAgent Factory: ID mismatch. Requested '{effective_agent_id}' but singleton is '{cls._instance.agent_id}'. Not in test_mode. Returning existing.")

            return cls._instance

    def __init__(self, 
                 agent_id: str, 
                 belief_system_instance: BeliefSystem,
                 coordinator_agent_instance: Optional[CoordinatorAgent], # Make explicitly optional for robust init
                 config_override: Optional[Config] = None,
                 _is_factory_called: bool = False,
                 test_mode: bool = False):
        
        if not _is_factory_called and (MastermindAgent._instance is None or MastermindAgent._instance is not self) : # pragma: no cover
             logger.warning(f"MastermindAgent direct instantiation for '{agent_id}'. Prefer using `await MastermindAgent.get_instance(...)` for singleton management.")

        if hasattr(self, '_initialized') and self._initialized and not test_mode: # pragma: no cover
            return

        self.agent_id = agent_id
        self.config = config_override or Config() # Ensures Config is initialized
        self.belief_system = belief_system_instance
        self.coordinator_agent = coordinator_agent_instance 
        self.log_prefix = f"Mastermind ({self.agent_id}):"

        # Define dedicated data directory for this Mastermind instance
        data_dir_relative_path = self.config.get(f"mastermind_agent.{self.agent_id}.data_dir_relative_to_project", 
                                            f"data/mastermind_work/{self.agent_id.replace(':', '_').replace(' ','_')}")
        self.data_dir: Path = PROJECT_ROOT / data_dir_relative_path
        self._ensure_data_dir()

        # Mastermind's own LLM for high-level strategic thought / goal formulation
        mastermind_llm_config_key_prefix = f"mastermind_agent.{self.agent_id}.llm"
        llm_provider_cfg = self.config.get(f"{mastermind_llm_config_key_prefix}.provider", self.config.get("llm.default_provider"))
        llm_model_cfg = self.config.get(f"{mastermind_llm_config_key_prefix}.model", 
                                       self.config.get(f"llm.{llm_provider_cfg}.default_model_for_strategy", 
                                                       self.config.get(f"llm.{llm_provider_cfg}.default_model")))
        self.llm_handler: LLMHandler = create_llm_handler(llm_provider_cfg, llm_model_cfg)
        logger.info(f"{self.log_prefix} Internal LLM set to: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default_for_provider'}")

        # Mastermind uses a BDI agent for its own strategic execution loop
        self.bdi_agent = BDIAgent(
            domain=f"mastermind_strategy_{self.agent_id.replace(':','_').replace(' ','_')}", # Sanitize domain
            belief_system_instance=self.belief_system, 
            config_override=self.config,
            test_mode=test_mode # Propagate test_mode
        )
        self._register_mastermind_bdi_actions()

        # IDManager for generating identities for new agents/tools MindX might create
        self.id_manager_agent: Optional[IDManagerAgent] = None
        if self.config.get(f"mastermind_agent.{self.agent_id}.enable_id_management", True): # Default True
            try:
                id_manager_instance_id = self.config.get(f"mastermind_agent.{self.agent_id}.id_manager_instance_id", 
                                                        f"id_manager_for_{self.agent_id}")
                # IDManagerAgent.get_instance is sync in current design.
                self.id_manager_agent = IDManagerAgent.get_instance(
                    agent_id=id_manager_instance_id, 
                    config_override=self.config,
                    test_mode=test_mode 
                )
            except Exception as e_id_mgr: # pragma: no cover
                logger.error(f"{self.log_prefix} Failed to initialize IDManagerAgent: {e_id_mgr}. Identity functionalities will be impaired.", exc_info=True)
        
        self.strategic_campaigns_history: List[Dict[str,Any]] = self._load_json_file("mastermind_campaigns_history.json", [])
        self.high_level_objectives: List[Dict[str,Any]] = self._load_json_file("mastermind_objectives.json", []) # Persisted objectives

        self.autonomous_loop_task: Optional[asyncio.Task] = None
        if self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.enabled", False) and not test_mode: # pragma: no cover
            self.start_autonomous_loop()

        logger.info(f"{self.log_prefix} Initialized. Data dir: {self.data_dir}. Autonomous Loop: {bool(self.autonomous_loop_task)}. ID Manager: {'Active' if self.id_manager_agent else 'Inactive'}")
        self._initialized = True

    def _ensure_data_dir(self): # pragma: no cover
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt': os.chmod(self.data_dir, stat.S_IRWXU) # rwx for owner
        except Exception as e: logger.error(f"{self.log_prefix} Failed to create or set permissions for data dir {self.data_dir}: {e}")

    def _load_json_file(self, file_name: str, default_value: Union[List, Dict]) -> Union[List, Dict]: # pragma: no cover
        file_path = self.data_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                with file_path.open("r", encoding="utf-8") as f: return json.load(f)
            except Exception as e: logger.error(f"{self.log_prefix} Error loading {file_name} from {file_path}: {e}")
        return copy.deepcopy(default_value) 

    def _save_json_file(self, file_name: str, data: Union[List, Dict]): # pragma: no cover
        file_path = self.data_dir / file_name
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure data_dir exists
            with file_path.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            logger.debug(f"{self.log_prefix} Saved data to {file_path}")
        except Exception as e: logger.error(f"{self.log_prefix} Error saving {file_name} to {file_path}: {e}")

    def _register_mastermind_bdi_actions(self): # pragma: no cover
        """Registers handlers for actions Mastermind's internal BDI agent can take."""
        self.bdi_agent._action_handlers["OBSERVE_MINDX_SYSTEM_STATE"] = self._bdi_action_observe_mindx_state
        self.bdi_agent._action_handlers["FORMULATE_STRATEGIC_CAMPAIGN_GOAL"] = self._bdi_action_formulate_campaign_goal
        self.bdi_agent._action_handlers["LAUNCH_IMPROVEMENT_CAMPAIGN_VIA_COORDINATOR"] = self._bdi_action_launch_improvement_campaign
        self.bdi_agent._action_handlers["REQUEST_NEW_ENTITY_IDENTITY"] = self._bdi_action_request_new_entity_identity
        self.bdi_agent._action_handlers["INITIATE_NEW_COMPONENT_DEVELOPMENT_CAMPAIGN"] = self._bdi_action_initiate_new_component_dev
        self.bdi_agent._action_handlers["REVIEW_CAMPAIGN_OUTCOMES"] = self._bdi_action_review_campaign_outcomes
        logger.info(f"{self.log_prefix} Registered BDI action handlers for Mastermind's strategic operations.")

    # --- Mastermind's BDI Action Handlers ---
    async def _bdi_action_observe_mindx_state(self, action_params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        logger.info(f"{self.log_prefix} BDI Action: Observing MindX system state via Coordinator.")
        if not self.coordinator_agent: return False, {"message": "CoordinatorAgent not available to Mastermind for state observation."}
        
        focus = action_params.get("analysis_focus", "current system health, active alerts, and top 2-3 pending high-priority improvement needs from Coordinator's backlog")
        
        interaction = await self.coordinator_agent.create_interaction(
            interaction_type=InteractionType.SYSTEM_ANALYSIS,
            content=f"Mastermind request: Provide system state summary focused on: {focus}",
            agent_id=self.agent_id, 
            metadata={"source": self.agent_id, "analysis_depth": "summary_for_mastermind"}
        )
        processed_interaction = await self.coordinator_agent.process_interaction(interaction)

        if processed_interaction.status == InteractionStatus.COMPLETED and isinstance(processed_interaction.response, dict):
            state_summary = processed_interaction.response
            state_summary["coordinator_backlog_size"] = len(self.coordinator_agent.improvement_backlog) # Add current backlog size
            state_summary["coordinator_pending_approval_count"] = len([item for item in self.coordinator_agent.improvement_backlog if item.get("status") == InteractionStatus.PENDING_APPROVAL.value])

            await self.bdi_agent.update_belief("mindx_state_summary_from_coordinator", state_summary, 0.85, BeliefSource.COMMUNICATION)
            return True, {"summary_retrieved": True, "state_summary_preview": str(state_summary)[:200]}
        else: 
            err_msg = f"Failed to get MindX state summary from Coordinator. Error: {processed_interaction.error or 'Unknown'}"
            logger.error(f"{self.log_prefix} {err_msg}")
            return False, {"message": err_msg}

    async def _bdi_action_formulate_campaign_goal(self, action_params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        high_level_directive = action_params.get("directive", await self.bdi_agent.get_belief("current_mastermind_directive") or "Proactively evolve and improve the MindX system's overall effectiveness and capabilities.")
        mindx_state_summary_val = await self.bdi_agent.get_belief("mindx_state_summary_from_coordinator")
        mindx_state_summary_str = json.dumps(mindx_state_summary_val, indent=2)[:2000] if mindx_state_summary_val else "No detailed state summary available."
        
        existing_objectives = [obj.get("goal_description") for obj in self.high_level_objectives if obj.get("status") == "active"]
        existing_objectives_str = ("Existing active Mastermind objectives:\n" + "\n".join([f"- {o}" for o in existing_objectives[:3]])) if existing_objectives else "No other active Mastermind objectives."


        prompt = (
            f"You are the strategic LLM for the MastermindAgent of MindX. Your current high-level directive is: '{high_level_directive}'.\n"
            f"Current MindX State Summary (from Coordinator):\n{mindx_state_summary_str}\n"
            f"{existing_object_str}\n\n"
            f"Based on the directive and current state, and avoiding duplication with existing objectives, formulate a single, specific, and actionable STRATEGIC CAMPAIGN GOAL for the MindX system. "
            f"This goal should be achievable through a series of tactical code improvements, analyses, or new component developments over multiple cycles. "
            f"It should be a clear, concise string representing a desirable future state or capability. "
            f"Example: 'Enhance the SelfImprovementAgent's evaluation metrics to include automated security vulnerability scanning for generated code.'\n"
            f"Respond ONLY with a JSON object: {{\"formulated_campaign_goal\": \"Your strategic goal description.\"}}"
        )
        try:
            response_str = await self.llm_handler.generate_text(prompt, max_tokens=self.config.get(f"mastermind_agent.{self.agent_id}.llm.max_tokens_goal_formulation", 400), temperature=0.4, json_mode=True)
            parsed_response = {}; 
            try: parsed_response = json.loads(response_str)
            except json.JSONDecodeError: match = re.search(r"(\{[\s\S]*?\})", response_str, re.DOTALL); # Find first JSON object
            if match: parsed_response = json.loads(match.group(1))
            else: raise ValueError("LLM response for campaign goal not JSON.")
            
            campaign_goal_desc = parsed_response.get("formulated_campaign_goal")
            if campaign_goal_desc and isinstance(campaign_goal_desc, str):
                await self.bdi_agent.update_belief("current_formulated_campaign_goal_for_bdi", campaign_goal_desc) # For BDI internal use
                return True, {"formulated_campaign_goal_description": campaign_goal_desc} # Return for plan param
            return False, {"message": "LLM failed to formulate a valid campaign goal string."} # pragma: no cover
        except Exception as e: return False, {"message": f"Error formulating campaign goal via LLM: {type(e).__name__}: {e}"} # pragma: no cover

    async def _bdi_action_launch_improvement_campaign(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        """This action tells the Coordinator to initiate a broad improvement effort based on the campaign_goal_description."""
        campaign_goal_desc = params.get("campaign_goal_description") # This should be the output of FORMULATE_STRATEGIC_CAMPAIGN_GOAL
        if not campaign_goal_desc: campaign_goal_desc = await self.bdi_agent.get_belief("current_formulated_campaign_goal_for_bdi")
        if not campaign_goal_desc: return False, {"message": "No campaign goal description available to launch."}
        if not self.coordinator_agent: return False, {"message":"Coordinator agent not available."} # pragma: no cover

        logger.info(f"{self.log_prefix} BDI Action: Requesting Coordinator to handle strategic campaign: '{campaign_goal_desc}'")
        
        # The Mastermind gives a high-level strategic goal. The Coordinator's SYSTEM_ANALYSIS
        # will be focused by this goal, and its autonomous loop will pick up resulting suggestions.
        # A more advanced Coordinator might have a dedicated "START_CAMPAIGN" InteractionType.
        interaction_meta = {
            "source": self.agent_id, 
            "mastermind_campaign_goal": campaign_goal_desc, # For Coordinator's context
            "analysis_context": f"System-wide analysis focused on achieving Mastermind strategic goal: {campaign_goal_desc}",
            "analysis_depth": "strategic_campaign_kickoff"
        }
        
        interaction = await self.coordinator_agent.create_interaction(
            interaction_type=InteractionType.SYSTEM_ANALYSIS, # Use SYSTEM_ANALYSIS to populate Coord's backlog
            content=f"Mastermind directive: Initiate activities for campaign goal: {campaign_goal_desc}",
            agent_id=self.agent_id, metadata=interaction_meta
        )
        processed_interaction = await self.coordinator_agent.process_interaction(interaction)

        if processed_interaction.status == InteractionStatus.COMPLETED:
            mastermind_run_id = self._internal_state.get("current_run_id", "unknown_run")
            mastermind_campaign_id = f"mstr_camp_{mastermind_run_id}_{str(uuid.uuid4())[:6]}"
            
            await self.bdi_agent.update_belief(f"mastermind_campaigns.{mastermind_campaign_id}.goal", campaign_goal_desc)
            await self.bdi_agent.update_belief(f"mastermind_campaigns.{mastermind_campaign_id}.status", "delegated_to_coordinator_analysis")
            await self.bdi_agent.update_belief(f"mastermind_campaigns.{mastermind_campaign_id}.coordinator_interaction_id", interaction.interaction_id)
            
            logger.info(f"{self.log_prefix} Strategic campaign '{campaign_goal_desc}' (MCID: {mastermind_campaign_id}) analysis phase delegated to Coordinator via interaction {interaction.interaction_id}.")
            return True, {"mastermind_campaign_id": mastermind_campaign_id, "message": "Campaign analysis phase delegated to Coordinator."}
        else: # pragma: no cover
            err_msg = f"Failed to delegate campaign analysis to Coordinator. Error: {processed_interaction.error}"
            logger.error(f"{self.log_prefix} {err_msg}")
            return False, {"message": err_msg}

    async def _bdi_action_request_new_entity_identity(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        if not self.id_manager_agent: return False, {"message": "IDManagerAgent not available to Mastermind."}
        
        entity_tag = params.get("entity_tag", f"mindx_entity_{str(uuid.uuid4())[:6]}")
        entity_description = params.get("entity_description", "A new component or agent within MindX.")
        logger.info(f"{self.log_prefix} BDI Action: Requesting new identity from IDManager for entity tagged '{entity_tag}'.")
        try:
            public_address, _ = self.id_manager_agent.create_new_wallet(entity_id=entity_tag)
            # Store this crucial info in Mastermind's BDI beliefs, namespaced for the entity
            belief_key_base = f"managed_entities.{entity_tag}"
            await self.bdi_agent.update_belief(f"{belief_key_base}.public_address", public_address)
            await self.bdi_agent.update_belief(f"{belief_key_base}.description", entity_description)
            await self.bdi_agent.update_belief(f"{belief_key_base}.status", "identity_provisioned")
            logger.info(f"{self.log_prefix} New identity created for '{entity_tag}': Address {public_address}")
            return True, {"entity_tag": entity_tag, "public_address": public_address}
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to create new entity identity via IDManagerAgent: {e}", exc_info=True)
            return False, {"message": f"Identity creation failed: {type(e).__name__}: {e}"}

    async def _bdi_action_initiate_new_component_dev(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        """Conceptual: Task Coordinator/SEA to develop a new agent/component based on description and identity."""
        new_comp_desc = params.get("description", "A new utility agent for specialized data parsing.")
        new_comp_identity_address = params.get("identity_public_address") # Should be output of previous action
        target_module_path_sugg = params.get("target_module_path", f"mindx.new_agents.{new_comp_desc.lower().replace(' ','_')[:20]}")

        logger.info(f"{self.log_prefix} BDI Action: Initiating development campaign for new component: {new_comp_desc}. Identity: {new_comp_identity_address or 'None provided'}. Target Path Suggestion: {target_module_path_sugg}")
        
        # This translates to a high-level strategic campaign goal for the Coordinator.
        campaign_goal_for_new_dev = (
            f"Spearhead the development and integration of a new MindX component. "
            f"Description: '{new_comp_desc}'. "
            f"Suggested initial module path: '{target_module_path_sugg}'. "
            f"{('Assigned public identity for this component: ' + new_comp_identity_address) if new_comp_identity_address else 'An identity may need to be provisioned if relevant.'} "
            f"This campaign will likely involve: detailed specification, code generation by SIA over multiple iterations for the initial file(s), "
            f"creation of basic unit tests, and initial registration with the system if applicable."
        )
        # Use the existing launch campaign BDI action to delegate this to Coordinator
        return await self._bdi_action_launch_improvement_campaign({"campaign_goal_description": campaign_goal_for_new_dev})


    async def _bdi_action_review_campaign_outcomes(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        # (Full logic from previous Mastermind, ensuring it uses its own LLM)
        campaign_id_to_review = params.get("campaign_id") or await self.bdi_agent.get_belief("last_launched_mastermind_campaign_id")
        if not campaign_id_to_review: return False, {"message": "No campaign_id provided for review."}
        logger.info(f"{self.log_prefix} BDI Action: Reviewing outcomes for Mastermind campaign ID '{campaign_id_to_review}'.")
        campaign_data_list = [c for c in self.strategic_campaigns_history if c.get("mastermind_run_id") == campaign_id_to_review or c.get("mastermind_campaign_id") == campaign_id_to_review]
        if not campaign_data_list: return False, {"message": f"No data for Mastermind campaign ID '{campaign_id_to_review}'."}
        campaign_data = campaign_data_list[0] 
        original_goal = campaign_data.get("directive", campaign_data.get("campaign_goal", "N/A"))
        bdi_outcome_msg = campaign_data.get("bdi_outcome_message", "N/A"); bdi_final_status = campaign_data.get("bdi_final_status", "N/A")
        # In future, this action could query Coordinator for more detailed tactical outcomes linked to this campaign.
        prompt = (f"Review MindX improvement campaign outcome.\nOriginal Goal: {original_goal}\n"
                  f"Internal BDI Status: {bdi_final_status}\nBDI Outcome Msg: {bdi_outcome_msg}\n"
                  f"Assess if campaign broadly met strategic objective. Respond ONLY JSON: "
                  f"{{\"campaign_objective_met\": bool, \"assessment_summary\": \"brief overall assessment\", \"suggested_follow_up\": \"string (e.g., 'Monitor', 'New campaign for X', 'Mark achieved')\"}}")
        try:
            response_str = await self.llm_handler.generate_text(prompt, max_tokens=400, temperature=0.1, json_mode=True)
            parsed_assessment = {}; # Placeholder for robust JSON parsing
            try: parsed_assessment = json.loads(response_str)
            except: match = re.search(r"(\{[\s\S]*?\})", response_str, re.DOTALL);
            if match: parsed_assessment = json.loads(match.group(1))
            else: raise ValueError("LLM review response not JSON.")
            if not isinstance(parsed_assessment, dict) or not all(k in parsed_assessment for k in ["campaign_objective_met", "assessment_summary", "suggested_follow_up"]): raise ValueError("LLM review missing keys.")
            await self.bdi_agent.update_belief(f"mastermind_campaign_review.{campaign_id_to_review}", parsed_assessment)
            return True, parsed_assessment
        except Exception as e: return False, {"message": f"LLM campaign review failed: {type(e).__name__}: {e}"}


    # --- Main Mastermind Orchestration Method & Autonomous Loop ---
    async def manage_mindx_evolution(self, top_level_directive: str, max_mastermind_bdi_cycles: int = 15) -> Dict[str, Any]: # pragma: no cover
        """Primary entry point for Mastermind to act on a high-level directive using its internal BDI agent."""
        self._internal_state["current_run_id"] = f"mastermind_run_{str(uuid.uuid4())[:8]}"
        run_id = self._internal_state["current_run_id"]
        logger.info(f"{self.log_prefix} Starting MindX evolution campaign (Run ID: {run_id}). Directive: '{top_level_directive}'")
        await self.update_belief(f"mastermind.current_campaign.directive", top_level_directive, 0.99, BeliefSource.EXTERNAL_INPUT, is_internal_state=True)
        await self.update_belief(f"mastermind.current_campaign.run_id", run_id, 0.99, BeliefSource.SELF_ANALYSIS, is_internal_state=True)

        # Add this directive as a high-level objective if not already present and active
        existing_objective = next((obj for obj in self.high_level_objectives if obj.get("directive") == top_level_directive and obj.get("status") == "active"), None)
        if not existing_objective:
            objective_id = str(uuid.uuid4())[:8]
            self.high_level_objectives.append({"id": objective_id, "directive": top_level_directive, "status": "active", "created_at": time.time(), "run_ids": [run_id]})
            self._save_json_file("mastermind_objectives.json", self.high_level_objectives)
        elif run_id not in existing_objective.get("run_ids",[]): # pragma: no cover
            existing_objective.setdefault("run_ids",[]).append(run_id) # Add current run to existing active objective
            self._save_json_file("mastermind_objectives.json", self.high_level_objectives)


        self.bdi_agent.set_goal( goal_description=f"Strategically fulfill Mastermind directive: {top_level_directive}", priority=1, is_primary=True, goal_id=f"mastermind_directive_{run_id}" )
        
        bdi_final_run_message = await self.bdi_agent.run(max_cycles=max_mastermind_bdi_cycles)
        bdi_final_agent_status = self.bdi_agent._internal_state["status"]
        campaign_successful = (bdi_final_agent_status == "COMPLETED_GOAL_ACHIEVED")

        campaign_outcome_summary = {
            "mastermind_run_id": run_id, "agent_id": self.agent_id, "directive": top_level_directive,
            "bdi_final_status": bdi_final_agent_status, "bdi_outcome_message": bdi_final_run_message,
            "overall_campaign_status": "SUCCESS" if campaign_successful else "FAILURE_OR_INCOMPLETE",
            "timestamp": time.time() }
        self.strategic_campaigns_history.append(campaign_outcome_summary)
        self._save_json_file("mastermind_campaigns_history.json", self.strategic_campaigns_history)
        
        # Update objective status
        for obj in self.high_level_objectives:
            if obj.get("directive") == top_level_directive and obj.get("status") == "active":
                obj["status"] = "completed_attempt" if campaign_successful else "failed_attempt"
                obj["last_run_id"] = run_id
                obj["last_run_status"] = campaign_outcome_summary['overall_campaign_status']
                break
        self._save_json_file("mastermind_objectives.json", self.high_level_objectives)

        await self.update_belief(f"mastermind.campaign_history.{run_id}", campaign_outcome_summary, 0.95, BeliefSource.SELF_ANALYSIS)
        logger.info(f"{self.log_prefix} MindX evolution campaign (Run ID: {run_id}) finished. BDI Status: {bdi_final_agent_status}. Overall Campaign: {campaign_outcome_summary['overall_campaign_status']}")
        return campaign_outcome_summary

    def start_autonomous_loop(self, interval_seconds: Optional[float] = None): # pragma: no cover
        if self.autonomous_loop_task and not self.autonomous_loop_task.done(): logger.warning(f"{self.log_prefix} Mastermind autonomous loop already running."); return
        loop_interval = interval_seconds or self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.interval_seconds", 3600 * 4)
        default_directive_config_key = f"mastermind_agent.{self.agent_id}.autonomous_loop.default_directive"
        default_directive = self.config.get(default_directive_config_key, "Proactively monitor MindX, identify strategic evolutionary opportunities, and initiate campaigns to enhance overall system health, capabilities, and efficiency based on current state and long-term goals.")
        max_bdi_cycles = self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.max_bdi_cycles", 25)
        if loop_interval <= 0: logger.error(f"{self.log_prefix} Invalid autonomous interval: {loop_interval}. Not starting."); return
        self.autonomous_loop_task = asyncio.create_task(self._mastermind_autonomous_worker(loop_interval, default_directive, max_bdi_cycles))
        logger.info(f"{self.log_prefix} Mastermind autonomous loop started. Interval: {loop_interval}s.")

    async def _mastermind_autonomous_worker(self, interval: float, default_directive: str, max_bdi_cycles: int): # pragma: no cover
        logger.info(f"{self.log_prefix} Mastermind autonomous worker started with directive: '{default_directive[:100]}...'")
        while True:
            try:
                await asyncio.sleep(interval)
                logger.info(f"{self.log_prefix} Autonomous worker: Initiating new strategic campaign with default directive.")
                await self.manage_mindx_evolution(default_directive, max_mastermind_bdi_cycles=max_bdi_cycles)
            except asyncio.CancelledError: logger.info(f"{self.log_prefix} Mastermind autonomous worker stopping."); break
            except Exception as e: logger.error(f"{self.log_prefix} Mastermind autonomous worker error: {e}", exc_info=True); await asyncio.sleep(interval) # Wait before retry
        logger.info(f"{self.log_prefix} Mastermind autonomous worker has stopped.")

    async def shutdown(self): # pragma: no cover
        logger.info(f"MastermindAgent '{self.agent_id}' shutting down...")
        if self.autonomous_loop_task and not self.autonomous_loop_task.done():
            self.autonomous_loop_task.cancel()
            try: await asyncio.wait_for(self.autonomous_loop_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError): pass
        if self.bdi_agent: await self.bdi_agent.shutdown()
        if self.id_manager_agent and hasattr(self.id_manager_agent, 'shutdown'):
            shutdown_method = self.id_manager_agent.shutdown
            if asyncio.iscoroutinefunction(shutdown_method): await shutdown_method()
            elif callable(shutdown_method): shutdown_method() # type: ignore
        self._save_json_file("mastermind_campaigns_history.json", self.strategic_campaigns_history)
        self._save_json_file("mastermind_objectives.json", self.high_level_objectives)
        logger.info(f"MastermindAgent '{self.agent_id}' shutdown complete.")

    @classmethod
    async def reset_all_instances_for_testing(cls): # pragma: no cover
        """Resets all cached Mastermind instances. Primarily for testing."""
        async with cls._lock:
            for instance in list(cls._instances.values()): await instance.shutdown()
            cls._instances.clear()
        logger.debug("All MastermindAgent instances reset.")
