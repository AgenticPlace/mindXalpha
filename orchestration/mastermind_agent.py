# mindx/orchestration/mastermind_agent.py
"""
MastermindAgent for MindX: The Apex Orchestrator and Evolutionary Director.

This agent sits at the highest level of the MindX operational hierarchy.
It is responsible for:
- Setting long-term, overarching evolutionary goals for the MindX system.
- Monitoring the overall health and progress of MindX via the CoordinatorAgent.
- Initiating strategic improvement or development campaigns by tasking the
  StrategicEvolutionAgent (SEA) (typically through the Coordinator).
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
from mindx.core.bdi_agent import BDIAgent # Mastermind uses a BDI agent for its own strategic execution
from mindx.core.id_manager_agent import IDManagerAgent # For generating secure identities
# Import CoordinatorAgent to interact with it
from mindx.orchestration.coordinator_agent import CoordinatorAgent, InteractionType, InteractionStatus 
# Import StrategicEvolutionAgent for type hinting if directly referenced, though usually via Coordinator
from mindx.learning.strategic_evolution_agent import StrategicEvolutionAgent

logger = get_logger(__name__)

# Conceptual Base class for tools (if Mastermind had its own distinct tools)
class MastermindBaseTool: # pragma: no cover
    def __init__(self, mastermind_ref: 'MastermindAgent'): self.mastermind = mastermind_ref
    async def execute(self, **kwargs) -> Any: raise NotImplementedError

class MastermindAgent:
    """
    The MastermindAgent: Oversees the evolution and high-level strategy of the MindX system.
    It can initiate the creation of new capabilities or agents by tasking subordinate systems.
    """
    _instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, # pragma: no cover
                           agent_id: str = "mastermind_alpha_001", 
                           config_override: Optional[Config] = None,
                           belief_system_override: Optional[BeliefSystem] = None,
                           coordinator_agent_override: Optional[CoordinatorAgent] = None,
                           test_mode: bool = False) -> 'MastermindAgent':
        """Factory method to get or create the singleton MastermindAgent instance."""
        async with cls._lock:
            if cls._instance is None or test_mode:
                if test_mode and cls._instance is not None:
                    logger.debug(f"MastermindAgent: Resetting instance for '{agent_id}' due to test_mode.")
                    await cls._instance.shutdown() # Ensure graceful shutdown of previous test instance
                    cls._instance = None # Allow re-creation
                
                logger.info(f"MastermindAgent: Creating new instance for ID '{agent_id}'.")
                # Dependencies must be ready for Mastermind's __init__
                cfg = config_override or Config(test_mode=test_mode) # Ensure config respects test_mode
                bs = belief_system_override or BeliefSystem(test_mode=test_mode)
                coord = coordinator_agent_override
                if not coord: # pragma: no cover # In real app, Coordinator is a prerequisite
                    from mindx.orchestration.coordinator_agent import get_coordinator_agent_mindx_async
                    logger.warning(f"MastermindAgent: CoordinatorAgent not provided, attempting to get/create default instance.")
                    coord = await get_coordinator_agent_mindx_async(config_override=cfg, test_mode=test_mode)

                instance = cls(
                    agent_id=agent_id, 
                    belief_system_instance=bs,
                    coordinator_agent_instance=coord,
                    config_override=cfg,
                    _is_factory_called=True,
                    test_mode=test_mode
                )
                cls._instance = instance
            return cls._instance

    def __init__(self, 
                 agent_id: str, 
                 belief_system_instance: BeliefSystem,
                 coordinator_agent_instance: CoordinatorAgent, # Requires a Coordinator instance
                 config_override: Optional[Config] = None,
                 _is_factory_called: bool = False, # Internal flag for factory use
                 test_mode: bool = False):
        
        if not _is_factory_called and agent_id not in MastermindAgent._instances : # pragma: no cover
             logger.warning(f"MastermindAgent direct instantiation for '{agent_id}'. Use get_instance() classmethod.")

        if hasattr(self, '_initialized') and self._initialized and not test_mode: # pragma: no cover
            return

        self.agent_id = agent_id
        self.config = config_override or Config()
        self.belief_system = belief_system_instance
        self.coordinator_agent = coordinator_agent_instance # Key dependency
        self.log_prefix = f"MastermindAgent ({self.agent_id}):"

        self.data_dir = PROJECT_ROOT / "data" / "mastermind_alpha_nexus" / self.agent_id.replace(":", "_")
        self._ensure_data_dir()

        # Mastermind's own LLM for high-level strategic thought / goal formulation
        mastermind_llm_provider = self.config.get(f"mastermind_agent.{self.agent_id}.llm.provider", self.config.get("llm.default_provider"))
        mastermind_llm_model = self.config.get(f"mastermind_agent.{self.agent_id}.llm.model", self.config.get(f"llm.{mastermind_llm_provider}.default_model_for_strategy", self.config.get(f"llm.{mastermind_llm_provider}.default_model")))
        self.llm_handler: LLMHandler = create_llm_handler(mastermind_llm_provider, mastermind_llm_model)
        logger.info(f"{self.log_prefix} Internal LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default'}")

        # Mastermind uses a BDI agent for its own strategic execution loop
        self.bdi_agent = BDIAgent(
            domain=f"mastermind_strategy_{self.agent_id}",
            belief_system_instance=self.belief_system, # Shared belief system
            config_override=self.config, # BDI will pick up its specific config
            test_mode=test_mode
        )
        self._register_mastermind_bdi_actions()

        # IDManager for generating identities for new agents/tools MindX might create
        self.id_manager_agent = IDManagerAgent.get_instance(
            agent_id=f"mastermind_id_manager_{self.agent_id}", 
            config_override=self.config,
            test_mode=test_mode
        )
        
        self.strategic_campaigns_history: List[Dict[str,Any]] = self._load_json_file("mastermind_campaigns_history.json", [])
        self.high_level_objectives: List[Dict[str,Any]] = self._load_json_file("mastermind_objectives.json", []) # Persisted objectives

        self.autonomous_loop_task: Optional[asyncio.Task] = None
        if self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.enabled", False) and not test_mode: # pragma: no cover
            self.start_autonomous_loop()

        logger.info(f"{self.log_prefix} Initialized. Data dir: {self.data_dir}. Autonomous: {bool(self.autonomous_loop_task)}")
        self._initialized = True

    def _ensure_data_dir(self): # pragma: no cover
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt': os.chmod(self.data_dir, stat.S_IRWXU) # rwx for owner
        except Exception as e: logger.error(f"{self.log_prefix} Failed to create data dir {self.data_dir}: {e}")

    def _load_json_file(self, file_name: str, default_value: Union[List, Dict]) -> Union[List, Dict]: # pragma: no cover
        file_path = self.data_dir / file_name
        if file_path.exists():
            try:
                with file_path.open("r", encoding="utf-8") as f: return json.load(f)
            except Exception as e: logger.error(f"{self.log_prefix} Error loading {file_name}: {e}")
        return default_value

    def _save_json_file(self, file_name: str, data: Union[List, Dict]): # pragma: no cover
        file_path = self.data_dir / file_name
        try:
            with file_path.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            logger.debug(f"{self.log_prefix} Saved data to {file_name}")
        except Exception as e: logger.error(f"{self.log_prefix} Error saving {file_name}: {e}")

    def _register_mastermind_bdi_actions(self): # pragma: no cover
        """Registers handlers for actions Mastermind's internal BDI agent can take."""
        self.bdi_agent._action_handlers["OBSERVE_MINDX_STATE"] = self._bdi_action_observe_mindx_state
        self.bdi_agent._action_handlers["FORMULATE_STRATEGIC_CAMPAIGN_GOAL"] = self._bdi_action_formulate_campaign_goal
        self.bdi_agent._action_handlers["LAUNCH_IMPROVEMENT_CAMPAIGN"] = self._bdi_action_launch_improvement_campaign
        self.bdi_agent._action_handlers["REQUEST_NEW_AGENT_IDENTITY"] = self._bdi_action_request_new_agent_identity
        self.bdi_agent._action_handlers["INITIATE_NEW_AGENT_DEVELOPMENT"] = self._bdi_action_initiate_new_agent_dev
        self.bdi_agent._action_handlers["REVIEW_CAMPAIGN_OUTCOMES"] = self._bdi_action_review_campaign_outcomes
        logger.info(f"{self.log_prefix} Registered custom BDI action handlers for strategic operations.")

    # --- Mastermind's BDI Action Handlers ---
    async def _bdi_action_observe_mindx_state(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        logger.info(f"{self.log_prefix} BDI Action: Observing MindX system state.")
        # Query Coordinator for its status, backlog size, monitor summaries
        # This requires Coordinator to have methods to provide this info, or query its beliefs.
        # For now, simulate by getting a high-level analysis from Coordinator.
        focus = params.get("analysis_focus", "current system health and top 2 pending improvement needs")
        analysis_interaction = await self.coordinator_agent.handle_user_input(
            content=f"Mastermind request: Provide system state summary focused on: {focus}",
            agent_id=self.agent_id,
            interaction_type=InteractionType.SYSTEM_ANALYSIS, # Use existing type
            metadata={"source": self.agent_id, "analysis_depth": "summary"}
        )
        if analysis_interaction.get("status") == InteractionStatus.COMPLETED.value and analysis_interaction.get("response"):
            # Store summary in Mastermind's BDI beliefs
            await self.bdi_agent.update_belief("mindx_state_summary", analysis_interaction["response"], 0.8, BeliefSource.COMMUNICATION)
            return True, {"summary_retrieved": True, "response": analysis_interaction["response"]}
        return False, {"message": f"Failed to get MindX state summary from Coordinator. Error: {analysis_interaction.get('error')}"}

    async def _bdi_action_formulate_campaign_goal(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        high_level_directive = params.get("directive", "Proactively evolve and improve the MindX system.")
        mindx_state_summary = await self.bdi_agent.get_belief("mindx_state_summary")
        
        prompt = (f"You are the MastermindAgent of MindX. Your directive is: '{high_level_directive}'.\n"
                  f"Current MindX State Summary (from Coordinator):\n{json.dumps(mindx_state_summary, indent=2)[:2000]}\n\n" # Truncate
                  f"Based on the directive and current state, formulate a single, specific, and actionable "
                  f"strategic campaign goal for the StrategicEvolutionAgent (SEA) or Coordinator. This goal should be achievable "
                  f"through a series of tactical code improvements or analyses. "
                  f"The goal should be a string. Respond ONLY with a JSON object: {{\"campaign_goal_description\": \"Your formulated goal.\"}}")
        try:
            response_str = await self.llm_handler.generate_text(prompt, max_tokens=300, temperature=0.4, json_mode=True)
            # ... (Robust JSON parsing for {"campaign_goal_description": "..."}) ...
            parsed_response = {}; # Placeholder
            try: parsed_response = json.loads(response_str)
            except json.JSONDecodeError: match = re.search(r"(\{[\s\S]*?\})", response_str, re.DOTALL);
            if match: parsed_response = json.loads(match.group(1))
            else: raise ValueError("LLM response for campaign goal not JSON.")
            
            campaign_goal_desc = parsed_response.get("campaign_goal_description")
            if campaign_goal_desc:
                await self.bdi_agent.update_belief("current_formulated_campaign_goal", campaign_goal_desc)
                return True, {"campaign_goal_description": campaign_goal_desc}
            return False, {"message": "LLM failed to formulate a campaign goal description."} # pragma: no cover
        except Exception as e: return False, {"message": f"Error formulating campaign goal: {e}"} # pragma: no cover

    async def _bdi_action_launch_improvement_campaign(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        campaign_goal_desc = params.get("campaign_goal_description") or await self.bdi_agent.get_belief("current_formulated_campaign_goal")
        if not campaign_goal_desc: return False, {"message": "No campaign goal description to launch."}

        # This Mastermind doesn't directly have an SEA instance.
        # It tells the Coordinator to *potentially* task an SEA or manage the campaign itself.
        # Let's assume Coordinator has a way to receive such high-level campaign goals.
        # For now, we make a special "SYSTEM_ANALYSIS" call with a very specific focus.
        # A more direct approach would be for Coordinator to have `start_improvement_campaign(goal)`
        logger.info(f"{self.log_prefix} BDI Action: Requesting Coordinator to initiate campaign for goal: '{campaign_goal_desc}'")
        
        interaction_meta = {
            "source": self.agent_id,
            "campaign_goal_for_mindx": campaign_goal_desc,
            "analysis_focus": f"Strategic campaign: {campaign_goal_desc}", # This hints to SYSTEM_ANALYSIS
            # If SEA is a registered agent the Coordinator knows:
            # "delegate_to_agent_id": "strategic_evolution_agent_instance_id_if_known"
        }
        # We'll use SYSTEM_ANALYSIS, and the Coordinator's autonomous loop will pick up suggestions.
        # Or, we can make a COMPONENT_IMPROVEMENT on "mindx.system_wide" (conceptual target)
        # Let's use a more direct task if Coordinator supports it, otherwise fall back to SYSTEM_ANALYSIS.
        # For this example, let's assume Coordinator can handle a general "COMPONENT_IMPROVEMENT"
        # where target_component is "mindx_system" and analysis_context is the campaign goal.
        
        # This is a high-level task for the Coordinator. The Coordinator will then
        # use its own SystemAnalysis, add to its backlog, and its autonomous loop
        # (or a specific new handler in Coordinator for "CAMPAIGN" type Interaction)
        # would pick these up and make targeted calls to SIA.
        # For now, let's simulate this by asking Coordinator to analyze with this focus.
        # The actual *execution* of sub-parts will be tracked by Coordinator's backlog.
        # This Mastermind BDI action is successful if the Coordinator accepts the request.
        
        interaction = await self.coordinator_agent.create_interaction(
            interaction_type=InteractionType.SYSTEM_ANALYSIS, # Or a new "START_CAMPAIGN" type
            content=f"Mastermind directive: Initiate system evolution campaign for goal: {campaign_goal_desc}",
            agent_id=self.agent_id,
            metadata=interaction_meta
        )
        processed_interaction = await self.coordinator_agent.process_interaction(interaction)

        if processed_interaction.status == InteractionStatus.COMPLETED:
            campaign_id = interaction.interaction_id # Use interaction_id as campaign_id
            await self.bdi_agent.update_belief(f"active_campaigns.{campaign_id}.goal", campaign_goal_desc)
            await self.bdi_agent.update_belief(f"active_campaigns.{campaign_id}.status", "initiated_via_coordinator")
            logger.info(f"{self.log_prefix} Improvement campaign '{campaign_goal_desc}' (ID {campaign_id}) initiated via Coordinator.")
            return True, {"campaign_id": campaign_id, "coordinator_interaction_id": interaction.interaction_id, "message": "Campaign initiated."}
        else: # pragma: no cover
            logger.error(f"{self.log_prefix} Failed to initiate campaign via Coordinator. Error: {processed_interaction.error}")
            return False, {"message": f"Failed to initiate campaign: {processed_interaction.error}"}

    async def _bdi_action_request_new_agent_identity(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        entity_id_for_identity = params.get("entity_id", f"mindx_generated_entity_{str(uuid.uuid4())[:6]}")
        logger.info(f"{self.log_prefix} BDI Action: Requesting new agent identity from IDManager for '{entity_id_for_identity}'.")
        try:
            public_address, _ = self.id_manager_agent.create_new_wallet(entity_id=entity_id_for_identity)
            await self.bdi_agent.update_belief(f"identities.{entity_id_for_identity}.public_address", public_address)
            return True, {"entity_id": entity_id_for_identity, "public_address": public_address}
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to create new agent identity: {e}", exc_info=True)
            return False, {"message": f"Identity creation failed: {e}"}

    async def _bdi_action_initiate_new_agent_dev(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        """Highly conceptual: Task Coordinator/SEA to develop a new agent/component."""
        new_agent_description = params.get("description", "A new utility agent for specialized data parsing.")
        new_agent_identity_belief = params.get("identity_belief_key") # e.g., "identities.new_parser_agent.public_address"
        
        public_address = await self.bdi_agent.get_belief(new_agent_identity_belief) if new_agent_identity_belief else None
        
        logger.info(f"{self.log_prefix} BDI Action: Initiating development of new agent: {new_agent_description}. Identity: {public_address or 'None yet'}")
        
        # This would be a very complex campaign for the Coordinator/SEA.
        # For this stub, we just log the intent and return success.
        # A real implementation would create a high-level CAMPAIGN_GOAL for the Coordinator.
        campaign_goal_for_new_agent = f"Develop and integrate a new agent/component: {new_agent_description}. Target identity if pre-assigned: {public_address}"
        
        # Simulate launching this as a new high-level campaign via existing handler
        return await self._bdi_action_launch_improvement_campaign({"campaign_goal_description": campaign_goal_for_new_agent})


    async def _bdi_action_review_campaign_outcomes(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        campaign_id_to_review = params.get("campaign_id") # From belief or param
        if not campaign_id_to_review: return False, {"message": "No campaign_id provided for review."}

        logger.info(f"{self.log_prefix} BDI Action: Reviewing outcomes for campaign ID '{campaign_id_to_review}'.")
        # Fetch data from Coordinator's campaign history or this Mastermind's history
        campaign_data = next((c for c in self.strategic_campaigns_history if c.get("campaign_run_id") == campaign_id_to_review or c.get("interaction_id") == campaign_id_to_review), None)
        
        if not campaign_data: return False, {"message": f"Campaign data for '{campaign_id_to_review}' not found."}

        # Use LLM to assess if the campaign met its original strategic goal
        original_goal = campaign_data.get("campaign_goal", "Unknown original goal")
        outcome_summary = campaign_data.get("bdi_outcome_message", campaign_data.get("summary_message", "No summary"))

        prompt = (f"Review the outcome of an MindX improvement campaign.\n"
                  f"Original Campaign Goal: {original_goal}\n"
                  f"Reported Outcome/Summary: {outcome_summary}\n"
                  f"Did this campaign successfully achieve its original goal? "
                  f"Respond ONLY with JSON: {{\"achieved_goal\": boolean, \"assessment_summary\": \"brief explanation\"}}")
        try:
            response_str = await self.llm_handler.generate_text(prompt, max_tokens=200, temperature=0.1, json_mode=True)
            # ... (Robust JSON parsing) ...
            parsed_assessment = {}; # Placeholder
            try: parsed_assessment = json.loads(response_str)
            except: match = re.search(r"(\{[\s\S]*?\})", response_str, re.DOTALL); 
            if match: parsed_assessment = json.loads(match.group(1))
            else: raise ValueError("LLM eval response not JSON.")
            if not isinstance(parsed_assessment, dict) or not all(k in parsed_assessment for k in ["achieved_goal", "assessment_summary"]): raise ValueError("LLM assessment missing keys.")
            
            await self.bdi_agent.update_belief(f"campaign_review.{campaign_id_to_review}", parsed_assessment)
            return True, parsed_assessment
        except Exception as e: return False, {"message": f"LLM review failed: {e}"} # pragma: no cover


    # --- Main Mastermind Orchestration Loop ---
    async def manage_mindx_evolution(self, top_level_directive: str, max_mastermind_run_cycles: int = 5): # pragma: no cover
        """
        The primary entry point for Mastermind to act on a high-level directive.
        It uses its internal BDI agent to plan and execute a campaign.
        """
        self._internal_state["current_run_id"] = f"mastermind_run_{str(uuid.uuid4())[:8]}"
        run_id = self._internal_state["current_run_id"]
        logger.info(f"{self.log_prefix} Starting MindX evolution campaign (Run ID: {run_id}). Directive: '{top_level_directive}'")
        await self.update_belief(f"mastermind.current_campaign.directive", top_level_directive, 0.99, BeliefSource.EXTERNAL_INPUT, is_internal_state=True)
        await self.update_belief(f"mastermind.current_campaign.run_id", run_id, 0.99, BeliefSource.SELF_ANALYSIS, is_internal_state=True)

        # Set the directive as the primary goal for Mastermind's BDI agent
        self.bdi_agent.set_goal(
            goal_description=f"Fulfill Mastermind directive: {top_level_directive}",
            priority=1, is_primary=True, goal_id=f"mastermind_directive_{run_id}"
        )
        
        # Run the BDI agent. Its plan will involve the _sea_action_* methods defined above.
        bdi_final_outcome = await self.bdi_agent.run(max_cycles=max_mastermind_run_cycles)
        
        campaign_outcome_summary = {
            "mastermind_run_id": run_id,
            "directive": top_level_directive,
            "bdi_final_status": self.bdi_agent._internal_state["status"],
            "bdi_outcome_message": bdi_final_outcome,
            "timestamp": time.time()
        }
        self.strategic_campaigns_history.append(campaign_outcome_summary)
        self._save_json_file("mastermind_campaigns_history.json", self.strategic_campaigns_history)
        
        await self.update_belief(f"mastermind.campaign_history.{run_id}", campaign_outcome_summary, 0.95, BeliefSource.SELF_ANALYSIS)
        logger.info(f"{self.log_prefix} MindX evolution campaign (Run ID: {run_id}) finished. BDI Status: {campaign_outcome_summary['bdi_final_status']}")
        return campaign_outcome_summary

    def start_autonomous_loop(self, interval_seconds: Optional[float] = None): # pragma: no cover
        """Starts the Mastermind's autonomous strategic loop."""
        if self.autonomous_loop_task and not self.autonomous_loop_task.done():
            logger.warning(f"{self.log_prefix} Autonomous loop already running.")
            return
        
        loop_interval = interval_seconds or self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.interval_seconds", 3600 * 6) # Default 6 hours
        default_directive = self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.default_directive", "Proactively monitor and enhance overall MindX system health and capabilities.")
        max_cycles = self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.max_bdi_cycles", 25)

        self.autonomous_loop_task = asyncio.create_task(self._mastermind_autonomous_worker(loop_interval, default_directive, max_cycles))
        logger.info(f"{self.log_prefix} Mastermind autonomous loop started. Interval: {loop_interval}s. Default Directive: '{default_directive}'")

    async def _mastermind_autonomous_worker(self, interval: float, default_directive: str, max_cycles: int): # pragma: no cover
        logger.info(f"{self.log_prefix} Mastermind autonomous worker started.")
        while True:
            try:
                await asyncio.sleep(interval)
                logger.info(f"{self.log_prefix} Autonomous worker: Initiating new strategic campaign.")
                # Potentially, could get a dynamic directive from beliefs or another source.
                await self.manage_mindx_evolution(default_directive, max_mastermind_run_cycles=max_cycles)
            except asyncio.CancelledError: logger.info(f"{self.log_prefix} Mastermind autonomous worker stopping."); break
            except Exception as e: logger.error(f"{self.log_prefix} Mastermind autonomous worker error: {e}", exc_info=True); await asyncio.sleep(interval) # Wait before retry on error

    async def shutdown(self): # pragma: no cover
        logger.info(f"MastermindAgent '{self.agent_id}' shutting down...")
        if self.autonomous_loop_task and not self.autonomous_loop_task.done():
            self.autonomous_loop_task.cancel()
            try: await self.autonomous_loop_task
            except asyncio.CancelledError: pass
        if self.bdi_agent: await self.bdi_agent.shutdown()
        if self.id_manager_agent and hasattr(self.id_manager_agent, 'shutdown'): # IDManager may not have async shutdown
            if asyncio.iscoroutinefunction(self.id_manager_agent.shutdown): await self.id_manager_agent.shutdown()
            else: self.id_manager_agent.shutdown() # type: ignore
        self._save_json_file("mastermind_campaigns_history.json", self.strategic_campaigns_history)
        self._save_json_file("mastermind_objectives.json", self.high_level_objectives)
        logger.info(f"MastermindAgent '{self.agent_id}' shutdown complete.")

    @classmethod
    async def reset_all_instances_for_testing(cls): # pragma: no cover
        """Resets all cached Mastermind instances. Primarily for testing."""
        async with cls._lock:
            for agent_id, instance in list(cls._instances.items()):
                await instance.shutdown()
            cls._instances.clear()
        logger.debug("All MastermindAgent instances reset.")

# Example of how Mastermind might be run (e.g. from a main application script)
async def run_mastermind_example(): # pragma: no cover
    config = Config(test_mode=True); Config.reset_instance(); config = Config(test_mode=True) # Fresh config
    belief_system = BeliefSystem(test_mode=True); BeliefSystem.reset_instance(); belief_system = BeliefSystem(test_mode=True)
    
    # Coordinator is a dependency for Mastermind (indirectly for SIA calls)
    # Use the async factory for Coordinator
    from mindx.orchestration.coordinator_agent import get_coordinator_agent_mindx_async
    coordinator = await get_coordinator_agent_mindx_async(config_override=config, test_mode=True)

    # Get/Create Mastermind instance
    mastermind = await MastermindAgent.get_instance(
        agent_id="mindx_overseer_alpha",
        config_override=config,
        belief_system_override=belief_system,
        coordinator_agent_override=coordinator, # Pass the initialized coordinator
        test_mode=True
    )

    # Define a high-level directive for the Mastermind
    directive = "Analyze the current MindX system and initiate one campaign to improve the robustness of a utility module, then confirm the outcome."
    
    logger.info(f"\n--- Mastermind Campaign Starting for Directive: {directive} ---")
    campaign_result = await mastermind.manage_mindx_evolution(
        top_level_directive=directive,
        max_mastermind_run_cycles=15 # Allow BDI enough cycles to plan and execute a few steps
    )
    logger.info(f"\n--- Mastermind Campaign Result ---")
    print(json.dumps(campaign_result, indent=2, default=str))

    # If Mastermind has an autonomous loop configured, it would run separately.
    # To test that:
    # mastermind.config.config_data["mastermind_agent"] = mastermind.config.config_data.get("mastermind_agent",{})
    # mastermind.config.config_data["mastermind_agent"][mastermind.agent_id] = mastermind.config.config_data["mastermind_agent"].get(mastermind.agent_id,{})
    # mastermind.config.config_data["mastermind_agent"][mastermind.agent_id]["autonomous_loop"] = {"enabled": True, "interval_seconds": 10, "max_bdi_cycles":5} # Short interval for test
    # mastermind.start_autonomous_loop()
    # await asyncio.sleep(30) # Let it run a few times
    # await mastermind.shutdown()


    # Final shutdown
    await mastermind.shutdown()
    await coordinator.shutdown()
    # Manually shutdown monitors if their factories don't link to coordinator's test_mode for shutdown
    rm = await get_resource_monitor_async(test_mode=True); rm.stop_monitoring()
    pm = await get_performance_monitor_async(test_mode=True); await pm.shutdown()


if __name__ == "__main__": # pragma: no cover
    project_r_main = Path(__file__).resolve().parent.parent.parent
    env_p_main = project_r_main / ".env"
    if env_p_main.exists(): from dotenv import load_dotenv; load_dotenv(dotenv_path=env_p_main, override=True)
    else: print(f"Main: .env not found at {env_p_main}.", file=sys.stderr)
    
    try:
        asyncio.run(run_mastermind_example())
    except KeyboardInterrupt: logger.info("Mastermind example interrupted.")
    finally: logger.info("Mastermind example finished.")
