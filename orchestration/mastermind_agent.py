# orchestration/mastermind_agent.py
import os
import asyncio
import json
import time
import uuid
import re
import copy
import stat
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable, Union, Set

from utils.config import Config, PROJECT_ROOT
from utils.logging_config import get_logger # Import get_logger first
logger = get_logger(__name__) # INITIALIZE LOGGER HERE

from core.belief_system import BeliefSystem, BeliefSource
from llm.llm_interface import LLMHandlerInterface
from llm.llm_factory import create_llm_handler
from core.bdi_agent import BDIAgent, BaseTool as BDIBaseTool
from learning.goal_management import Goal as BDIGoal, GoalSt as BDIGoalStatus
from core.id_manager_agent import IDManagerAgent
from .coordinator_agent import CoordinatorAgent, InteractionType, InteractionStatus, Interaction
import importlib

# Attempt to import CodeBaseGenerator from the 'tools' package
CodeBaseGenerator = None # Initialize to None
try:
    # This assumes base_gen_agent.py is in the 'tools' top-level package
    from tools.base_gen_agent import BaseGenAgent as ImportedCodeBaseGenerator
    CodeBaseGenerator = ImportedCodeBaseGenerator
    logger.info("MastermindAgent: CodeBaseGenerator successfully imported from tools.base_gen_agent.")
except ImportError as e1: # pragma: no cover
    logger.warning(f"MastermindAgent: Could not import CodeBaseGenerator from tools.base_gen_agent ({e1}). This is expected if 'tools.base_gen_agent' is not the correct path or 'tools' is not a package.")
    try:
        from base_gen_agent import BaseGenAgent as FallbackCodeBaseGenerator # If it's a top-level module
        CodeBaseGenerator = FallbackCodeBaseGenerator
        logger.info("MastermindAgent: CodeBaseGenerator imported as a top-level module (fallback).")
    except ImportError as e2:
        logger.warning(f"MastermindAgent: CodeBaseGenerator (from base_gen_agent) also not found ({e2}). Code analysis capabilities will be limited.")
        # CodeBaseGenerator remains None if both attempts fail

class MastermindAgent:
    _instance = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls,
                           agent_id: Optional[str] = None,
                           config_override: Optional[Config] = None,
                           belief_system_override: Optional[BeliefSystem] = None,
                           coordinator_agent_override: Optional[CoordinatorAgent] = None,
                           test_mode: bool = False) -> 'MastermindAgent':
        async with cls._lock:
            temp_config = config_override or Config(test_mode=test_mode)
            effective_agent_id = agent_id or temp_config.get("mastermind_agent.default_agent_id", "mastermind_prime_augmentic")

            if cls._instance is None or test_mode or cls._instance.agent_id != effective_agent_id:
                if test_mode and cls._instance is not None and cls._instance.agent_id == effective_agent_id:
                    logger.debug(f"MastermindAgent Factory (mindX): Resetting test instance '{effective_agent_id}'.")
                    await cls._instance.shutdown()
                    cls._instance = None
                elif test_mode and cls._instance is not None and cls._instance.agent_id != effective_agent_id: # pragma: no cover
                    logger.debug(f"MastermindAgent Factory (mindX): Test mode switching ID from '{cls._instance.agent_id}' to '{effective_agent_id}'.")
                    await cls._instance.shutdown()
                    cls._instance = None

                logger.info(f"MastermindAgent Factory (mindX): Creating new instance for ID '{effective_agent_id}'.")
                cfg = config_override or Config(test_mode=test_mode)
                bs = belief_system_override or BeliefSystem(test_mode=test_mode)

                coord = coordinator_agent_override
                if not coord and not test_mode: # pragma: no cover
                    from .coordinator_agent import get_coordinator_agent_mindx_async
                    logger.warning(f"MastermindAgent Factory (mindX): CoordinatorAgent not provided for '{effective_agent_id}', getting default.")
                    coord = await get_coordinator_agent_mindx_async(config_override=cfg, test_mode=test_mode)
                elif not coord and test_mode:
                    logger.warning(f"MastermindAgent Factory (mindX test_mode): CoordinatorAgent not provided for '{effective_agent_id}'.")

                if not coord and not test_mode:
                     logger.error(f"MastermindAgent Factory (mindX): CRITICAL - CoordinatorAgent could not be obtained for '{effective_agent_id}'.")

                instance = cls(
                    agent_id=effective_agent_id,
                    belief_system_instance=bs,
                    coordinator_agent_instance=coord,
                    config_override=cfg,
                    _is_factory_called=True,
                    test_mode=test_mode
                )
                await instance._async_init_components()
                cls._instance = instance
            elif cls._instance.agent_id != effective_agent_id: # pragma: no cover
                logger.error(f"MastermindAgent Factory (mindX): ID mismatch. Requested '{effective_agent_id}' but singleton is '{cls._instance.agent_id}'.")
            return cls._instance

    def __init__(self,
                 agent_id: str,
                 belief_system_instance: BeliefSystem,
                 coordinator_agent_instance: Optional[CoordinatorAgent],
                 config_override: Optional[Config] = None,
                 _is_factory_called: bool = False,
                 test_mode: bool = False):

        if not _is_factory_called and (MastermindAgent._instance is None or MastermindAgent._instance is not self) : # pragma: no cover
             logger.warning(f"MastermindAgent (mindX) direct instantiation for '{agent_id}'. Prefer `await MastermindAgent.get_instance(...)`.")

        if hasattr(self, '_initialized_sync') and self._initialized_sync and not test_mode: # pragma: no cover
            return

        self.agent_id = agent_id
        self.config = config_override or Config(test_mode=test_mode)
        self.belief_system = belief_system_instance
        self.coordinator_agent = coordinator_agent_instance
        self.log_prefix = f"Mastermind ({self.agent_id} of mindX):"
        self.test_mode = test_mode

        data_dir_config_key = f"mastermind_agent.{self.agent_id}.data_dir_relative_to_project"
        default_data_dir = f"data/mastermind_work/{self.agent_id.replace(':', '_').replace(' ','_')}"
        data_dir_relative_path = self.config.get(data_dir_config_key, default_data_dir)
        self.data_dir: Path = PROJECT_ROOT / data_dir_relative_path
        self._ensure_data_dir()

        tools_registry_rel_path = self.config.get(
            f"mastermind_agent.{self.agent_id}.tools_registry_path",
            "data/config/official_tools_registry.json"
        )
        self.tools_registry_file_path: Path = PROJECT_ROOT / tools_registry_rel_path
        self.tools_registry: Dict[str, Any] = self._load_tools_registry()

        self.llm_handler: Optional[LLMHandlerInterface] = None

        self.bdi_agent = BDIAgent(
            domain=f"mastermind_strategy_{self.agent_id.replace(':','_').replace(' ','_')}",
            belief_system_instance=self.belief_system,
            config_override=self.config,
            test_mode=self.test_mode
        )

        self.code_base_analyzer: Optional[CodeBaseGenerator] = None # type: ignore
        if CodeBaseGenerator:
            base_gen_config_path_str = self.config.get(
                "mastermind_agent.base_gen_config_path_override",
                None
            )
            try:
                self.code_base_analyzer = CodeBaseGenerator(
                    config_file_path_override_str=base_gen_config_path_str,
                    agent_id=f"base_gen_for_{self.agent_id}"
                )
                logger.info(f"{self.log_prefix} CodeBaseGenerator initialized for code analysis by Mastermind.")
            except Exception as e_cba: # pragma: no cover
                logger.error(f"{self.log_prefix} Failed to initialize CodeBaseGenerator: {e_cba}", exc_info=True)
        else: # pragma: no cover
            logger.warning(f"{self.log_prefix} CodeBaseGenerator class not available (import failed), Mastermind's code analysis capabilities are limited.")

        self._register_mastermind_bdi_actions()

        self.id_manager_agent: Optional[IDManagerAgent] = None
        self.strategic_campaigns_history: List[Dict[str,Any]] = self._load_json_file("mastermind_campaigns_history.json", [])
        self.high_level_objectives: List[Dict[str,Any]] = self._load_json_file("mastermind_objectives.json", [])
        self._internal_state: Dict[str, Any] = {}
        self.autonomous_loop_task: Optional[asyncio.Task] = None
        self._initialized_sync = True
        self._initialized_async = False

    async def _async_init_components(self):
        if self._initialized_async and not self.test_mode: # pragma: no cover
            return

        mastermind_llm_config_key_prefix = f"mastermind_agent.{self.agent_id}.llm"
        llm_provider_cfg = self.config.get(f"{mastermind_llm_config_key_prefix}.provider", self.config.get("llm.default_provider"))
        llm_model_cfg = self.config.get(f"{mastermind_llm_config_key_prefix}.model",
                                       self.config.get(f"llm.{llm_provider_cfg}.default_model_for_strategy",
                                                       self.config.get(f"llm.{llm_provider_cfg}.default_model")))
        try:
            self.llm_handler = await create_llm_handler(provider_name=llm_provider_cfg, model_name=llm_model_cfg)
            if self.llm_handler:
                 logger.info(f"{self.log_prefix} Internal Augmentic Intelligence LLM set to: {self.llm_handler.provider_name}/{self.llm_handler.model_name_for_api or 'default_for_provider'}")
            else: # pragma: no cover
                 logger.error(f"{self.log_prefix} Augmentic Intelligence LLM handler creation returned None. Operations will fail.")
                 from llm.mock_llm_handler import MockLLMHandler
                 self.llm_handler = MockLLMHandler(model_name="mock_due_to_creation_failure")
        except Exception as e: # pragma: no cover
            logger.error(f"{self.log_prefix} Failed to create Augmentic Intelligence LLM handler: {e}. Operations will fail.", exc_info=True)
            from llm.mock_llm_handler import MockLLMHandler
            self.llm_handler = MockLLMHandler(model_name="mock_due_to_error")

        if self.bdi_agent and hasattr(self.bdi_agent, 'async_init_components'):
            await self.bdi_agent.async_init_components()

        if self.config.get(f"mastermind_agent.{self.agent_id}.enable_id_management", True):
            try:
                id_manager_instance_id = self.config.get(f"mastermind_agent.{self.agent_id}.id_manager_instance_id",
                                                        f"id_manager_for_{self.agent_id}")
                self.id_manager_agent = await IDManagerAgent.get_instance(
                    agent_id=id_manager_instance_id,
                    config_override=self.config,
                    test_mode=self.test_mode
                )
            except Exception as e_id_mgr: # pragma: no cover
                logger.error(f"{self.log_prefix} Failed to initialize IDManagerAgent for mindX: {e_id_mgr}.", exc_info=True)

        if self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.enabled", False) and not self.test_mode: # pragma: no cover
            self.start_autonomous_loop()

        llm_status = 'Mock/Inactive'
        if self.llm_handler and self.llm_handler.provider_name != 'mock' and not "mock_due_to" in (self.llm_handler.model_name_for_api or ""):
            llm_status = 'Active (Augmentic Intelligence)'
        bdi_llm_ready = self.bdi_agent and self.bdi_agent.llm_handler is not None and self.bdi_agent.llm_handler.provider_name != 'mock'
        logger.info(f"{self.log_prefix} Asynchronously initialized. Mastermind LLM: {llm_status}. BDI LLM: {'Ready' if bdi_llm_ready else 'Not Ready/Mock'}. ID Manager: {'Active' if self.id_manager_agent else 'Inactive'}. Loop: {bool(self.autonomous_loop_task)}")
        self._initialized_async = True

    def _ensure_data_dir(self): # pragma: no cover
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt': os.chmod(self.data_dir, stat.S_IRWXU)
        except Exception as e: logger.error(f"{self.log_prefix} Failed to create data dir {self.data_dir} for mindX: {e}")

    def _load_json_file(self, file_name: str, default_value: Union[List, Dict]) -> Union[List, Dict]: # pragma: no cover
        file_path = self.data_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                with file_path.open("r", encoding="utf-8") as f: return json.load(f)
            except Exception as e: logger.error(f"{self.log_prefix} Error loading {file_name} for mindX from {file_path}: {e}")
        return copy.deepcopy(default_value)

    def _save_json_file(self, file_name: str, data: Union[List, Dict]): # pragma: no cover
        file_path = self.data_dir / file_name
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            logger.debug(f"{self.log_prefix} Saved data to {file_path} for mindX")
        except Exception as e: logger.error(f"{self.log_prefix} Error saving {file_name} to {file_path} for mindX: {e}")

    def _load_tools_registry(self) -> Dict[str, Any]: # pragma: no cover
        if self.tools_registry_file_path.exists() and self.tools_registry_file_path.is_file():
            try:
                with self.tools_registry_file_path.open("r", encoding="utf-8") as f:
                    registry_data = json.load(f)
                logger.info(f"{self.log_prefix} Loaded official tools registry from {self.tools_registry_file_path}")
                if "registered_tools" not in registry_data or not isinstance(registry_data["registered_tools"], dict):
                    logger.warning(f"{self.log_prefix} Tools registry missing 'registered_tools' dict. Initializing empty.")
                    registry_data["registered_tools"] = {}
                return registry_data
            except Exception as e:
                logger.error(f"{self.log_prefix} Error loading tools registry {self.tools_registry_file_path}: {e}. Starting with empty registry.", exc_info=True)
        else:
            logger.warning(f"{self.log_prefix} Official tools registry {self.tools_registry_file_path} not found. Starting with empty registry.")
        return {"tools_schema_version": "1.2", "last_updated_by": self.agent_id, "last_updated_at": time.time(), "registered_tools": {}}

    def _save_tools_registry(self): # pragma: no cover
        self.tools_registry["last_updated_by"] = self.agent_id
        self.tools_registry["last_updated_at"] = time.time()
        try:
            self.tools_registry_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.tools_registry_file_path.open("w", encoding="utf-8") as f:
                json.dump(self.tools_registry, f, indent=2, sort_keys=True)
            logger.info(f"{self.log_prefix} Saved official tools registry to {self.tools_registry_file_path}")
        except Exception as e:
            logger.error(f"{self.log_prefix} Error saving tools registry to {self.tools_registry_file_path}: {e}", exc_info=True)

    def _register_mastermind_bdi_actions(self): # pragma: no cover
        self.bdi_agent._action_handlers["OBSERVE_MINDX_SYSTEM_STATE"] = self._bdi_action_observe_mindx_state
        self.bdi_agent._action_handlers["FORMULATE_STRATEGIC_CAMPAIGN_GOAL"] = self._bdi_action_formulate_campaign_goal
        self.bdi_agent._action_handlers["LAUNCH_IMPROVEMENT_CAMPAIGN_VIA_COORDINATOR"] = self._bdi_action_launch_improvement_campaign
        self.bdi_agent._action_handlers["REQUEST_NEW_ENTITY_IDENTITY"] = self._bdi_action_request_new_entity_identity
        self.bdi_agent._action_handlers["INITIATE_NEW_COMPONENT_DEVELOPMENT_CAMPAIGN"] = self._bdi_action_initiate_new_component_dev
        self.bdi_agent._action_handlers["REVIEW_CAMPAIGN_OUTCOMES"] = self._bdi_action_review_campaign_outcomes
        self.bdi_agent._action_handlers["ASSESS_TOOL_SUITE_EFFECTIVENESS"] = self._bdi_action_assess_tool_suite
        self.bdi_agent._action_handlers["CONCEPTUALIZE_NEW_TOOL"] = self._bdi_action_conceptualize_new_tool
        self.bdi_agent._action_handlers["INITIATE_TOOL_CODING_CAMPAIGN"] = self._bdi_action_initiate_tool_coding_campaign
        self.bdi_agent._action_handlers["REGISTER_OR_UPDATE_TOOL_IN_REGISTRY"] = self._bdi_action_register_or_update_tool
        self.bdi_agent._action_handlers["MARK_TOOL_STATUS_IN_REGISTRY"] = self._bdi_action_mark_tool_status
        self.bdi_agent._action_handlers["EXECUTE_TOOL"] = self._bdi_action_execute_tool
        self.bdi_agent._action_handlers["ANALYZE_CODEBASE_FOR_STRATEGY"] = self._bdi_action_analyze_codebase_for_strategy
        logger.info(f"{self.log_prefix} Registered BDI action handlers, including codebase analysis and tool management.")

    async def _bdi_action_observe_mindx_state(self, action_params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        logger.info(f"{self.log_prefix} BDI Action: Observing mindX system state via Coordinator.")
        if not self.coordinator_agent: return False, {"message": "CoordinatorAgent not available."}
        focus = action_params.get("analysis_focus", "current system health, alerts, top pending improvements for mindX, and tool registry status")
        tool_registry_summary = f"{len(self.tools_registry.get('registered_tools', {}))} tools in official registry."
        content_for_coord = f"Mastermind request for mindX: Provide system state summary focused on: {focus}. Tool Registry Summary: {tool_registry_summary}"
        interaction = await self.coordinator_agent.create_interaction(
            interaction_type=InteractionType.SYSTEM_ANALYSIS, content=content_for_coord,
            agent_id=self.agent_id, metadata={"source": self.agent_id, "analysis_depth": "summary_for_mastermind_with_tools"}
        )
        processed_interaction = await self.coordinator_agent.process_interaction(interaction)
        if processed_interaction.status == InteractionStatus.COMPLETED and isinstance(processed_interaction.response, dict):
            state_summary = processed_interaction.response
            state_summary["coordinator_backlog_size"] = len(self.coordinator_agent.improvement_backlog)
            state_summary["coordinator_pending_approval_count"] = len([item for item in self.coordinator_agent.improvement_backlog if item.get("status") == InteractionStatus.PENDING_APPROVAL.value])
            state_summary["official_tool_registry_count"] = len(self.tools_registry.get("registered_tools", {}))
            await self.bdi_agent.update_belief("mindx_state_summary_from_coordinator", state_summary, 0.85, BeliefSource.COMMUNICATION)
            return True, {"summary_retrieved": True, "state_summary_preview": str(state_summary)[:200]}
        else:
            err_msg = f"Failed to get mindX state summary from Coordinator. Error: {processed_interaction.error or 'Unknown'}"
            logger.error(f"{self.log_prefix} {err_msg}")
            return False, {"message": err_msg}

    async def _bdi_action_formulate_campaign_goal(self, action_params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        if not self.llm_handler or not self.llm_handler.model_name_for_api: return False, {"message": "Mastermind LLM not initialized."}
        high_level_directive = action_params.get("directive", await self.bdi_agent.get_belief("current_mastermind_directive") or "Proactively evolve mindX using Augmentic Intelligence principles, including its toolset.")
        mindx_state_summary_val = await self.bdi_agent.get_belief("mindx_state_summary_from_coordinator")
        mindx_state_summary_str = json.dumps(mindx_state_summary_val, indent=2)[:2000] if mindx_state_summary_val else "No detailed state summary for mindX available."
        existing_objectives = [obj.get("goal_description") for obj in self.high_level_objectives if obj.get("status") == "active"]
        existing_objectives_str = ("Existing active Mastermind objectives for mindX:\n" + "\n".join([f"- {o}" for o in existing_objectives[:3]])) if existing_objectives else "No other active Mastermind objectives for mindX."
        tool_registry_preview = { "count": len(self.tools_registry.get("registered_tools", {})), "sample_tools": list(self.tools_registry.get("registered_tools", {}).keys())[:3] }
        tool_registry_summary_str = f"Official Tool Registry Summary: {json.dumps(tool_registry_preview, indent=2)}"
        prompt = (
            f"You are the strategic Augmentic Intelligence LLM for the MastermindAgent of mindX. Directive: '{high_level_directive}'.\n"
            f"Current mindX State (includes tool registry count):\n{mindx_state_summary_str}\n"
            f"{tool_registry_summary_str}\n"
            f"{existing_objectives_str}\n\n"
            f"Formulate a single, specific, and actionable STRATEGIC CAMPAIGN GOAL for the mindX system. This could involve component improvements, new component/tool development, or tool lifecycle management. "
            f"Example: 'Enhance the 'web_search_tool' to support advanced query operators and result filtering.' or 'Develop a new 'CodeDocumentationGenerator' tool based on existing 'base_gen_agent.py' capability.'\n"
            f"Respond ONLY JSON: {{\"formulated_campaign_goal\": \"Goal description for mindX.\"}}"
        )
        try:
            response_str = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api,
                                                                max_tokens=self.config.get(f"mastermind_agent.{self.agent_id}.llm.max_tokens_goal_formulation", 500),
                                                                temperature=0.4, json_mode=True)
            if not response_str: return False, {"message": "Augmentic Intelligence LLM returned empty response."}
            parsed_response = json.loads(response_str)
            campaign_goal_desc = parsed_response.get("formulated_campaign_goal")
            if campaign_goal_desc and isinstance(campaign_goal_desc, str):
                await self.bdi_agent.update_belief("current_formulated_campaign_goal_for_bdi", campaign_goal_desc)
                return True, {"formulated_campaign_goal_description": campaign_goal_desc}
            return False, {"message": "Augmentic Intelligence LLM failed to formulate valid campaign goal string."}
        except Exception as e: return False, {"message": f"Error formulating campaign goal via Augmentic Intelligence LLM: {e}"}

    async def _bdi_action_launch_improvement_campaign(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        campaign_goal_desc = params.get("campaign_goal_description") or await self.bdi_agent.get_belief("current_formulated_campaign_goal_for_bdi")
        if not campaign_goal_desc: return False, {"message": "No campaign goal to launch for mindX."}
        if not self.coordinator_agent: return False, {"message":"Coordinator agent not available for mindX."}
        logger.info(f"{self.log_prefix} BDI: Requesting Coordinator for mindX strategic campaign: '{campaign_goal_desc}'")
        interaction_meta = {"source": self.agent_id, "mastermind_campaign_goal": campaign_goal_desc, "analysis_context": f"System-wide analysis for mindX Mastermind goal: {campaign_goal_desc}", "analysis_depth": "strategic_campaign_kickoff"}
        interaction = await self.coordinator_agent.create_interaction(interaction_type=InteractionType.SYSTEM_ANALYSIS, content=f"Mastermind directive for mindX: Initiate for campaign goal: {campaign_goal_desc}", agent_id=self.agent_id, metadata=interaction_meta)
        processed_interaction = await self.coordinator_agent.process_interaction(interaction)
        if processed_interaction.status == InteractionStatus.COMPLETED:
            mastermind_run_id = self._internal_state.get("current_run_id", "unknown_run")
            mastermind_campaign_id = f"mstr_camp_{mastermind_run_id}_{str(uuid.uuid4())[:6]}"
            await self.bdi_agent.update_belief(f"mastermind_campaigns.{mastermind_campaign_id}.goal", campaign_goal_desc)
            await self.bdi_agent.update_belief(f"mastermind_campaigns.{mastermind_campaign_id}.status", "delegated_to_coordinator_analysis")
            await self.bdi_agent.update_belief(f"mastermind_campaigns.{mastermind_campaign_id}.coordinator_interaction_id", interaction.interaction_id)
            logger.info(f"{self.log_prefix} mindX Campaign '{campaign_goal_desc}' (MCID: {mastermind_campaign_id}) analysis delegated to Coordinator (Interaction ID: {interaction.interaction_id}).")
            return True, {"mastermind_campaign_id": mastermind_campaign_id, "message": "Campaign analysis delegated."}
        else:
            err_msg = f"Failed to delegate mindX campaign analysis. Error: {processed_interaction.error or 'Unknown'}"
            logger.error(f"{self.log_prefix} {err_msg}")
            return False, {"message": err_msg}

    async def _bdi_action_request_new_entity_identity(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        if not self.id_manager_agent: return False, {"message": "IDManagerAgent not available for mindX."}
        entity_tag = params.get("entity_tag", f"mindX_entity_{str(uuid.uuid4())[:6]}")
        entity_description = params.get("entity_description", "New mindX component/agent/tool (Augmentic Intelligence).")
        logger.info(f"{self.log_prefix} BDI: Requesting identity for mindX entity '{entity_tag}'.")
        try:
            public_address, private_key_env_var_name = self.id_manager_agent.create_new_wallet(entity_id=entity_tag)
            belief_key_base = f"managed_entities.{entity_tag}"
            await self.bdi_agent.update_belief(f"{belief_key_base}.public_address", public_address)
            await self.bdi_agent.update_belief(f"{belief_key_base}.description", entity_description)
            await self.bdi_agent.update_belief(f"{belief_key_base}.private_key_env_var", private_key_env_var_name)
            await self.bdi_agent.update_belief(f"{belief_key_base}.status", "identity_provisioned")
            logger.info(f"{self.log_prefix} Identity created for '{entity_tag}': Address {public_address}, PK Env Var: {private_key_env_var_name}")
            return True, {"entity_tag": entity_tag, "public_address": public_address, "private_key_env_var": private_key_env_var_name}
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to create mindX entity identity: {e}", exc_info=True)
            return False, {"message": f"Identity creation failed: {e}"}

    async def _bdi_action_initiate_new_component_dev(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        new_comp_desc = params.get("description", "Specialized data parsing utility agent for mindX (Augmentic Intelligence).")
        new_comp_identity_address = params.get("identity_public_address")
        target_module_path_sugg = params.get("target_module_path", f"new_agents.{new_comp_desc.lower().replace(' ','_').replace('(augmentic_intelligence)','ai').replace('mindx','')[:30]}")
        logger.info(f"{self.log_prefix} BDI: Initiating dev campaign for mindX component: {new_comp_desc}. Identity: {new_comp_identity_address or 'N/A'}. Target Path Suggestion (relative): {target_module_path_sugg}")
        campaign_goal_for_new_dev = (
            f"Spearhead development of new mindX (Augmentic Intelligence) component: '{new_comp_desc}'. "
            f"Suggested initial module path (relative to a suitable top-level package like 'agents' or 'tools'): '{target_module_path_sugg}'. "
            f"{('Assigned public identity for this component: ' + new_comp_identity_address) if new_comp_identity_address else 'An identity may need to be provisioned if relevant.'} "
            f"This campaign will likely involve: detailed specification, code generation by SIA over multiple iterations for the initial file(s), "
            f"creation of basic unit tests, and initial registration with the system if applicable."
        )
        return await self._bdi_action_launch_improvement_campaign({"campaign_goal_description": campaign_goal_for_new_dev})

    async def _bdi_action_review_campaign_outcomes(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        if not self.llm_handler or not self.llm_handler.model_name_for_api: return False, {"message": "Mastermind Augmentic Intelligence LLM handler not properly initialized."}
        campaign_id_to_review = params.get("campaign_id") or await self.bdi_agent.get_belief("last_launched_mastermind_campaign_id")
        if not campaign_id_to_review: return False, {"message": "No campaign_id for mindX review."}
        logger.info(f"{self.log_prefix} BDI: Reviewing outcomes for mindX Mastermind campaign '{campaign_id_to_review}'.")
        campaign_data_list = [c for c in self.strategic_campaigns_history if c.get("mastermind_run_id") == campaign_id_to_review or c.get("mastermind_campaign_id") == campaign_id_to_review]
        if not campaign_data_list: return False, {"message": f"No data for mindX campaign '{campaign_id_to_review}'."}
        campaign_data = campaign_data_list[0]
        original_goal = campaign_data.get("directive", campaign_data.get("campaign_goal", "N/A"))
        bdi_outcome_msg = campaign_data.get("bdi_outcome_message", "N/A"); bdi_final_status = campaign_data.get("bdi_final_status", "N/A")
        prompt = (f"Review mindX (Augmentic Intelligence) campaign outcome.\nGoal: {original_goal}\n"
                  f"BDI Status: {bdi_final_status}\nBDI Msg: {bdi_outcome_msg}\n"
                  f"Assess if objective met. Respond ONLY JSON: "
                  f"{{\"campaign_objective_met\": bool, \"assessment_summary\": \"brief assessment of mindX campaign\", \"suggested_follow_up\": \"string for mindX\"}}")
        try:
            response_str = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=400, temperature=0.1, json_mode=True)
            if not response_str: return False, {"message": "Augmentic Intelligence LLM returned empty response for review."}
            parsed_assessment = json.loads(response_str)
            if not isinstance(parsed_assessment, dict) or not all(k in parsed_assessment for k in ["campaign_objective_met", "assessment_summary", "suggested_follow_up"]):
                raise ValueError("LLM review missing keys.")
            await self.bdi_agent.update_belief(f"mastermind_campaign_review.{campaign_id_to_review}", parsed_assessment)
            return True, parsed_assessment
        except Exception as e: return False, {"message": f"Augmentic Intelligence LLM campaign review failed: {e}"}

    async def _bdi_action_assess_tool_suite(self, action_params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        logger.info(f"{self.log_prefix} BDI Action: Assessing tool suite effectiveness using official registry.")
        if not self.llm_handler or not self.llm_handler.model_name_for_api:
             return False, {"message": "Mastermind LLM not available."}

        tools_for_prompt = []
        for tool_id, tool_data in self.tools_registry.get("registered_tools", {}).items():
            capabilities_summary = [cap.get("name", "unknown_capability") for cap in tool_data.get("capabilities", [])[:3]]
            tools_for_prompt.append({
                "id": tool_id,
                "description": tool_data.get("description", "N/A")[:100],
                "status": tool_data.get("status", "unknown"),
                "version": tool_data.get("version", "N/A"),
                "capabilities_preview": capabilities_summary
            })
        active_objectives_desc = [obj.get("directive", "N/A") for obj in self.high_level_objectives if obj.get("status") == "active"]
        objectives_summary = "Current high-level system objectives:\n" + "\n".join(f"- {o}" for o in active_objectives_desc[:3]) if active_objectives_desc else "No specific active system objectives defined."
        tool_performance_summary = "Tool performance metrics (e.g., success rate, latency) are not yet integrated into this assessment."
        prompt = (
            f"You are the strategic LLM for MastermindAgent of mindX.\n"
            f"Task: Assess the current tool suite's effectiveness in supporting system objectives and identify gaps.\n"
            f"Official Registered Tools (summary):\n{json.dumps(tools_for_prompt, indent=2)}\n\n"
            f"System Objectives:\n{objectives_summary}\n\n"
            f"Tool Performance (placeholder):\n{tool_performance_summary}\n\n"
            f"Based on this, provide:\n"
            f"1. `overall_assessment`: Brief (1-2 sentences) on current tool suite coverage and effectiveness.\n"
            f"2. `strategic_tool_recommendations`: A list of recommendations, each a dictionary with:\n"
            f"   - `recommendation_type`: 'CONCEPTUALIZE_NEW_TOOL', 'ENHANCE_EXISTING_TOOL', or 'MARK_TOOL_STATUS_DEPRECATED'.\n"
            f"   - `tool_id_or_concept_name`: ID of existing tool OR a descriptive name for a new tool concept.\n"
            f"   - `justification`: Why this recommendation supports system objectives or addresses a gap/inefficiency.\n"
            f"   - `priority`: (1-10, 10 highest) for Mastermind to consider this recommendation.\n"
            f"Respond ONLY with a single JSON object containing `overall_assessment` and `strategic_tool_recommendations`."
        )
        try:
            response_str = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=1500, temperature=0.4, json_mode=True)
            if not response_str: return False, {"message": "LLM returned empty response for tool assessment."}
            assessment_result = json.loads(response_str)
            if not isinstance(assessment_result, dict) or not all(k in assessment_result for k in ["overall_assessment", "strategic_tool_recommendations"]):
                raise ValueError("LLM assessment missing required keys.")
            await self.bdi_agent.update_belief("mindx.tool_suite_assessment_result", assessment_result, 0.8, BeliefSource.SELF_ANALYSIS)
            logger.info(f"{self.log_prefix} Tool suite assessment completed. Overall: {assessment_result.get('overall_assessment')}")
            return True, assessment_result
        except Exception as e:
            logger.error(f"{self.log_prefix} LLM tool suite assessment failed: {e}", exc_info=True)
            return False, {"message": f"Tool assessment LLM error: {e}"}

    async def _bdi_action_conceptualize_new_tool(self, action_params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        logger.info(f"{self.log_prefix} BDI Action: Conceptualizing a new tool based on need: '{action_params.get('identified_need', 'N/A')}'.")
        if not self.llm_handler or not self.llm_handler.model_name_for_api: return False, {"message": "Mastermind LLM not available."}
        identified_need = action_params.get("identified_need", "a general-purpose utility or a tool to address a strategic gap")
        existing_tool_ids = list(self.tools_registry.get("registered_tools", {}).keys())
        prompt = (
            f"You are MastermindAgent's strategic LLM for mindX, tasked with conceptualizing new tools.\n"
            f"Identified Need/Gap: '{identified_need}'\n"
            f"Existing Tool IDs (for awareness, avoid direct duplication of core purpose unless versioning): {json.dumps(existing_tool_ids)}\n\n"
            f"Define a concept for a new tool. Provide the following details in a JSON object:\n"
            f"  - `tool_id`: A unique, Python-identifier-style string (e.g., 'code_analyzer_v1', 'agent_nft_minter_v0_1'). Must be unique.\n"
            f"  - `display_name`: A human-readable name (e.g., 'Python Code Analyzer', 'Agent NFT Minting Service').\n"
            f"  - `description`: What the tool does and its primary benefit to the mindX system.\n"
            f"  - `module_path`: Suggested Python module path (e.g., 'tools.code_analysis', 'tools.blockchain.nft').\n"
            f"  - `class_name`: Suggested Python class name for the tool (e.g., 'CodeAnalyzerTool', 'AgentNFTMinter').\n"
            f"  - `capabilities`: List of key functions/actions as dictionaries, each with 'name' (string, method name), 'description' (string), 'input_schema' (JSON schema dict), and 'output_schema' (JSON schema dict).\n"
            f"     Example capability: {{'name': 'analyze_python_file', 'description': 'Analyzes a Python file for complexity.', 'input_schema': {{'type':'object', 'properties':{{'file_path':{{'type':'string'}}}}}}, 'output_schema': {{'type':'object', 'properties':{{'complexity_score':{{'type':'number'}}}}}} }}\n"
            f"  - `initialization_params`: (Optional) List of dictionaries for __init__ params, each with 'name', 'type' (str, int, bool, Path), 'description', and optionally 'config_key'.\n"
            f"  - `needs_identity`: Boolean (true/false) - does this tool need its own cryptographic identity?\n"
            f"  - `initial_version`: e.g., '0.1.0-alpha'.\n"
            f"  - `initial_status`: 'experimental' or 'under_development'.\n"
            f"  - `prompt_template_for_llm_interaction`: A natural language template explaining how an LLM or agent should invoke this tool and its capabilities.\n"
            f"  - `metadata`: An object for additional tags, category, author etc. (e.g., {{\"category\": \"blockchain\", \"tags\": [\"nft\", \"identity\"]}})\n"
            f"Respond ONLY with the JSON object."
        )
        try:
            response_str = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=3000, temperature=0.5, json_mode=True)
            if not response_str: return False, {"message": "LLM returned empty response for new tool conceptualization."}
            tool_concept = json.loads(response_str)
            if not tool_concept.get("tool_id"): tool_concept["tool_id"] = f"conceptualized_tool_{str(uuid.uuid4())[:6]}"; logger.warning(f"LLM concept missing tool_id, assigned: {tool_concept['tool_id']}")
            required_keys = ["tool_id", "display_name", "description", "module_path", "class_name", "capabilities", "needs_identity", "initial_version", "initial_status", "prompt_template_for_llm_interaction", "metadata"]
            if not all(k in tool_concept for k in required_keys): missing = [k for k in required_keys if k not in tool_concept]; raise ValueError(f"LLM tool concept missing required keys: {missing}")
            await self.bdi_agent.update_belief(f"mindx.new_tool_concept.{tool_concept['tool_id']}", tool_concept, 0.85, BeliefSource.SELF_ANALYSIS)
            logger.info(f"{self.log_prefix} New tool conceptualized: ID '{tool_concept['tool_id']}', Name: '{tool_concept.get('display_name')}'")
            return True, tool_concept
        except Exception as e:
            logger.error(f"{self.log_prefix} LLM new tool conceptualization failed: {e}", exc_info=True)
            return False, {"message": f"New tool conceptualization LLM error: {e}"}

    async def _bdi_action_initiate_tool_coding_campaign(self, action_params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        logger.info(f"{self.log_prefix} BDI Action: Initiating tool coding campaign.")
        if not self.coordinator_agent: return False, {"message": "CoordinatorAgent not available."}
        tool_concept = action_params.get("tool_concept")
        campaign_type = action_params.get("campaign_type", "NEW_TOOL_DEVELOPMENT")
        if not isinstance(tool_concept, dict) or not tool_concept.get("tool_id"):
            return False, {"message": "Valid 'tool_concept' dictionary with 'tool_id' is required."}
        tool_id = tool_concept["tool_id"]
        target_module_path = tool_concept.get("module_path", f"tools.{tool_id}")
        target_file_path_str = f"{target_module_path.replace('.', '/')}.py"
        logger.info(f"{self.log_prefix} Attempting to {campaign_type.lower().replace('_',' ')} tool: {tool_id} (Target file hint: {target_file_path_str})")
        campaign_description = (
            f"Initiate {campaign_type.lower().replace('_', ' ')} for tool '{tool_id}' ({tool_concept.get('display_name', '')}).\n"
            f"Target file: {target_file_path_str}\n"
            f"Full Specification:\n{json.dumps(tool_concept, indent=2)}\n"
            f"Task SIA to generate/update the Python code. The tool should inherit from 'core.bdi_agent.BaseTool'. "
            f"After SIA completes, Mastermind will update the official tool registry."
        )
        interaction_meta = {
            "source": self.agent_id, "mastermind_campaign_goal": f"Tool Coding: {tool_id}",
            "analysis_context": campaign_description, "analysis_depth": "tool_development_kickoff",
            "target_component": target_file_path_str, "tool_specification_data": tool_concept,
            "is_new_tool_creation": (campaign_type == "NEW_TOOL_DEVELOPMENT")
        }
        interaction = await self.coordinator_agent.create_interaction(
            interaction_type=InteractionType.COMPONENT_IMPROVEMENT,
            content=f"Mastermind directive: {campaign_type} for tool '{tool_id}'. Target: {target_file_path_str}",
            agent_id=self.agent_id, metadata=interaction_meta
        )
        processed_interaction = await self.coordinator_agent.process_interaction(interaction)
        if processed_interaction.status == InteractionStatus.COMPLETED and \
           isinstance(processed_interaction.response, dict) and \
           processed_interaction.response.get("status") == "SUCCESS":
            logger.info(f"{self.log_prefix} SIA successfully tasked for tool '{tool_id}'. SIA response stored in belief.")
            await self.bdi_agent.update_belief(f"mindx.tool_coding_campaign.{tool_id}.sia_response", processed_interaction.response)
            return True, {"message": "SIA tasked for tool coding.", "sia_output": processed_interaction.response}
        else:
            err_msg = processed_interaction.error or (processed_interaction.response.get("message") if isinstance(processed_interaction.response, dict) else "Unknown error from Coordinator/SIA")
            logger.error(f"{self.log_prefix} Failed to task SIA for tool '{tool_id}': {err_msg}")
            return False, {"message": f"SIA tasking failed: {err_msg}"}

    async def _bdi_action_register_or_update_tool(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        tool_definition = params.get("tool_definition")
        update_type = params.get("update_type", "REGISTER_NEW")
        if not isinstance(tool_definition, dict) or not tool_definition.get("id"):
            return False, {"message": "Valid 'tool_definition' (dict with 'id') is required."}
        tool_id = tool_definition["id"]
        self.tools_registry.setdefault("registered_tools", {})
        msg = ""
        if update_type == "REGISTER_NEW":
            if tool_id in self.tools_registry["registered_tools"]:
                logger.warning(f"{self.log_prefix} Tool '{tool_id}' already exists. Switching to update mode for registration.")
                update_type = "UPDATE_EXISTING"
            else:
                entry = tool_definition.copy()
                entry["id"] = tool_id
                entry.setdefault("status", "experimental")
                entry.setdefault("version", "0.1.0-alpha")
                entry.setdefault("added_by", self.agent_id)
                entry.setdefault("added_at", time.time())
                entry.setdefault("performance_summary", {})
                entry.setdefault("last_assessment", {})
                for schema_key in ["display_name", "description", "module_path", "class_name", "capabilities", "initialization_params", "dependencies", "needs_identity", "identity_details", "prompt_template_for_llm_interaction", "metadata"]:
                    if schema_key not in entry and schema_key in tool_definition: entry[schema_key] = tool_definition[schema_key]
                    elif schema_key not in entry:
                        if schema_key in ["capabilities", "initialization_params", "dependencies"]: entry[schema_key] = []
                        elif schema_key == "metadata": entry[schema_key] = {}
                        elif schema_key == "needs_identity": entry[schema_key] = False
                        else: entry[schema_key] = None
                self.tools_registry["registered_tools"][tool_id] = entry
                msg = f"Tool '{tool_id}' registered in official registry."

        if update_type == "UPDATE_EXISTING":
            if tool_id not in self.tools_registry["registered_tools"]:
                return False, {"message": f"Tool '{tool_id}' not found for update."}
            existing_entry = self.tools_registry["registered_tools"][tool_id]
            for key, value in tool_definition.items():
                if key not in ["id", "added_by", "added_at"]:
                    if isinstance(existing_entry.get(key), dict) and isinstance(value, dict) and key in ["performance_summary", "last_assessment", "metadata", "identity_details"]:
                        existing_entry[key].update(value)
                    else: existing_entry[key] = value
            existing_entry["last_updated_by_mastermind_action"] = self.agent_id
            existing_entry["last_updated_at_mastermind_action"] = time.time()
            msg = f"Tool '{tool_id}' updated in official registry."
        if not msg: return False, {"message": "Unknown error in register_or_update_tool."}
        self._save_tools_registry()
        logger.info(f"{self.log_prefix} {msg}")
        return True, {"message": msg, "tool_id": tool_id}

    async def _bdi_action_mark_tool_status(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        tool_id = params.get("tool_id")
        new_status_str = params.get("new_status")
        reason = params.get("reason")
        if not tool_id or not new_status_str: return False, {"message": "'tool_id' and 'new_status' are required."}
        if tool_id not in self.tools_registry.get("registered_tools", {}): return False, {"message": f"Tool '{tool_id}' not found in registry."}
        valid_statuses = ["active", "experimental", "deprecated", "under_development", "archived", "failed_development"]
        if new_status_str not in valid_statuses:
            logger.warning(f"{self.log_prefix} Invalid status '{new_status_str}' for tool '{tool_id}'. Must be one of {valid_statuses}.")
            return False, {"message": f"Invalid tool status '{new_status_str}'."}
        entry = self.tools_registry["registered_tools"][tool_id]
        old_status = entry.get("status")
        entry["status"] = new_status_str
        entry["last_status_update_at"] = time.time()
        entry["last_status_updated_by"] = self.agent_id
        if new_status_str.lower() == "deprecated" and reason: entry["deprecation_reason"] = reason
        elif new_status_str.lower() != "deprecated": entry.pop("deprecation_reason", None)
        self._save_tools_registry()
        msg = f"Status of tool '{tool_id}' updated from '{old_status}' to '{new_status_str}' in registry."
        logger.info(f"{self.log_prefix} {msg}")
        return True, {"message": msg, "tool_id": tool_id, "new_status": new_status_str}

    async def _bdi_action_execute_tool(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        tool_id = params.get("tool_id")
        capability_name = params.get("capability_name")
        tool_specific_params = params.get("tool_specific_params", {})
        if not tool_id or not capability_name: return False, "EXECUTE_TOOL requires 'tool_id' and 'capability_name'."
        tool_entry = self.tools_registry.get("registered_tools", {}).get(tool_id)
        if not tool_entry or tool_entry.get("status") != "active":
            msg = f"Tool '{tool_id}' not found in registry or is not active."; logger.warning(f"{self.log_prefix} {msg}"); return False, msg
        logger.info(f"{self.log_prefix} Attempting to execute tool '{tool_id}', capability '{capability_name}' with params: {tool_specific_params}")
        try:
            module_path = tool_entry.get("module_path")
            class_name = tool_entry.get("class_name")
            if not module_path or not class_name: return False, f"Tool '{tool_id}' registry entry missing module_path or class_name."
            module = importlib.import_module(module_path)
            ToolClass = getattr(module, class_name)
            tool_instance = ToolClass(config=self.config, llm_handler=self.llm_handler, bdi_agent_ref=self.bdi_agent)
        except Exception as e_load: logger.error(f"{self.log_prefix} Error loading/instantiating tool '{tool_id}': {e_load}", exc_info=True); return False, f"Error loading tool '{tool_id}': {e_load}"

        if hasattr(tool_instance, capability_name) and callable(getattr(tool_instance, capability_name)):
            try:
                method_to_call = getattr(tool_instance, capability_name)
                if asyncio.iscoroutinefunction(method_to_call): result = await method_to_call(**tool_specific_params)
                else: loop = asyncio.get_running_loop(); result = await loop.run_in_executor(None, lambda: method_to_call(**tool_specific_params))
                logger.info(f"{self.log_prefix} Tool '{tool_id}' capability '{capability_name}' executed. Result preview: {str(result)[:100]}")
                await self.bdi_agent.update_belief(f"tool_execution_results.{tool_id}.{capability_name}", result, 0.9, BeliefSource.SELF_ANALYSIS)
                return True, result
            except Exception as e_exec: logger.error(f"{self.log_prefix} Error executing tool '{tool_id}' capability '{capability_name}': {e_exec}", exc_info=True); return False, f"Error during tool '{tool_id}' execution: {str(e_exec)[:200]}"
        else: msg = f"Capability '{capability_name}' not found or not callable on tool '{tool_id}'."; logger.warning(f"{self.log_prefix} {msg}"); return False, msg

    async def _bdi_action_analyze_codebase_for_strategy(self, params: Dict[str, Any], **kwargs) -> Tuple[bool, Any]: # pragma: no cover
        logger.info(f"{self.log_prefix} BDI Action: Analyzing codebase for strategy.")
        if not self.code_base_analyzer: return False, {"message": "CodeBaseGenerator tool is not available to Mastermind."}
        if not self.llm_handler or not self.llm_handler.model_name_for_api: return False, {"message": "Mastermind LLM not available for interpreting analysis."}

        target_path_str = params.get("target_path")
        analysis_focus = params.get("analysis_focus", "Identify key modules, functionalities, and potential areas for tool integration or improvement.")
        output_belief_key_suffix = params.get("output_belief_key_suffix", Path(target_path_str).name if target_path_str else "generic_codebase")

        if not target_path_str: return False, {"message": "'target_path' parameter is required."}
        target_path = Path(target_path_str)
        if not target_path.is_absolute(): target_path = (PROJECT_ROOT / target_path).resolve()
        if not target_path.exists(): return False, {"message": f"Target path '{target_path}' does not exist."}

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_md_filename = f"code_analysis_{target_path.name}_{timestamp}.md"
        output_md_path = self.data_dir / "codebase_analyses" / output_md_filename
        output_md_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"{self.log_prefix} Running CodeBaseGenerator on '{target_path}' -> '{output_md_path}'")
            gen_result = self.code_base_analyzer.generate_markdown_summary(
                root_path_str=str(target_path),
                output_file_str=str(output_md_path),
                include_patterns=params.get("include_patterns"),
                user_exclude_patterns=params.get("user_exclude_patterns"),
                use_gitignore=params.get("use_gitignore", True)
            )
            if gen_result.get("status") != "SUCCESS":
                return False, {"message": f"CodeBaseGenerator failed: {gen_result.get('message')}"}
            logger.info(f"{self.log_prefix} Codebase summary generated: {output_md_path}")

            with output_md_path.open("r", encoding="utf-8") as f: summary_content = f.read()
            max_summary_len = self.config.get("mastermind_agent.code_analysis_max_prompt_len", 15000)
            summary_content_for_prompt = summary_content[:max_summary_len] + "\n... (summary truncated) ..." if len(summary_content) > max_summary_len else summary_content

            prompt_for_interpretation = (
                f"You are MastermindAgent's strategic LLM. A codebase analysis summary has been generated for '{target_path_str}'.\n"
                f"Your analysis focus is: '{analysis_focus}'\n\n"
                f"Codebase Summary (Markdown):\n{summary_content_for_prompt}\n\n"
                f"Based on this summary and your focus, provide a concise interpretation. "
                f"Highlight key findings relevant to the focus. For example, if focus is 'identify core APIs', list them. "
                f"If focus is 'tool integration', suggest how mindX tools could interface or what new tools could be built from this. "
                f"Output your interpretation as a structured JSON object, with a main 'interpretation_summary' (string) and "
                f"any specific 'identified_elements' (list of strings or dicts)."
            )
            llm_interpretation_str = await self.llm_handler.generate_text(
                prompt_for_interpretation, model=self.llm_handler.model_name_for_api,
                max_tokens=1024, temperature=0.3, json_mode=True
            )
            if not llm_interpretation_str: return False, {"message": "LLM failed to interpret codebase summary."}
            llm_interpretation = json.loads(llm_interpretation_str)
            analysis_data_to_store = {
                "target_path": str(target_path), "analysis_focus": analysis_focus,
                "markdown_summary_path": str(output_md_path.relative_to(PROJECT_ROOT)),
                "llm_interpretation": llm_interpretation, "timestamp": time.time()
            }
            belief_key = f"codebase_analysis.{output_belief_key_suffix.replace('.', '_')}"
            await self.bdi_agent.update_belief(belief_key, analysis_data_to_store, 0.9, BeliefSource.SELF_ANALYSIS)
            return True, {"message": f"Codebase '{target_path_str}' analyzed. Summary at '{output_md_path}'. Interpretation stored in belief '{belief_key}'.", "analysis": analysis_data_to_store}
        except Exception as e:
            logger.error(f"{self.log_prefix} Codebase analysis failed for '{target_path_str}': {e}", exc_info=True)
            return False, {"message": f"Codebase analysis error: {e}"}

    async def manage_mindx_evolution(self, top_level_directive: str, max_mastermind_bdi_cycles: int = 15) -> Dict[str, Any]: # pragma: no cover
        if not self._initialized_async:
            logger.warning(f"{self.log_prefix} Mastermind async components for mindX not initialized. Attempting now.")
            await self._async_init_components()
            if not self._initialized_async:
                logger.error(f"{self.log_prefix} Mastermind async initialization for mindX failed. Cannot manage evolution.")
                return {"error": "mindX Mastermind not fully initialized.", "overall_campaign_status": "FAILURE_OR_INCOMPLETE"}

        if self.bdi_agent and not self.bdi_agent._initialized:
            logger.warning(f"{self.log_prefix} BDI agent async components not initialized. Attempting now.")
            await self.bdi_agent.async_init_components()
            if not self.bdi_agent._initialized:
                logger.error(f"{self.log_prefix} BDI agent async initialization failed. Cannot manage evolution via BDI.")
                return {"error": "mindX BDI Agent not fully initialized for Mastermind.", "overall_campaign_status": "FAILURE_OR_INCOMPLETE"}

        self._internal_state["current_run_id"] = f"mastermind_run_{str(uuid.uuid4())[:8]}"
        run_id = self._internal_state["current_run_id"]
        logger.info(f"{self.log_prefix} Starting mindX (Augmentic Intelligence) evolution campaign (Run ID: {run_id}). Directive: '{top_level_directive}'")
        await self.bdi_agent.update_belief(f"mastermind.current_campaign.directive", top_level_directive, 0.99, BeliefSource.EXTERNAL_INPUT)
        await self.bdi_agent.update_belief(f"mastermind.current_campaign.run_id", run_id, 0.99, BeliefSource.SELF_ANALYSIS)

        existing_objective = next((obj for obj in self.high_level_objectives if obj.get("directive") == top_level_directive and obj.get("status") == "active"), None)
        if not existing_objective:
            objective_id = str(uuid.uuid4())[:8]
            self.high_level_objectives.append({"id": objective_id, "directive": top_level_directive, "status": "active", "created_at": time.time(), "run_ids": [run_id]})
        elif run_id not in existing_objective.get("run_ids",[]): # pragma: no cover
            existing_objective.setdefault("run_ids",[]).append(run_id)
        self._save_json_file("mastermind_objectives.json", self.high_level_objectives)

        self.bdi_agent.set_goal(goal_description=f"Strategically fulfill Mastermind directive for mindX: {top_level_directive}", priority=1, is_primary=True, goal_id=f"mastermind_directive_{run_id}")
        bdi_final_run_message = await self.bdi_agent.run(max_cycles=max_mastermind_bdi_cycles)
        bdi_final_agent_status = self.bdi_agent._internal_state["status"]
        
        campaign_successful = False
        if bdi_final_agent_status == "COMPLETED_GOAL_ACHIEVED":
             campaign_successful = True
        else:
             try:
                 if BDIGoalStatus(bdi_final_agent_status) == BDIGoalStatus.COMPLETED_SUCCESS:
                     campaign_successful = True
             except ValueError:
                 campaign_successful = False

        campaign_outcome_summary = {
            "mastermind_run_id": run_id, "agent_id": self.agent_id, "directive": top_level_directive,
            "bdi_final_status": bdi_final_agent_status, "bdi_outcome_message": bdi_final_run_message,
            "overall_campaign_status": "SUCCESS" if campaign_successful else "FAILURE_OR_INCOMPLETE",
            "timestamp": time.time()
        }
        self.strategic_campaigns_history.append(campaign_outcome_summary)
        self._save_json_file("mastermind_campaigns_history.json", self.strategic_campaigns_history)

        for obj in self.high_level_objectives:
            if obj.get("directive") == top_level_directive and obj.get("status") == "active":
                obj["status"] = "completed_attempt" if campaign_successful else "failed_attempt"
                obj["last_run_id"] = run_id
                obj["last_run_status"] = campaign_outcome_summary['overall_campaign_status']
                break
        self._save_json_file("mastermind_objectives.json", self.high_level_objectives)

        await self.bdi_agent.update_belief(f"mastermind.campaign_history.{run_id}", campaign_outcome_summary, 0.95, BeliefSource.SELF_ANALYSIS)
        logger.info(f"{self.log_prefix} mindX (Augmentic Intelligence) evolution campaign (Run ID: {run_id}) finished. BDI Status: {bdi_final_agent_status}. Overall: {campaign_outcome_summary['overall_campaign_status']}")
        return campaign_outcome_summary

    def start_autonomous_loop(self, interval_seconds: Optional[float] = None): # pragma: no cover
        if self.autonomous_loop_task and not self.autonomous_loop_task.done(): logger.warning(f"{self.log_prefix} mindX autonomous loop already running."); return
        loop_interval = interval_seconds or self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.interval_seconds", 3600 * 4)
        default_directive = self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.default_directive", "Proactively monitor and evolve mindX based on Augmentic Intelligence principles, including its toolset and codebase.")
        max_bdi_cycles = self.config.get(f"mastermind_agent.{self.agent_id}.autonomous_loop.max_bdi_cycles", 25)
        if loop_interval <= 0: logger.error(f"{self.log_prefix} Invalid autonomous interval for mindX: {loop_interval}."); return
        self.autonomous_loop_task = asyncio.create_task(self._mastermind_autonomous_worker(loop_interval, default_directive, max_bdi_cycles))
        logger.info(f"{self.log_prefix} mindX autonomous loop started. Interval: {loop_interval}s.")

    async def _mastermind_autonomous_worker(self, interval: float, default_directive: str, max_bdi_cycles: int): # pragma: no cover
        logger.info(f"{self.log_prefix} mindX autonomous worker started with directive: '{default_directive[:100]}...'")
        while True:
            try:
                if not self._initialized_async:
                    await self._async_init_components()
                    if not self._initialized_async: logger.error(f"{self.log_prefix} Mastermind Async init failed in worker. Skipping cycle."); await asyncio.sleep(interval); continue
                if self.bdi_agent and not self.bdi_agent._initialized:
                    await self.bdi_agent.async_init_components()
                    if not self.bdi_agent._initialized: logger.error(f"{self.log_prefix} BDI Async init failed in worker. Skipping cycle."); await asyncio.sleep(interval); continue

                logger.info(f"{self.log_prefix} Autonomous worker for mindX: Sleeping for {interval}s.")
                await asyncio.sleep(interval)
                logger.info(f"{self.log_prefix} Autonomous worker for mindX: Initiating new strategic campaign.")
                await self.manage_mindx_evolution(default_directive, max_mastermind_bdi_cycles=max_bdi_cycles)
            except asyncio.CancelledError: logger.info(f"{self.log_prefix} mindX autonomous worker stopping."); break
            except Exception as e: logger.error(f"{self.log_prefix} mindX autonomous worker error: {e}", exc_info=True); await asyncio.sleep(interval)
        logger.info(f"{self.log_prefix} mindX autonomous worker has stopped.")

    async def shutdown(self): # pragma: no cover
        logger.info(f"MastermindAgent '{self.agent_id}' for mindX shutting down...")
        if self.autonomous_loop_task and not self.autonomous_loop_task.done():
            self.autonomous_loop_task.cancel()
            try: await asyncio.wait_for(self.autonomous_loop_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError): pass
        if self.bdi_agent: await self.bdi_agent.shutdown()
        if self.id_manager_agent and hasattr(self.id_manager_agent, 'shutdown'):
            shutdown_method = self.id_manager_agent.shutdown
            if asyncio.iscoroutinefunction(shutdown_method): await shutdown_method()
            elif callable(shutdown_method): shutdown_method()

        self._save_json_file("mastermind_campaigns_history.json", self.strategic_campaigns_history)
        self._save_json_file("mastermind_objectives.json", self.high_level_objectives)
        self._save_tools_registry()
        logger.info(f"MastermindAgent '{self.agent_id}' for mindX shutdown complete.")

    @classmethod
    async def reset_singleton_instance_for_testing(cls): # pragma: no cover
        async with cls._lock:
            if cls._instance:
                logger.debug(f"Resetting mindX MastermindAgent singleton '{cls._instance.agent_id}'.")
                await cls._instance.shutdown()
                cls._instance = None
            else: logger.debug("No active mindX MastermindAgent singleton to reset.")
        logger.debug("mindX MastermindAgent singleton reset.")
