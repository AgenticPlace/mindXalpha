# mindx/evolution/blueprint_agent.py
"""
BlueprintAgent for MindX Strategic Evolution Planning.

This agent analyzes the current state of the MindX system and uses an LLM
to propose a strategic blueprint for the next iteration of MindX's own
self-improvement and development.
"""

import logging
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from mindx.core.belief_system import BeliefSystem, BeliefSource # For context
from mindx.llm.llm_factory import create_llm_handler, LLMHandler
# To access other agents' states or capabilities (conceptually)
from mindx.orchestration.coordinator_agent import CoordinatorAgent # To get backlog, history
from mindx.learning.self_improve_agent import SELF_AGENT_FILENAME as SIA_FILENAME # To reference SIA
from mindx.learning.strategic_evolution_agent import StrategicEvolutionAgent # If BlueprintAgent needs to know about SEA

logger = get_logger(__name__)

class BlueprintAgent:
    """
    Generates a strategic blueprint for the next iteration of MindX's development
    and self-improvement capabilities.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs): # pragma: no cover
        if not cls._instance:
            cls._instance = super(BlueprintAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 belief_system: BeliefSystem,
                 coordinator_ref: CoordinatorAgent, # Reference to the live Coordinator
                 config_override: Optional[Config] = None,
                 test_mode: bool = False): # pragma: no cover
        
        if hasattr(self, '_initialized') and self._initialized and not test_mode:
            return

        self.config = config_override or Config()
        self.agent_id: str = self.config.get("evolution.blueprint_agent.agent_id", "blueprint_agent_mindx_v1")
        self.belief_system = belief_system
        self.coordinator_ref = coordinator_ref # To access backlog, campaign history etc.
        
        # LLM for this agent's strategic blueprinting
        bp_llm_provider = self.config.get(f"evolution.blueprint_agent.llm.provider", self.config.get("llm.default_provider"))
        bp_llm_model = self.config.get(f"evolution.blueprint_agent.llm.model", self.config.get(f"llm.{bp_llm_provider}.default_model_for_reasoning"))
        self.llm_handler: LLMHandler = create_llm_handler(bp_llm_provider, bp_llm_model)

        logger.info(f"BlueprintAgent '{self.agent_id}' initialized. LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default'}")
        self._initialized = True

    async def _gather_mindx_system_state_summary(self) -> Dict[str, Any]: # pragma: no cover
        """Gathers a summary of the current MindX system state for analysis."""
        summary = {}

        # 1. Capabilities (from Coordinator's cache or rescan)
        if self.coordinator_ref.system_capabilities_cache:
            summary["num_scanned_capabilities"] = len(self.coordinator_ref.system_capabilities_cache)
            summary["example_module_paths"] = list(set(c.get("module") for c in self.coordinator_ref.system_capabilities_cache.values()))[:5]
        else: # Fallback: ask coordinator to scan if cache is empty
            caps = await self.coordinator_ref._scan_codebase_capabilities() # Call protected for direct access
            summary["num_scanned_capabilities"] = len(caps)
            summary["example_module_paths"] = list(set(c.get("module") for c in caps.values()))[:5]

        # 2. Improvement Backlog Summary (from Coordinator)
        backlog = self.coordinator_ref.improvement_backlog # Direct access for this agent
        summary["backlog_total_items"] = len(backlog)
        summary["backlog_pending_items"] = len([item for item in backlog if item.get("status") == "PENDING"])
        summary["backlog_pending_approval_items"] = len([item for item in backlog if item.get("status") == "PENDING_APPROVAL"])
        if backlog:
            summary["backlog_top_pending_priorities"] = sorted(list(set(item.get("priority",0) for item in backlog if item.get("status") == "PENDING")), reverse=True)[:3]

        # 3. Recent Campaign History (from Coordinator)
        campaign_history = self.coordinator_ref.improvement_campaign_history
        summary["recent_campaigns_count"] = len(campaign_history)
        if campaign_history:
            summary["last_campaign_outcomes"] = [
                {
                    "target": entry.get("target_component_id"), 
                    "sia_status": entry.get("status_from_sia_json"),
                    "final_op_status": entry.get("final_sia_op_status")
                }
                for entry in campaign_history[-3:] # Last 3 campaigns
            ]
        
        # 4. Monitoring Summaries (from Monitors via Coordinator's access)
        if self.coordinator_ref.resource_monitor:
            res_usage = self.coordinator_ref.resource_monitor.get_resource_usage()
            summary["resource_monitor_snapshot"] = {
                "cpu": f"{res_usage.get('cpu_percent',0):.1f}%",
                "memory": f"{res_usage.get('memory_percent',0):.1f}%"
            }
        if self.coordinator_ref.performance_monitor:
            perf_report = self.coordinator_ref.performance_monitor.get_all_metrics() # Get structured data
            total_llm_requests = sum(m.get("requests",0) for m in perf_report.values())
            summary["performance_monitor_snapshot"] = {
                "tracked_llm_metric_keys": len(perf_report),
                "total_llm_requests_tracked": total_llm_requests
            }
            # Identify if any LLM key has very low success rate
            low_sr_models = [key for key, metrics in perf_report.items() if metrics.get("requests",0) > 10 and metrics.get("success_rate", 1.0) < 0.7]
            if low_sr_models: summary["performance_monitor_snapshot"]["low_success_rate_llms"] = low_sr_models


        # 5. Known Limitations / TODOs (could be from a dedicated belief or a file)
        # For this stub, we'll imagine this comes from beliefs
        # conceptual_todos = await self.belief_system.query_beliefs(partial_key="mindx.system.todo", min_confidence=0.7)
        # summary["known_todos_or_limitations"] = [b.value for b in conceptual_todos[:5]]
        summary["conceptual_known_limitations"] = [
            "SIA evaluation relies heavily on LLM critique, needs more functional testing.",
            "System restart required for self-updates of Coordinator/SIA to take effect.",
            "Multi-file refactoring is not yet supported by SIA.",
            "Peripheral agents (MMA, BDI-general, Docs) are stubs."
        ]
        
        # 6. Core Agent Versions (conceptual)
        summary["core_agent_versions"] = {
            "CoordinatorAgent": self.config.get("system.agents.coordinator.version", "v_prod_cand_final"),
            "SelfImprovementAgent": self.config.get("system.agents.sia.version", "v3.5_cli_focused"),
            "StrategicEvolutionAgent": self.config.get("system.agents.sea.version", "v1.0_bdi_driven")
        }

        return summary

    async def generate_next_evolution_blueprint(
            self, 
            current_mindx_version: str,
            high_level_directive: Optional[str] = None, # e.g., "Focus on safety", "Improve performance"
            look_ahead_iterations: int = 1 # How many major iterations to plan for
        ) -> Dict[str, Any]: # Returns the blueprint
        """
        Generates a blueprint for the next self-improvement iteration(s) of MindX.
        """
        logger.info(f"{self.agent_id}: Generating next evolution blueprint for MindX v{current_mindx_version}. Directive: {high_level_directive or 'General Evolution'}")

        system_state_summary = await self._gather_mindx_system_state_summary()
        
        prompt_parts = [
            f"You are a Chief Architect AI for the MindX Self-Improving System (Augmentic Project), currently at version {current_mindx_version}.",
            "Your task is to define a strategic blueprint for the NEXT {look_ahead_iterations} major evolution iteration(s) of MindX itself.",
            "Consider the current system state and the overall goal of creating more autonomous, robust, and capable self-improving AI.",
            f"Current MindX System State Summary:\n{json.dumps(system_state_summary, indent=2)}",
        ]
        if high_level_directive: # pragma: no cover
            prompt_parts.append(f"\nA specific high-level directive for this blueprint is: '{high_level_directive}'")

        prompt_parts.append(
            "\nBlueprint Requirements:\n"
            "1. Identify 2-4 key strategic **Focus Areas** for the next iteration(s). Examples: 'Enhanced SIA Evaluation', 'Advanced Planning for Coordinator', 'Multi-File Refactoring Capability', 'Improved System Resilience & Rollback'.\n"
            "2. For each Focus Area, define 1-3 specific, actionable **Development Goals**. These goals should be achievable within one iteration and ideally target improvements to specific MindX agents or core utilities.\n"
            "   - Each goal should specify: 'target_component_module_path' (e.g., 'mindx.learning.self_improve_agent'), a 'description' of the change, and a 'justification' linking it to the focus area.\n"
            "3. Suggest 1-2 **Key Performance Indicators (KPIs)** or metrics that could be used to measure the success of this evolution iteration.\n"
            "4. Briefly note any **Potential Risks or Challenges** associated with pursuing this blueprint.\n\n"
            "Respond ONLY with a single, valid JSON object. The top-level keys should be: "
            "\"blueprint_title\" (string), \"target_mindx_version_increment\" (string, e.g., \"+0.1.0\"), "
            "\"focus_areas\" (list of objects, each with 'area_title' and 'development_goals' list), "
            "\"key_performance_indicators\" (list of strings), "
            "\"potential_risks_challenges\" (list of strings)."
        )
        prompt = "\n".join(prompt_parts).format(look_ahead_iterations=look_ahead_iterations)

        logger.debug(f"{self.agent_id}: Blueprint generation prompt (first 500 chars): {prompt[:500]}...")
        
        blueprint_json: Optional[Dict[str, Any]] = None
        try:
            max_tokens = self.config.get("evolution.blueprint_agent.llm.max_tokens", 2500)
            temperature = self.config.get("evolution.blueprint_agent.llm.temperature", 0.3)
            response_str = await self.llm_handler.generate_text(prompt, max_tokens=max_tokens, temperature=temperature, json_mode=True)

            if response_str and not response_str.startswith("Error:"):
                try: blueprint_json = json.loads(response_str)
                except json.JSONDecodeError: # pragma: no cover
                    match = re.search(r"\{[\s\S]*\}", response_str)
                    if match: blueprint_json = json.loads(match.group(0))
                    else: logger.error(f"{self.agent_id}: LLM blueprint response not valid JSON. Raw: {response_str[:300]}"); raise ValueError("Blueprint LLM response not JSON")
            else: # pragma: no cover
                logger.error(f"{self.agent_id}: LLM blueprint generation failed or returned empty: {response_str}")
                raise ValueError(f"Blueprint LLM generation failed: {response_str}")

            # Validate blueprint structure (basic check)
            if not blueprint_json or not all(k in blueprint_json for k in ["blueprint_title", "focus_areas"]): # pragma: no cover
                logger.error(f"{self.agent_id}: Generated blueprint missing essential keys. Blueprint: {str(blueprint_json)[:500]}")
                raise ValueError("Generated blueprint missing essential keys.")
            
            logger.info(f"{self.agent_id}: Successfully generated evolution blueprint titled '{blueprint_json.get('blueprint_title', 'Untitled Blueprint')}'.")
            # Persist blueprint to BeliefSystem
            await self.belief_system.add_belief(
                f"mindx.evolution.blueprint.latest",
                blueprint_json, 0.95, BeliefSource.SELF_ANALYSIS,
                metadata={"generated_at": time.time(), "mindx_version_input": current_mindx_version, "directive": high_level_directive}
            )
            return blueprint_json

        except Exception as e: # pragma: no cover
            logger.error(f"{self.agent_id}: Exception during blueprint generation: {e}", exc_info=True)
            return {"error": f"Blueprint generation exception: {type(e).__name__}: {e}", "blueprint_title": "Error Blueprint"}

    async def shutdown(self): # pragma: no cover
        logger.info(f"BlueprintAgent '{self.agent_id}' shutting down.")
        # No specific async tasks to cancel in this agent for now.

    @classmethod
    async def reset_instance_async(cls): # For testing # pragma: no cover
        async with cls._lock:
            if cls._instance:
                if hasattr(cls._instance, "shutdown") and asyncio.iscoroutinefunction(cls._instance.shutdown):
                    await cls._instance.shutdown()
                cls._instance._initialized = False 
                cls._instance = None
        logger.debug("BlueprintAgent instance reset asynchronously.")

# --- Factory Functions ---
async def get_blueprint_agent_async(
    belief_system: BeliefSystem, 
    coordinator_ref: CoordinatorAgent, # Needs live coordinator
    config_override: Optional[Config] = None, 
    test_mode: bool = False
) -> BlueprintAgent: # pragma: no cover
    """Asynchronously gets or creates the BlueprintAgent singleton instance."""
    # This agent is more stateful with coordinator_ref, so test_mode should always create new if instance exists.
    if not BlueprintAgent._instance or test_mode:
        async with BlueprintAgent._lock:
            if BlueprintAgent._instance is None or test_mode:
                if test_mode and BlueprintAgent._instance is not None:
                    await BlueprintAgent._instance.shutdown()
                    BlueprintAgent._instance = None
                BlueprintAgent._instance = BlueprintAgent(
                    belief_system=belief_system, 
                    coordinator_ref=coordinator_ref,
                    config_override=config_override, 
                    test_mode=test_mode)
    return BlueprintAgent._instance

# --- Example Usage (Conceptual, typically called by Coordinator or a main loop) ---
async def example_run_blueprint_agent(): # pragma: no cover
    # This requires Coordinator to be up and running first
    from mindx.orchestration.coordinator_agent import get_coordinator_agent_mindx_async

    config = Config(test_mode=True); Config.reset_instance(); config=Config(test_mode=True) # Fresh config
    shared_bs = BeliefSystem(test_mode=True); BeliefSystem.reset_instance(); shared_bs=BeliefSystem(test_mode=True)
    
    logger.info("BlueprintAgent Example: Initializing Coordinator...")
    # Need to ensure coordinator factory can also run in test_mode for its singletons
    try:
        # Ensure Coordinator factory is called in a way that respects test_mode for its own singletons if necessary
        coordinator = await get_coordinator_agent_mindx_async(config_override=config, test_mode=True)
        logger.info("BlueprintAgent Example: Coordinator initialized.")
    except Exception as e:
        logger.error(f"BlueprintAgent Example: Failed to initialize Coordinator: {e}", exc_info=True)
        return

    logger.info("BlueprintAgent Example: Initializing BlueprintAgent...")
    bp_agent = await get_blueprint_agent_async(belief_system=shared_bs, coordinator_ref=coordinator, config_override=config, test_mode=True)
    logger.info("BlueprintAgent Example: BlueprintAgent initialized.")

    current_version = config.get("system.version", "0.4.0") # Get current MindX version
    directive = "Enhance the safety and robustness of the SelfImprovementAgent's self-update process."

    print(f"\n--- Generating Blueprint for MindX v{current_version} ---")
    print(f"Directive: {directive}\n")

    blueprint = await bp_agent.generate_next_evolution_blueprint(
        current_mindx_version=current_version,
        high_level_directive=directive,
        look_ahead_iterations=1
    )

    print("\n--- Generated Evolution Blueprint ---")
    print(json.dumps(blueprint, indent=2))
    print("-" * 50)

    # The blueprint (JSON) would then be used by developers or a higher-level
    # meta-agent to guide the *next actual development or self-improvement tasks*
    # for MindX. For example, the 'development_goals' could be added to the
    # Coordinator's improvement_backlog.
    if "focus_areas" in blueprint:
        for area in blueprint["focus_areas"]:
            print(f"\nFocus Area: {area.get('area_title')}")
            for goal in area.get("development_goals",[]):
                print(f"  - Goal: {goal.get('description')}")
                print(f"    Target: {goal.get('target_component_module_path')}")
                print(f"    Justification: {goal.get('justification')}")
                # Example: coordinator.add_to_improvement_backlog(goal_from_blueprint, source="blueprint_agent")

    await bp_agent.shutdown()
    await coordinator.shutdown()
    # Also shutdown monitors explicitly if not handled by coordinator's shutdown in test mode
    res_mon = await get_resource_monitor_async(test_mode=True); perf_mon = await get_performance_monitor_async(test_mode=True)
    if res_mon.monitoring: res_mon.stop_monitoring()
    await perf_mon.shutdown()


if __name__ == "__main__": # pragma: no cover
    # Ensure .env is loaded
    project_r_main = Path(__file__).resolve().parent.parent.parent
    env_p_main = project_r_main / ".env"
    if env_p_main.exists(): from dotenv import load_dotenv; load_dotenv(dotenv_path=env_p_main, override=True)
    else: print(f"BlueprintAgent Main: .env not found at {env_p_main}. Using defaults/env vars.", file=sys.stderr)
    
    try:
        asyncio.run(example_run_blueprint_agent())
    except KeyboardInterrupt:
        logger.info("BlueprintAgent example interrupted by user.")
    finally:
        logger.info("BlueprintAgent example finished.")
