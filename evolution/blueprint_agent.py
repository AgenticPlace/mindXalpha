# mindx/evolution/blueprint_agent.py
"""
BlueprintAgent for MindX Strategic Evolution Planning.

This agent analyzes the current state of the mindX system and uses an LLM
to propose a strategic blueprint for the next iteration of MindX's own
self-improvement and development.

This module also supports direct CLI execution for generating blueprints.
To see available command-line options, run:
    python -m mindx.evolution.blueprint_agent --help
(If running from the project root and `mindx` is in PYTHONPATH or installed) or:
    python evolution/blueprint_agent.py --help
(If running the script file directly from the project root).
"""

import logging
import asyncio
import json
import time
import re
import sys
import argparse # Added for CLI argument parsing
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

from utils.config import Config, PROJECT_ROOT
from utils.logging_config import get_logger
from core.belief_system import BeliefSystem, BeliefSource # For context
from llm.llm_factory import create_llm_handler, LLMHandler
# To access other agents' states or capabilities (conceptually)
from orchestration.coordinator_agent import CoordinatorAgent # To get backlog, history
from learning.self_improve_agent import SELF_AGENT_FILENAME as SIA_FILENAME # To reference SIA
from learning.strategic_evolution_agent import StrategicEvolutionAgent # If BlueprintAgent needs to know about SEA

logger = get_logger(__name__)

class BlueprintAgent:
    """
    Generates a strategic blueprint for the next iteration of MindX's development
    and self-improvement capabilities.

    Configuration:
        The BlueprintAgent uses the following configuration keys from the main application
        configuration (`Config` object):

        - `evolution.blueprint_agent.agent_id` (str):
            The unique identifier for this agent.
            Default: "blueprint_agent_mindx_v1"

        - `evolution.blueprint_agent.llm.provider` (str):
            The LLM provider to use for generating blueprints. If not specified,
            it falls back to the value of `llm.default_provider`.

        - `evolution.blueprint_agent.llm.model` (str):
            The specific LLM model to use for blueprint generation. If not specified,
            it falls back to the default reasoning model for the selected provider,
            which is determined by `llm.{provider}.default_model_for_reasoning`
            (e.g., `llm.openai.default_model_for_reasoning`).

        - `evolution.blueprint_agent.llm.max_tokens` (int):
            The maximum number of tokens to generate for the blueprint response.
            Default: 2500

        - `evolution.blueprint_agent.llm.temperature` (float):
            The temperature setting for the LLM during blueprint generation,
            controlling the randomness of the output.
            Default: 0.3

        The following conceptual configuration keys are used for context gathering within
        the `_gather_mindx_system_state_summary` method or in example runs, reflecting
        the versions of different parts of the MindX system. These versions are
        "reported" or "believed" versions, not necessarily derived programmatically
        by `BlueprintAgent` from code/metadata, but used to inform the LLM.

        - `system.agents.coordinator.version` (str):
            Specifies the conceptual or reported version of the CoordinatorAgent.
            This version string is included in the system state summary provided
            to the LLM, offering context about the perceived maturity or iteration
            of this core component. It is not programmatically derived from the
            CoordinatorAgent's code by `BlueprintAgent`.
            Default: If not set in the configuration, `BlueprintAgent` uses the
            hardcoded fallback string "v_prod_cand_final" when constructing
            the summary in `_gather_mindx_system_state_summary`.

        - `system.agents.sia.version` (str):
            Specifies the conceptual or reported version of the SelfImprovementAgent.
            This information is part of the system context for the LLM.
            Default: If not set in the configuration, `BlueprintAgent` uses the
            hardcoded fallback "v3.5_cli_focused" in `_gather_mindx_system_state_summary`.

        - `system.agents.sea.version` (str):
            Specifies the conceptual or reported version of the StrategicEvolutionAgent.
            This information is part of the system context for the LLM.
            Default: If not set in the configuration, `BlueprintAgent` uses the
            hardcoded fallback "v1.0_bdi_driven" in `_gather_mindx_system_state_summary`.

        - `system.version` (str):
            Specifies the overall conceptual or reported version of the MindX system.
            This is used, for example, as the `current_mindx_version` parameter
            in `example_run_blueprint_agent` and helps the LLM understand the
            current baseline of the entire system.
            Default: If not set in the configuration, the `example_run_blueprint_agent`
            function uses the hardcoded fallback "0.4.0".

        Note on LLM Fallbacks:
        The agent relies on global LLM configuration for fallbacks:
        - `llm.default_provider`: Used if `evolution.blueprint_agent.llm.provider` is not set.
        - `llm.{provider}.default_model_for_reasoning`: Used if `evolution.blueprint_agent.llm.model`
          is not set, where `{provider}` is the determined LLM provider.

    Data Dependencies on CoordinatorAgent:
        The `BlueprintAgent` relies on `coordinator_ref` (an instance of
        `CoordinatorAgent`) to provide several pieces of data for constructing
        the system state summary. The key data structures and fields utilized are:

        - `coordinator_ref.system_capabilities_cache` (Dict[str, Dict[str, Any]]):
            Expected to be a dictionary where each value is another dictionary
            containing at least a `'module': str` key-value pair, representing
            the module path of a scanned capability. Used to summarize available
            system capabilities.

        - `coordinator_ref.improvement_backlog` (List[Dict[str, Any]]):
            A list of dictionaries, where each dictionary represents an item in
            the improvement backlog. `BlueprintAgent` specifically looks for:
            - `'status': str`: e.g., "PENDING", "PENDING_APPROVAL".
            - `'priority': int`: Used for sorting; defaults to 0 if missing.

        - `coordinator_ref.improvement_campaign_history` (List[Dict[str, Any]]):
            A list of dictionaries detailing past improvement campaigns. For the
            last 3 entries, `BlueprintAgent` extracts:
            - `'target_component_id': Any`
            - `'status_from_sia_json': Any`
            - `'final_sia_op_status': Any`

        - Data from `coordinator_ref.resource_monitor.get_resource_usage()` (Dict[str, Any]):
            If `coordinator_ref.resource_monitor` exists, this method is called.
            The returned dictionary is expected to contain:
            - `'cpu_percent': float`: Defaults to 0.0 if key is missing.
            - `'memory_percent': float`: Defaults to 0.0 if key is missing.

        - Data from `coordinator_ref.performance_monitor.get_all_metrics()` (Dict[str, Dict[str, Any]]):
            If `coordinator_ref.performance_monitor` exists, this method is called.
            This is expected to be a dictionary of dictionaries. Each inner metrics
            dictionary is expected to have:
            - `'requests': int`: Defaults to 0 if key is missing.
            - `'success_rate': float`: Defaults to 1.0 if key is missing.
            This data is used to summarize LLM performance and identify potential
            low success rate models.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        """
        Ensures that only one instance of BlueprintAgent is created (Singleton pattern).

        Uses a class-level `_instance` variable to store the single instance
        and `_lock` for thread/async safety during the first instantiation,
        although the primary async safety for instantiation is typically handled
        in the factory method `get_blueprint_agent_async`.
        """
        if not cls._instance:
            # The lock is more critical in the factory method that might be called concurrently.
            # Here, it's a secondary check.
            cls._instance = super(BlueprintAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 belief_system: BeliefSystem,
                 coordinator_ref: CoordinatorAgent, # Reference to the live Coordinator
                 config_override: Optional[Config] = None,
                 test_mode: bool = False):
        """
        Initializes the BlueprintAgent's state.

        This method sets up the agent's configuration, references to essential
        components like the BeliefSystem and CoordinatorAgent, and initializes
        the LLM handler for blueprint generation.

        The `_initialized` flag and `test_mode` parameter work together to
        support the singleton pattern:
        - If the instance is already initialized (`hasattr(self, '_initialized') and self._initialized`)
          and `test_mode` is `False`, the initialization is skipped to prevent
          re-initializing a live singleton.
        - If `test_mode` is `True`, re-initialization can occur, which is useful
          for isolated testing.

        Args:
            belief_system (BeliefSystem): Reference to the MindX BeliefSystem.
            coordinator_ref (CoordinatorAgent): Reference to the live CoordinatorAgent
                for accessing system state (backlog, history, etc.).
            config_override (Optional[Config]): An optional Config object to
                override default configurations. Useful for testing.
            test_mode (bool): If True, allows re-initialization even if the
                singleton instance was previously initialized. Defaults to False.
        """
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

    async def _gather_mindx_system_state_summary(self) -> Dict[str, Any]:
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
        # Placeholder: Intended to be sourced dynamically (e.g., from BeliefSystem) in future versions.
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
        if high_level_directive:
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

            if not response_str or response_str.startswith("Error:"):
                logger.error(f"{self.agent_id}: LLM blueprint generation failed or returned empty/error: {response_str}")
                raise ValueError(f"LLM generation failed or returned error: {response_str}")

            try:
                blueprint_json = json.loads(response_str)
            except json.JSONDecodeError as jde:
                logger.error(f"{self.agent_id}: LLM blueprint response not valid JSON. Error: {jde}. Raw response (first 300 chars): {response_str[:300]}", exc_info=True)
                # Attempt to extract JSON from a potentially larger string if it's embedded
                match = re.search(r"\{[\s\S]*\}", response_str)
                if match:
                    try:
                        blueprint_json = json.loads(match.group(0))
                        logger.info(f"{self.agent_id}: Successfully extracted and parsed JSON from malformed LLM response.")
                    except json.JSONDecodeError as inner_jde:
                        extracted_json_str = match.group(0)
                        logger.error(f"{self.agent_id}: Failed to parse extracted JSON. Error: {inner_jde}. Extracted part (first 300 chars): {extracted_json_str[:300]}", exc_info=True)
                        raise ValueError(f"Blueprint LLM response contained an extractable JSON-like structure, but it was still invalid. Details: {inner_jde.msg}, position: {inner_jde.pos}, problematic part: '{extracted_json_str[max(0, inner_jde.pos-10):inner_jde.pos+10]}'") from inner_jde
                else:
                    # Original JDE details are more relevant here as no sub-structure was found
                    raise ValueError(f"Blueprint LLM response was not valid JSON and no JSON object found within. Details: {jde.msg}, position: {jde.pos}, problematic part: '{response_str[max(0, jde.pos-10):jde.pos+10]}'") from jde

            # Validate blueprint structure (basic check)
            if not blueprint_json or not all(k in blueprint_json for k in ["blueprint_title", "focus_areas"]):
                logger.error(f"{self.agent_id}: Generated blueprint missing essential keys. Blueprint: {str(blueprint_json)[:500]}")
                raise ValueError(f"Generated blueprint missing essential keys (e.g., 'blueprint_title', 'focus_areas'). Received: {list(blueprint_json.keys()) if blueprint_json else 'None'}")
            
            logger.info(f"{self.agent_id}: Successfully generated evolution blueprint titled '{blueprint_json.get('blueprint_title', 'Untitled Blueprint')}'.")
            # Persist blueprint to BeliefSystem
            await self.belief_system.add_belief(
                f"mindx.evolution.blueprint.latest",
                blueprint_json, 0.95, BeliefSource.SELF_ANALYSIS,
                metadata={"generated_at": time.time(), "mindx_version_input": current_mindx_version, "directive": high_level_directive}
            )
            return blueprint_json

        except json.JSONDecodeError as jde: # Specific handler for JSON parsing errors
            # This should ideally be caught by the inner try-except, but as a fallback.
            err_msg = f"Failed to parse LLM response as JSON. Details: {jde.msg}. Position: {jde.pos}. Near text: '{jde.doc[max(0, jde.pos-20):jde.pos+20]}'"
            logger.error(f"{self.agent_id}: {err_msg}", exc_info=True)
            return {"error": err_msg, "blueprint_title": "Error Blueprint - JSON Parsing Failed"}

        except ValueError as ve: # Specific handler for known value errors (LLM failure, validation)
            logger.error(f"{self.agent_id}: Value error during blueprint generation: {ve}", exc_info=True)
            return {"error": f"Blueprint generation value error: {ve}", "blueprint_title": "Error Blueprint - Validation/Input Error"}

        except Exception as e: # Generic handler for any other unexpected errors
            logger.error(f"{self.agent_id}: Unexpected exception during blueprint generation: {e}", exc_info=True)
            return {"error": f"Unexpected blueprint generation exception: {type(e).__name__}: {e}", "blueprint_title": "Error Blueprint - Unexpected"}

    async def shutdown(self):
        logger.info(f"BlueprintAgent '{self.agent_id}' shutting down.")
        # No specific async tasks to cancel in this agent for now.

    @classmethod
    async def reset_instance_async(cls): # For testing
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
) -> BlueprintAgent:
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

async def run_cli_blueprint_generation(args: argparse.Namespace) -> bool:
    """
    Main logic for blueprint generation when script is run from CLI.
    Returns True if successful, False otherwise.
    """
    # This requires Coordinator to be up and running first
    from mindx.orchestration.coordinator_agent import get_coordinator_agent_mindx_async

    # Ensure a fresh state for these singletons for the example run
    Config.reset_instance()
    # Pass config_file to Config if provided and Config supports it.
    # Assuming Config might take a custom_config_path or similar.
    # If not, this arg won't directly affect Config's file loading without Config changes.
    config_params = {"test_mode": True}
    if args.config_file:
        # This is an assumed way Config might take a specific file.
        # If Config.load_config() is the only way, this won't override default loading behavior.
        config_params["custom_config_path"] = str(args.config_file)
        if not args.quiet:
            print(f"Attempting to use config file: {args.config_file}", file=sys.stderr)
    config = Config(**config_params)
    
    BeliefSystem.reset_instance()
    shared_bs = BeliefSystem(test_mode=True) # test_mode=True typically means in-memory/transient

    if not args.quiet:
        logger.info("BlueprintAgent CLI: Initializing Coordinator...")
    try:
        coordinator = await get_coordinator_agent_mindx_async(config_override=config, test_mode=True)
        if not args.quiet:
            logger.info("BlueprintAgent CLI: Coordinator initialized.")
    except Exception as e:
        logger.error(f"BlueprintAgent CLI: Failed to initialize Coordinator: {e}", exc_info=True)
        print(f"Error: Failed to initialize Coordinator: {e}", file=sys.stderr)
        return False # Signal failure

    if not args.quiet:
        logger.info("BlueprintAgent CLI: Initializing BlueprintAgent...")
    try:
        bp_agent = await get_blueprint_agent_async(belief_system=shared_bs, coordinator_ref=coordinator, config_override=config, test_mode=True)
        if not args.quiet:
            logger.info("BlueprintAgent CLI: BlueprintAgent initialized.")
    except Exception as e:
        logger.error(f"BlueprintAgent CLI: Failed to initialize BlueprintAgent: {e}", exc_info=True)
        print(f"Error: Failed to initialize BlueprintAgent: {e}", file=sys.stderr)
        return False # Signal failure

    if not args.quiet:
        print(f"\n--- Generating Blueprint for MindX v{args.mindx_version} ---")
        if args.directive:
            print(f"Directive: {args.directive}")
        print(f"Iterations to look ahead: {args.iterations}\n")

    blueprint = await bp_agent.generate_next_evolution_blueprint(
        current_mindx_version=args.mindx_version,
        high_level_directive=args.directive,
        look_ahead_iterations=args.iterations
    )

    if "error" in blueprint:
        # Error details are already logged by the agent.
        print(f"Error: Blueprint generation failed - {blueprint['error']}", file=sys.stderr)
        return False # Signal failure
    elif args.output_file:
        output_path = Path(args.output_file)
        try:
            with open(output_path, 'w') as f:
                json.dump(blueprint, f, indent=2)
            if not args.quiet:
                print(f"Blueprint successfully written to: {output_path}")
        except IOError as e:
            print(f"Error: Failed to write blueprint to file {output_path}: {e}", file=sys.stderr)
            # Fallback to stdout if file write fails
            if not args.quiet:
                print("\n--- Generated Evolution Blueprint (fallback to stdout) ---")
            print(json.dumps(blueprint, indent=2))
            # Still considered a success if blueprint is produced, even if file write failed.
    else: # No output file, print to stdout (even if quiet, this is the primary output)
        print(json.dumps(blueprint, indent=2))

    # If we reached here and blueprint doesn't have an error, it's a success.
    # The detailed printout of focus areas is informational and its absence doesn't mean failure.
    if not args.quiet and blueprint and "focus_areas" in blueprint:
        print("-" * 50)
        for area in blueprint.get("focus_areas", []): # Use .get for safety, though already checked "error"
            print(f"\nFocus Area: {area.get('area_title')}")
            for goal in area.get("development_goals",[]):
                print(f"  - Goal: {goal.get('description')}")
                print(f"    Target: {goal.get('target_component_module_path')}")
                print(f"    Justification: {goal.get('justification')}")

    if not args.quiet:
        logger.info("BlueprintAgent CLI: Shutting down agents...")
    await bp_agent.shutdown()
    await coordinator.shutdown()
    if not args.quiet:
        logger.info("BlueprintAgent CLI: Agents shutdown complete.")
    return True # Signal success


if __name__ == "__main__": # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate an evolution blueprint for the MindX system.")
    parser.add_argument(
        "--mindx-version", "-v",
        required=True,
        type=str,
        help="Current MindX version (e.g., '0.5.0')."
    )
    parser.add_argument(
        "--directive", "-d",
        type=str,
        default=None,
        help="Optional high-level directive for the blueprint (e.g., 'Focus on safety')."
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=1,
        help="Number of major iterations to plan for (default: 1)."
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str, # Path will be handled by Path()
        default=None,
        help="Optional file path to save the generated blueprint JSON. Prints to stdout if not provided."
    )
    parser.add_argument(
        "--config-file", "-c",
        type=str, # Path will be handled by Path()
        default=None,
        help="Optional path to a custom configuration file for MindX."
    )
    parser.add_argument(
        "--quiet", "-q",
        action='store_true',
        help="Suppress informational logging and print only the blueprint JSON (or errors to stderr)."
    )

    cli_args = parser.parse_args()

    # Handle .env loading
    # Note: If a config-file is specified via CLI, it might define how .env is loaded or if it's used.
    # This basic .env loading here might be superseded by Config class's own loading logic if it's sophisticated.
    project_r_main = Path(__file__).resolve().parent.parent.parent
    env_p_main = project_r_main / ".env"
    if env_p_main.exists():
        from dotenv import load_dotenv
        # If quiet, we might not want .env to override existing env vars if not explicitly desired.
        # However, override=False is generally safer.
        load_dotenv(dotenv_path=env_p_main, override=False)
        if not cli_args.quiet:
            print(f"BlueprintAgent Main: Loaded .env file from {env_p_main}", file=sys.stderr)
    elif not cli_args.quiet: # Only print if not quiet and .env doesn't exist
        print(f"BlueprintAgent Main: .env not found at {env_p_main}. Using defaults/env vars.", file=sys.stderr)
    
    # Basic quiet mode: suppress logger's own console output for INFO/DEBUG if possible
    # This is a simple way; a more robust way involves configuring logging handlers.
    if cli_args.quiet:
        # Find the console handler and set its level higher, or remove it.
        # This is a bit hacky without direct access to logging_config.py setup.
        # For now, the conditional print statements in run_cli_blueprint_generation will handle most quietness.
        pass # Deferring more complex logger manipulation

    succeeded = False # Initialize status
    try:
        succeeded = asyncio.run(run_cli_blueprint_generation(cli_args))
    except KeyboardInterrupt:
        if not cli_args.quiet:
            logger.info("BlueprintAgent CLI interrupted by user.")
        print("Error: Operation cancelled by user.", file=sys.stderr)
        sys.exit(130) # Exit code for SIGINT
    except Exception as e:
        logger.error(f"BlueprintAgent CLI: An unexpected error occurred: {e}", exc_info=True)
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1) # General error exit code
    finally:
        if not cli_args.quiet:
            logger.info("BlueprintAgent CLI finished.")

    sys.exit(0 if succeeded else 1)
