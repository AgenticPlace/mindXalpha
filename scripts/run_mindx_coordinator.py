# scripts/run_mindx.py
import asyncio
import json
import sys
from pathlib import Path
import readline
from typing import Optional, Dict, Any
from datetime import datetime
import time

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from orchestration.mastermind_agent import MastermindAgent
from orchestration.coordinator_agent import InteractionType, InteractionStatus
from utils.logging_config import get_logger
from utils.config import Config, PROJECT_ROOT as CONFIG_PROJECT_ROOT

logger = None

async def main_cli_loop(mastermind: MastermindAgent):
    global logger
    if not logger: logger = get_logger(__name__)

    logger.info("mindX Mastermind Agent (Augmentic Intelligence) ready for CLI interaction.")
    print("\nWelcome to the mindX Command Line Interface (Mastermind Edition - Augmentic Intelligence).")
    print("This system operates on the principles of Augmentic Intelligence.")
    print(f"Project Root (calculated by script): {project_root}")
    print(f"Project Root (from utils.config): {CONFIG_PROJECT_ROOT}")
    print("Type 'help' for a list of commands, or 'quit'/'exit' to stop.")

    while True:
        try:
            user_input_str = await asyncio.to_thread(input, "mindX (Mastermind) > ")
        except (EOFError, KeyboardInterrupt): # pragma: no cover
            logger.info("EOF or KeyboardInterrupt received. Exiting mindX CLI gracefully.")
            break
        if not user_input_str.strip(): continue
        if user_input_str.lower() in ["quit", "exit"]: logger.info("Exit command received. Shutting down mindX."); break

        if user_input_str.lower() == "help": # pragma: no cover
            print("\nAvailable mindX CLI Commands (Mastermind Edition - Augmentic Intelligence):")
            print("  evolve <directive>                               - Task Mastermind with a high-level evolutionary directive.")
            print("     (e.g., evolve Enhance system-wide Augmentic Intelligence logging)")
            print("     (e.g., evolve Assess current tool suite and propose new code analysis tool)")
            print("  mastermind_status                              - Display Mastermind's objectives and campaign history.")
            print("  show_tool_registry                             - Display the official tool registry content.")
            print("  analyze_codebase <path_to_code> [focus_prompt] - Mastermind: Analyze codebase using BaseGenAgent.")
            print("\n  --- Coordinator Commands (via Mastermind for mindX) ---")
            print("  coord_query <your question>                      - Send query to Coordinator's LLM.")
            print("  coord_analyze [optional context]               - Trigger Coordinator's system analysis.")
            print("  coord_improve <component_id> [optional context]- Request Coordinator improve a component.")
            print("     (e.g. coord_improve utils.config Add new feature)")
            print("  coord_backlog                                  - Display Coordinator's improvement backlog.")
            print("  coord_process_backlog                          - Trigger Coordinator to process one backlog item.")
            print("  coord_approve <backlog_item_id>                - Approve a Coordinator backlog item.")
            print("  coord_reject <backlog_item_id>                 - Reject a Coordinator backlog item.")
            print("-" * 60)
            print("  help                                             - Show this help message.")
            print("  quit / exit                                      - Shut down mindX and exit CLI.")
            print("-" * 60)
            continue

        parts = user_input_str.strip().split(" ", 1)
        command_verb = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        if command_verb == "evolve":
            if not args_str: print("Error: 'evolve' command requires a <directive> string."); continue
            print(f"\nmindX Mastermind: Initiating evolution campaign with directive: '{args_str}'...")
            try:
                result = await mastermind.manage_mindx_evolution(top_level_directive=args_str)
                print("\n=== mindX Mastermind Evolution Campaign Outcome ===")
                print(json.dumps(result, indent=2, default=str))
            except Exception as e: logger.error(f"Error during Mastermind evolution: {e}", exc_info=True); print(f"CLI Error: {e}")
        elif command_verb == "mastermind_status": # pragma: no cover
            print("\n--- mindX Mastermind Status (Augmentic Intelligence) ---")
            print("High-Level Objectives for mindX:"); obiettivi = mastermind.high_level_objectives
            if obiettivi:
                for i, obj in enumerate(obiettivi): print(f"  {i+1}. ID: {obj.get('id', 'N/A')} | Status: {obj.get('status', 'N/A')}\n     Directive: {obj.get('directive', 'N/A')[:100]}...\n     Created: {datetime.fromtimestamp(obj.get('created_at',0)).strftime('%Y-%m-%d %H:%M') if obj.get('created_at',0) else 'N/A'}")
            else: print("  No high-level objectives defined.")
            print("\nStrategic Campaigns History for mindX (last 5):"); camp_hist = mastermind.strategic_campaigns_history
            if camp_hist:
                for i, camp in enumerate(camp_hist[-5:]): print(f"  {i+1}. Run ID: {camp.get('mastermind_run_id', 'N/A')} | Status: {camp.get('overall_campaign_status', 'N/A')}\n     Directive: {camp.get('directive', 'N/A')[:100]}...\n     Timestamp: {datetime.fromtimestamp(camp.get('timestamp',0)).strftime('%Y-%m-%d %H:%M') if camp.get('timestamp',0) else 'N/A'}")
            else: print("  No strategic campaigns recorded.")
            print("-" * 30)
        elif command_verb == "show_tool_registry":
            print("\n--- Official mindX Tool Registry ---")
            print(json.dumps(mastermind.tools_registry, indent=2, default=str))
            print("-" * 30)
        elif command_verb == "analyze_codebase":
            code_path_parts = args_str.split(" ", 1)
            if not code_path_parts or not code_path_parts[0]: print("Error: analyze_codebase requires <path_to_code>."); continue
            target_code_path = code_path_parts[0]
            focus = code_path_parts[1] if len(code_path_parts) > 1 else "General strategic analysis and tool extrapolation opportunities."
            print(f"\nMastermind: Initiating codebase analysis for '{target_code_path}' with focus: '{focus}'")
            directive = f"Analyze codebase at '{target_code_path}' focusing on '{focus}', and store results. Then, assess tool suite based on new findings."
            result = await mastermind.manage_mindx_evolution(top_level_directive=directive, max_mastermind_bdi_cycles=10)
            print("\n=== Mastermind Codebase Analysis Campaign Outcome ===")
            print(json.dumps(result, indent=2, default=str))

        elif command_verb.startswith("coord_"):
            if not mastermind.coordinator_agent: print("Error: Mastermind's CoordinatorAgent is not available."); continue # pragma: no cover
            coordinator = mastermind.coordinator_agent
            interaction_type: Optional[InteractionType] = None
            content_for_interaction = args_str
            metadata_for_interaction: Dict[str, Any] = {"cli_source_command": user_input_str, "routed_via_mastermind_cli_for_mindX": True}
            if command_verb == "coord_query":
                interaction_type = InteractionType.QUERY
                if not args_str: print("Error: 'coord_query' command requires a question."); continue
                content_for_interaction = f"Query for mindX Coordinator (Augmentic Intelligence): {args_str}"
            elif command_verb == "coord_analyze":
                interaction_type = InteractionType.SYSTEM_ANALYSIS
                content_for_interaction = "System analysis of mindX requested via Mastermind CLI."
                if args_str: metadata_for_interaction["analysis_context"] = f"mindX analysis context: {args_str}"
            elif command_verb == "coord_improve":
                interaction_type = InteractionType.COMPONENT_IMPROVEMENT
                improve_args_parts = args_str.split(" ", 1)
                if not improve_args_parts or not improve_args_parts[0]: print("Error: 'coord_improve' command requires a <component_id>."); continue
                metadata_for_interaction["target_component"] = improve_args_parts[0]
                context_detail = f"General improvement request for mindX component {metadata_for_interaction['target_component']} from Mastermind CLI."
                if len(improve_args_parts) > 1: context_detail = improve_args_parts[1]
                metadata_for_interaction["analysis_context"] = context_detail
                content_for_interaction = f"CLI Request (via Mastermind for mindX): Improve component {metadata_for_interaction['target_component']}"
            elif command_verb == "coord_backlog": # pragma: no cover
                print("\n--- mindX Coordinator Improvement Backlog (via Mastermind) ---")
                if coordinator.improvement_backlog:
                    for i, item in enumerate(coordinator.improvement_backlog):
                        added_ts = item.get('added_at', 0); added_dt = datetime.fromtimestamp(added_ts).strftime('%Y-%m-%d %H:%M') if added_ts else "N/A"
                        print(f"{i+1}. ID: {item.get('id', 'N/A')[:8]} | Prio: {item.get('priority', '?')} | Status: {item.get('status', 'unknown')} | Target: {item.get('target_component_path', 'N/A')}\n   Suggestion: {item.get('suggestion', 'N/A')[:100]}...\n   Source: {item.get('source', 'N/A')} | Added: {added_dt}")
                else: print("mindX Coordinator's improvement backlog is currently empty.")
                print("-" * 30); continue
            elif command_verb == "coord_process_backlog": # pragma: no cover
                logger.info("CLI (via Mastermind for mindX): User requested Coordinator processing of one backlog item.")
                actionable_item: Optional[Dict[str, Any]] = None
                for item in coordinator.improvement_backlog:
                    if item.get("status") == InteractionStatus.PENDING.value:
                        is_critical_target = item.get("is_critical_target", False) or any(crit_stem in item.get("target_component_path","") for crit_stem in getattr(coordinator, "critical_components_for_approval", []))
                        if getattr(coordinator, "require_human_approval_for_critical", False) and is_critical_target and item.get("approved_at") is None: continue
                        actionable_item = item; break
                if actionable_item:
                    print(f"Attempting to process mindX Coordinator backlog item ID: {actionable_item.get('id','N/A')[:8]} for target: {actionable_item['target_component_path']}")
                    interaction_type = InteractionType.COMPONENT_IMPROVEMENT
                    metadata_for_interaction = {"target_component": actionable_item["target_component_path"], "analysis_context": actionable_item["suggestion"], "source": "cli_manual_backlog_process_via_mastermind_for_mindX", "original_priority": actionable_item.get("priority"), "backlog_item_id": actionable_item.get("id")}
                    content_for_interaction = f"CLI (via Mastermind for mindX): Process backlog item '{actionable_item.get('id','N/A')[:8]}': {actionable_item['suggestion'][:100]}"
                    actionable_item["status"] = InteractionStatus.IN_PROGRESS.value; actionable_item["last_attempted_at"] = time.time(); actionable_item["attempt_count"] = actionable_item.get("attempt_count", 0) + 1
                    if hasattr(coordinator, "_save_backlog") and callable(coordinator._save_backlog): coordinator._save_backlog()
                    else: logger.warning("mindX Coordinator instance does not have _save_backlog method.") # pragma: no cover
                else: print("No actionable items found in mindX Coordinator's backlog to process."); continue
            elif command_verb == "coord_approve":
                interaction_type = InteractionType.APPROVE_IMPROVEMENT
                if not args_str.strip(): print("Error: 'coord_approve' command requires a <mindx_backlog_item_id>."); continue
                metadata_for_interaction["backlog_item_id"] = args_str.strip()
                content_for_interaction = f"CLI Request (via Mastermind for mindX): Approve backlog item {metadata_for_interaction['backlog_item_id']}"
            elif command_verb == "coord_reject":
                interaction_type = InteractionType.REJECT_IMPROVEMENT
                if not args_str.strip(): print("Error: 'coord_reject' command requires a <mindx_backlog_item_id>."); continue
                metadata_for_interaction["backlog_item_id"] = args_str.strip()
                content_for_interaction = f"CLI Request (via Mastermind for mindX): Reject backlog item {metadata_for_interaction['backlog_item_id']}"
            else: print(f"Unknown 'coord_*' command for mindX: '{command_verb}'. Type 'help'."); continue # pragma: no cover

            if interaction_type and hasattr(coordinator, "handle_user_input"):
                try:
                    result_dict = await coordinator.handle_user_input(content=content_for_interaction, user_id="cli_user_mastermind_edition_mindX", interaction_type=interaction_type, metadata=metadata_for_interaction)
                    print("\n=== mindX Coordinator Response (via Mastermind) ==="); print(json.dumps(result_dict, indent=2, default=str)); print("=" * 30)
                except Exception as e_handle: # pragma: no cover
                    if logger: logger.error(f"Error during mindX coordinator.handle_user_input: {e_handle}", exc_info=True)
                    print(f"CLI Error: Failed to handle mindX Coordinator input - {e_handle}")
            elif not hasattr(coordinator, "handle_user_input"): # pragma: no cover
                 print("Error: mindX Mastermind's Coordinator instance does not have 'handle_user_input' method.")
        else:
            if command_verb not in ["help", "quit", "exit"]:
                 print(f"Unknown command for mindX: '{command_verb}'. Type 'help'.")
        print("-" * 60)

async def main_entry(): # pragma: no cover
    global logger
    mastermind_instance: Optional[MastermindAgent] = None
    try:
        logger = get_logger(__name__)
        logger.info(f"mindX CLI (Mastermind Edition - Augmentic Intelligence). Project Root (calculated): {project_root}")
        logger.info(f"mindX CLI trying to use Project Root (from config defined in utils.config): {CONFIG_PROJECT_ROOT}")
        app_config = Config()
        mastermind_instance = await MastermindAgent.get_instance(config_override=app_config)
        if not mastermind_instance:
            logger.critical("Failed to initialize mindX MastermindAgent instance. CLI cannot start.")
            print("Fatal Error: Could not start mindX Mastermind. Check logs.")
            return
        await main_cli_loop(mastermind_instance)
    except ModuleNotFoundError as e_mod:
        print(f"FATAL: A module was not found: {e_mod}")
        print("This usually means that the primary package (e.g., 'orchestration', 'utils') is not discoverable or `__init__.py` files are missing within them.")
        print(f"Please ensure you are running this script from the project root (e.g., {Path.cwd()}) and that your package directories (like 'orchestration', 'utils', etc., directly under the project root) contain an `__init__.py` file.")
        print(f"Current sys.path (Python's search path for modules):")
        for p_item in sys.path: print(f"  - {p_item}")
        if logger: logger.critical(f"ModuleNotFoundError during mindX CLI startup: {e_mod}", exc_info=True)
    except Exception as e:
        if logger: logger.critical(f"Critical error during mindX CLI startup or main loop: {e}", exc_info=True)
        else: print(f"Critical error (logger not initialized): {e}")
        print(f"A critical error occurred with mindX. Check logs. Error: {e}")
    finally:
        if mastermind_instance:
            if logger: logger.info("Shutting down mindX Mastermind Agent from CLI...")
            await mastermind_instance.shutdown()
            if logger: logger.info("mindX Mastermind Agent shutdown complete.")
        else:
            if logger: logger.info("mindX CLI (Mastermind Edition - Augmentic Intelligence) finished. MastermindAgent was not initialized or failed.")
        await asyncio.sleep(0.1)

if __name__ == "__main__": # pragma: no cover
    try: asyncio.run(main_entry())
    except KeyboardInterrupt:
        if logger: logger.info("mindX CLI (Mastermind Edition - Augmentic Intelligence) terminated by user (KeyboardInterrupt).")
        else: print("mindX CLI terminated by user.")
    except Exception as e_top:
        if logger: logger.critical(f"mindX CLI (Mastermind Edition - Augmentic Intelligence) top-level error: {e_top}", exc_info=True)
        else: print(f"mindX CLI top-level error (logger not initialized): {e_top}")
        print(f"mindX CLI (Mastermind Edition - Augmentic Intelligence) encountered a fatal error: {e_top}")
    finally: print("mindX CLI (Mastermind Edition - Augmentic Intelligence) exited.")
