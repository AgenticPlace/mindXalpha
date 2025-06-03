# scripts/run_mindx_coordinator.py
import asyncio
import json
import sys
from pathlib import Path
import readline # For better input experience, history, editing (optional but nice for CLI)

# Adjust Python's search path to find the 'mindx' package
# Assumes this script is in 'scripts/' and 'mindx/' is a sibling directory
# (i.e., project_root/scripts/ and project_root/mindx/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mindx.orchestration.coordinator_agent import get_coordinator_agent_mindx_async, InteractionType, CoordinatorAgent
from mindx.utils.logging_config import get_logger # Use project's logger
from mindx.utils.config import Config, PROJECT_ROOT # Use project's config and resolved PROJECT_ROOT

logger = get_logger(__name__)

async def main_cli_loop(coordinator: CoordinatorAgent):
    """Main interactive loop for the CLI."""
    logger.info("MindX Coordinator Agent ready for CLI interaction.")
    print("\nWelcome to the MindX Command Line Interface (Augmentic Project).")
    print(f"Project Root: {PROJECT_ROOT}")
    print("Type 'help' for a list of commands, or 'quit'/'exit' to stop.")

    while True:
        try:
            # Use asyncio.to_thread for blocking input() call in async context
            user_input_str = await asyncio.to_thread(input, "MindX CLI > ")
        except (EOFError, KeyboardInterrupt): # pragma: no cover
            logger.info("EOF or KeyboardInterrupt received. Exiting CLI gracefully.")
            break # Exit loop on Ctrl+D or Ctrl+C

        if not user_input_str.strip(): # Skip empty input
            continue

        if user_input_str.lower() in ["quit", "exit"]: # pragma: no cover
            logger.info("Exit command received. Shutting down.")
            break
        
        if user_input_str.lower() == "help": # pragma: no cover
            print("\nAvailable MindX CLI Commands:")
            print("  query <your question>                            - Send a general query to the Coordinator's LLM.")
            print("  analyze_system [optional context for analysis]   - Trigger a system-wide analysis for improvements.")
            print("  improve <component_id> [optional context]        - Request an improvement for a specific component.")
            print("    (e.g., improve mindx.core.belief_system Add persistence layer)")
            print("    (e.g., improve self_improve_agent_cli_mindx Enhance LLM prompt for code generation)")
            print("    (<component_id> is module path like 'mindx.utils.config' or 'self_improve_agent_cli_mindx')")
            print("  backlog                                          - Display the current improvement backlog.")
            print("  process_backlog                                  - Manually trigger processing of one pending backlog item.")
            print("  approve <backlog_item_id>                        - Approve a backlog item marked 'pending_approval'.")
            print("  reject <backlog_item_id>                         - Reject a backlog item marked 'pending_approval'.")
            print("  help                                             - Show this help message.")
            print("  quit / exit                                      - Shut down the Coordinator and exit CLI.")
            print("-" * 60)
            continue

        parts = user_input_str.strip().split(" ", 1)
        command_verb = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""
        
        interaction_type: Optional[InteractionType] = None
        content_for_interaction = args_str # Default content
        metadata_for_interaction: Dict[str, Any] = {"cli_source_command": user_input_str} # Track original command

        if command_verb == "query":
            interaction_type = InteractionType.QUERY
            if not args_str: print("Error: 'query' command requires a question."); continue
        elif command_verb == "analyze_system":
            interaction_type = InteractionType.SYSTEM_ANALYSIS
            content_for_interaction = "System analysis requested via CLI."
            if args_str: metadata_for_interaction["analysis_context"] = args_str
        elif command_verb == "improve":
            interaction_type = InteractionType.COMPONENT_IMPROVEMENT
            improve_args_parts = args_str.split(" ", 1)
            if not improve_args_parts or not improve_args_parts[0]:
                print("Error: 'improve' command requires a <component_id> (e.g., 'mindx.core.belief_system' or 'self_improve_agent_cli_mindx').")
                continue
            metadata_for_interaction["target_component"] = improve_args_parts[0]
            if len(improve_args_parts) > 1:
                metadata_for_interaction["analysis_context"] = improve_args_parts[1]
            else: # Default context if none provided for improve command
                metadata_for_interaction["analysis_context"] = f"General improvement request for {metadata_for_interaction['target_component']} from CLI."
            content_for_interaction = f"CLI Request: Improve component {metadata_for_interaction['target_component']}"
        elif command_verb == "backlog": # pragma: no cover
            print("\n--- MindX Improvement Backlog ---")
            if coordinator.improvement_backlog:
                for i, item in enumerate(coordinator.improvement_backlog):
                    print(
                        f"{i+1}. ID: {item.get('id', 'N/A')[:8]} | Prio: {item.get('priority', '?')} | "
                        f"Status: {item.get('status', 'unknown')} | Target: {item.get('target_component_path', 'N/A')}\n"
                        f"   Suggestion: {item.get('suggestion', 'N/A')[:100]}...\n"
                        f"   Source: {item.get('source', 'N/A')} | Added: {datetime.fromtimestamp(item.get('added_at',0)).strftime('%Y-%m-%d %H:%M')}"
                    )
            else:
                print("Improvement backlog is currently empty.")
            print("-" * 30)
            continue
        elif command_verb == "process_backlog": # pragma: no cover
            logger.info("CLI: User requested processing of one backlog item.")
            # Find the first actionable item (pending and approved if critical)
            actionable_item: Optional[Dict[str, Any]] = None
            for item in coordinator.improvement_backlog:
                if item.get("status") == InteractionStatus.PENDING.value:
                    target_is_critical = any(crit_stem in item["target_component_path"] for crit_stem in coordinator.critical_components_for_approval) or item.get("is_critical_target", False)
                    if coordinator.require_human_approval_for_critical and target_is_critical and item.get("approved_at") is None:
                        continue # Skip, needs approval
                    actionable_item = item
                    break 
            
            if actionable_item:
                print(f"Attempting to process backlog item ID: {actionable_item.get('id','N/A')[:8]} for target: {actionable_item['target_component_path']}")
                # Mark as in_progress_sia via metadata for the interaction.
                # The autonomous loop does this directly on the backlog item.
                # Here, we create a new COMPONENT_IMPROVEMENT interaction.
                interaction_type = InteractionType.COMPONENT_IMPROVEMENT
                metadata_for_interaction = {
                    "target_component": actionable_item["target_component_path"],
                    "analysis_context": actionable_item["suggestion"],
                    "source": "cli_manual_backlog_process",
                    "original_priority": actionable_item["priority"],
                    "backlog_item_id": actionable_item.get("id") # Link to backlog item
                }
                content_for_interaction = f"CLI: Process backlog item '{actionable_item.get('id','N/A')[:8]}': {actionable_item['suggestion'][:100]}"
                # Update backlog item status immediately (optimistic)
                actionable_item["status"] = InteractionStatus.IN_PROGRESS.value
                actionable_item["last_attempted_at"] = time.time()
                actionable_item["attempt_count"] = actionable_item.get("attempt_count",0) + 1
                coordinator._save_backlog()
            else:
                print("No actionable (pending and approved, if critical) items found in the improvement backlog to process.")
                continue
        elif command_verb == "approve":
            interaction_type = InteractionType.APPROVE_IMPROVEMENT
            if not args_str.strip(): print("Error: 'approve' command requires a <backlog_item_id>."); continue
            metadata_for_interaction["backlog_item_id"] = args_str.strip()
            content_for_interaction = f"CLI Request: Approve backlog item {metadata_for_interaction['backlog_item_id']}"
        elif command_verb == "reject":
            interaction_type = InteractionType.REJECT_IMPROVEMENT
            if not args_str.strip(): print("Error: 'reject' command requires a <backlog_item_id>."); continue
            metadata_for_interaction["backlog_item_id"] = args_str.strip()
            content_for_interaction = f"CLI Request: Reject backlog item {metadata_for_interaction['backlog_item_id']}"
        else:
            print(f"Unknown command: '{command_verb}'. Type 'help' for available commands.")
            continue

        # Dispatch the interaction if one was determined
        if interaction_type:
            try:
                result_dict = await coordinator.handle_user_input(
                    content=content_for_interaction,
                    user_id="cli_user_augmentic_prod_cand",
                    interaction_type=interaction_type,
                    metadata=metadata_for_interaction
                )
                print("\n=== MindX Coordinator Response ===")
                try:
                    # Attempt to pretty-print JSON response
                    print(json.dumps(result_dict, indent=2, default=str)) # default=str for non-serializable
                except TypeError: # pragma: no cover
                    print(str(result_dict)) # Fallback if result_dict contains non-JSON serializable parts
                print("=" * 30)
            except Exception as e_handle: # pragma: no cover
                logger.error(f"Error during coordinator.handle_user_input: {e_handle}", exc_info=True)
                print(f"CLI Error: Failed to handle input - {type(e_handle).__name__}: {e_handle}")
        else: # Should not be reached if logic above is correct # pragma: no cover
             print(f"Internal CLI Error: Could not determine interaction type for input: {user_input_str}")
        
        print("-" * 60) # Separator for next prompt

async def main_entry(): # pragma: no cover
    """Main entry point for the script."""
    # Config() will load .env due to its __init__ if called first.
    # The factory get_coordinator_agent_mindx_async also creates a Config instance.
    app_config = Config() # Ensure it's loaded once for the application session
    logger.info(f"MindX CLI (Augmentic Project) v_prod_candidate. Project Root: {PROJECT_ROOT}")
    
    coordinator_instance: Optional[CoordinatorAgent] = None
    try:
        coordinator_instance = await get_coordinator_agent_mindx_async(config_override=app_config)
        await main_cli_loop(coordinator_instance)
    except Exception as e:
        logger.critical(f"Critical error during MindX CLI startup or main loop: {e}", exc_info=True)
        print(f"A critical error occurred. Please check logs. Error: {e}")
    finally:
        if coordinator_instance:
            logger.info("Shutting down MindX Coordinator Agent from CLI...")
            await coordinator_instance.shutdown()
            logger.info("MindX Coordinator Agent shutdown complete.")
        else: # pragma: no cover
            logger.info("MindX CLI finished, Coordinator was not initialized.")
        # Small delay to allow final log messages to flush, especially if file handlers are async.
        await asyncio.sleep(0.1)

if __name__ == "__main__": # pragma: no cover
    # This structure ensures that if this script is run directly, it sets up and runs the asyncio event loop.
    try:
        asyncio.run(main_entry())
    except KeyboardInterrupt:
        logger.info("MindX CLI terminated by user (KeyboardInterrupt).")
    except Exception as e_top: # Catch any other top-level exceptions during asyncio.run
        logger.critical(f"MindX CLI top-level error: {e_top}", exc_info=True)
        print(f"MindX CLI encountered a fatal error: {e_top}")
    finally:
        print("MindX CLI exited.")
