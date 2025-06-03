# MindX Coordinator CLI (`run_mindx_coordinator.py`) - Production Candidate

## Introduction

This script provides a Command Line Interface (CLI) for interacting with the `CoordinatorAgent` of the MindX system (Augmentic Project). It serves as the primary user-facing tool for issuing commands, requesting system analyses, triggering component improvements, managing the improvement backlog, and observing the system's behavior.

## Explanation

### Core Functionality

1.  **Initialization:**
    *   Adjusts Python's `sys.path` to correctly locate the `mindx` package modules.
    *   Initializes the global `Config` object, which loads settings from `.env` files and environment variables. This configuration is then passed to the `CoordinatorAgent`.
    *   Asynchronously obtains an instance of the `CoordinatorAgent` using the `get_coordinator_agent_mindx_async` factory function. This ensures all necessary sub-components of the Coordinator (like monitors, LLM handlers) are also initialized.
    *   Handles potential errors during Coordinator initialization.

2.  **Interactive Command Loop (`main_cli_loop`):**
    *   Provides a `MindX CLI > ` prompt for user input.
    *   Uses `asyncio.to_thread(input, ...)` to handle the blocking `input()` call in an asynchronous environment without freezing the event loop.
    *   **Command Parsing:**
        *   Parses simple commands like `query`, `analyze_system`, `improve`, `backlog`, `process_backlog`, `approve`, `reject`, `help`, `quit`, `exit`.
        *   For commands like `improve` and `analyze_system`, it extracts arguments (e.g., `target_component`, `analysis_context`) from the input string.
        *   For `collab` (if it were fully implemented), it would parse a prompt and a JSON metadata string. (Currently, `collab` is not a primary command in this version).
    *   **Interaction Creation:** Based on the parsed command, it determines the appropriate `InteractionType` and constructs `metadata` for the `CoordinatorAgent.handle_user_input` method.
    *   **Output Display:** Receives a JSON dictionary response from the `CoordinatorAgent` and pretty-prints it to the console. Uses `default=str` in `json.dumps` to handle non-serializable objects like Enums in the response.
    *   **Loop Control:** Continues until "quit" or "exit" is typed, or `Ctrl+C`/`Ctrl+D` is pressed.

3.  **Supported CLI Commands:**
    *   `query <your question>`: Sends the question as a general query to the `CoordinatorAgent`, which typically routes it to its configured LLM.
    *   `analyze_system [optional context for analysis focus]`: Triggers a system-wide analysis by the `CoordinatorAgent`. The Coordinator will use its LLM, codebase scan, and monitor data to generate improvement suggestions, which are added to its internal backlog.
    *   `improve <component_id> [optional_context_for_improvement]`: Requests the `CoordinatorAgent` to initiate an improvement task for a specific component.
        -   `<component_id>`: Can be a full Python module path (e.g., `mindx.core.belief_system`) or a special registered agent ID like `self_improve_agent_cli_mindx` (for SIA to improve itself).
        -   `[optional_context_for_improvement]`: Textual guidance for the improvement.
        -   This command results in the Coordinator invoking the `SelfImprovementAgent` CLI.
    *   `backlog`: Displays the current list of improvement suggestions stored in the `CoordinatorAgent`'s backlog, showing their ID (first 8 chars), priority, target, status, and a snippet of the suggestion.
    *   `process_backlog`: Manually tells the `CoordinatorAgent` to attempt to process the highest-priority actionable (pending and approved, if critical) item from its improvement backlog. This is useful for testing the autonomous loop's decision-making or for stepping through improvements.
    *   `approve <backlog_item_id>`: Approves a specific improvement item in the backlog that is currently in `pending_approval` status. This allows the autonomous loop to proceed with it if it's a critical target.
    *   `reject <backlog_item_id>`: Rejects a specific `pending_approval` item, marking it as `rejected_manual`.
    *   `help`: Displays a list of available commands and their usage.
    *   `quit` / `exit`: Gracefully shuts down the `CoordinatorAgent` and exits the CLI.

4.  **Shutdown (`main_entry`'s `finally` block):**
    *   Ensures that `coordinator.shutdown()` is called when the CLI exits, allowing the `CoordinatorAgent` and its managed components (like monitors and the autonomous loop task) to clean up resources and persist final states (e.g., saving the backlog).

## Technical Details

-   **Asynchronous:** The entire CLI and its interaction with the `CoordinatorAgent` are built using `asyncio`.
-   **Path Management:** Uses `pathlib.Path` and `sys.path.insert` to ensure the `mindx` package is discoverable. Relies on `PROJECT_ROOT` from `mindx.utils.config`.
-   **Input Handling:** `readline` is imported for an improved CLI experience (history, line editing) on systems where it's available. `asyncio.to_thread` prevents `input()` from blocking.
-   **Error Handling:** Includes `try-except` blocks for user interruption (`KeyboardInterrupt`, `EOFError`) and general exceptions within the main loop and during Coordinator initialization/shutdown.

## Usage

1.  **Ensure Dependencies:** Make sure all dependencies listed in `pyproject.toml` (or `requirements.txt`) are installed.
2.  **Set up `.env`:** Create a `.env` file in the `augmentic_mindx/` project root with necessary configurations, especially `MINDX_LLM__...` API keys and model preferences.
3.  **Run from Project Root:**
    ```bash
    cd /path/to/augmentic_mindx/
    python scripts/run_mindx_coordinator.py
    ```
4.  **Interact:** Once the `MindX CLI > ` prompt appears, type commands as described above.

This CLI script is the primary way for a developer or another system to interact with the MindX self-improvement capabilities at a high level.
