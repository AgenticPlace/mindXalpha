# Coordinator Agent (`coordinator_agent.py`) - Production Candidate v2

## Introduction

The `CoordinatorAgent` is the central orchestrator for the MindX system, an Augmentic Project initiative. It manages high-level system operations, including task delegation (e.g., to LLMs or specialized agents), strategic decision-making for self-improvement, and serves as the primary interface for user and inter-agent interactions. This version integrates data from `ResourceMonitor` and `PerformanceMonitor` to inform its system analysis and autonomously drives an improvement backlog, including Human-in-the-Loop (HITL) steps for critical changes. It delegates actual code modifications to the `SelfImprovementAgent` (SIA) via its robust CLI.

## Explanation

### Core Responsibilities & Workflow

1.  **Interaction Management:**
    *   Handles incoming requests encapsulated as `Interaction` objects. Each interaction has a specific `InteractionType` (e.g., `QUERY`, `SYSTEM_ANALYSIS`, `COMPONENT_IMPROVEMENT`, `APPROVE_IMPROVEMENT`).
    *   Tracks the lifecycle and history of each interaction.

2.  **System-Level Analysis (`_process_system_analysis`):**
    *   Triggered on demand (e.g., by user command) or periodically by the autonomous improvement loop.
    *   **Comprehensive Data Gathering:**
        *   Scans the MindX codebase (`_scan_codebase_capabilities`) using Abstract Syntax Trees (AST) to create a snapshot of current modules, classes, and functions.
        *   Queries `ResourceMonitor` for current CPU, memory, and disk usage, highlighting any conditions near or exceeding configured alert thresholds.
        *   Queries `PerformanceMonitor` for a summary of LLM interaction performance, noting models or task types with low success rates or high latencies.
        *   (Optionally) Reviews recent `improvement_campaign_history` for context on past successes or failures.
    *   **LLM-Driven Insight Generation:** Constructs a detailed prompt for its own LLM instance (`self.llm_handler`). This prompt includes the aggregated system data and a request to identify 1-3 high-impact improvement areas.
    *   **Structured Suggestions:** The LLM is instructed to return suggestions in a specific JSON format, detailing:
        *   `target_component_path`: The full Python module path (e.g., `mindx.learning.self_improve_agent`).
        *   `suggestion`: A concise description of the proposed improvement and its rationale.
        *   `priority`: An integer (1-10) indicating urgency/impact.
        *   `is_critical_target`: A boolean flag set by the LLM if it deems the target component core to system stability (e.g., the SIA, Coordinator, or `Config`).
    *   Generated suggestions are added to the `improvement_backlog`.

3.  **Improvement Backlog Management:**
    *   `add_to_improvement_backlog()`: Adds new suggestions to an internal list (`self.improvement_backlog`). It assigns a unique ID, sets initial status to "pending", and performs basic de-duplication against existing pending items with similar targets and descriptions.
    *   The backlog is sorted to prioritize items: `pending_approval` items appear first, then items are sorted by `priority` (descending), and finally by `added_at` (oldest first for same priority).
    *   The backlog is persisted to `PROJECT_ROOT/data/improvement_backlog.json` to maintain state across Coordinator restarts.

4.  **Delegation to SelfImprovementAgent (SIA) via CLI (`_process_component_improvement_cli`):**
    *   This method is the executive arm for `COMPONENT_IMPROVEMENT` interactions.
    *   **Path Resolution:** It resolves the `target_component` identifier (which can be a module path or a registered agent ID like `self_improve_agent_cli_mindx`) to an absolute file system path using `_resolve_component_path_for_sia`.
    *   **SIA CLI Invocation:**
        *   Constructs a command to execute `self_improve_agent.py` as a separate subprocess using `asyncio.create_subprocess_exec`.
        *   Passes the target file path, the improvement context/goal (from `analysis_context` in metadata, written to a temporary file if large), maximum cycles for SIA, and any LLM/parameter overrides for the SIA's internal operations.
        *   Critically, it passes `--output-json` to the SIA CLI.
        *   An `asyncio.Semaphore` (`sia_concurrency_limit`) is used to control the number of concurrent SIA subprocesses.
    *   **Result Parsing & Handling:**
        *   Waits for the SIA subprocess to complete (with a configurable timeout).
        *   Checks the SIA process `returncode`.
        *   Parses the **guaranteed JSON output** from SIA's `stdout`. This JSON contains SIA's own detailed report: `{"status": "SUCCESS"|"FAILURE", "message": "...", "data": {...full_sia_operation_result...}}`.
        *   Logs SIA's `stderr` for debugging.
        *   If the SIA `data` payload indicates `code_updated_requires_restart: true` (meaning SIA updated itself or the Coordinator), a high-confidence belief (`system.restart_required.reason`) is added to the `BeliefSystem`, and a warning is logged.
    *   Updates the `improvement_campaign_history` with the high-level outcome of this SIA invocation.

5.  **Autonomous Improvement Loop (`_autonomous_improvement_worker`):**
    *   If enabled via configuration (`coordinator.autonomous_improvement.enabled`), this background `asyncio.Task` runs periodically.
    *   **Cycle Steps:**
        1.  **Analyze:** Calls `_process_system_analysis` to generate/refresh improvement suggestions, which are added to the backlog.
        2.  **Resource Check:** Queries `ResourceMonitor`. If system CPU usage is above `coordinator.autonomous_improvement.max_cpu_before_sia`, it defers attempting improvements for this cycle.
        3.  **Select Task from Backlog:** Picks the highest-priority "pending" item from the `improvement_backlog` that is not currently in a "cool-down" period (a configurable duration after a failed attempt on that same component).
        4.  **Human-in-the-Loop (HITL) for Critical Changes:**
            *   If `coordinator.autonomous_improvement.require_human_approval_for_critical` is true AND the selected target component is identified as critical (either by being in the `critical_components_for_approval` config list or flagged as `is_critical_target: true` by the analysis LLM) AND it hasn't been approved yet:
            *   The item's status in the backlog is changed to `pending_approval`.
            *   A prominent warning is logged, indicating human intervention is needed.
            *   The autonomous loop will *not* process this item further until it's manually approved (e.g., via a `approve_improvement <item_id>` CLI command to the Coordinator).
        5.  **Execute Improvement:** If the item is approved (or not critical / approval not required), the Coordinator marks the backlog item as `in_progress` (conceptually) and initiates a `COMPONENT_IMPROVEMENT` interaction for it (which in turn calls `_process_component_improvement_cli` to invoke SIA).
        6.  **Update Backlog Item:** Based on the detailed JSON result from the SIA, the backlog item's status is updated (e.g., `completed_sia_success`, `failed_sia_reported`). If an attempt failed, the `cool_down_store` for that component is updated to prevent immediate retries.
    *   The loop includes robust error handling and longer sleep intervals on unhandled exceptions to prevent rapid failure cycles.

6.  **Monitoring Integration & Internal Callbacks:**
    *   The Coordinator registers internal asynchronous callbacks (`handle_resource_alert`, `handle_resource_resolve`) with the `ResourceMonitor`.
    *   When the monitor detects high CPU/memory usage or subsequent resolution, these callbacks update the `BeliefSystem` with this information (e.g., `system_health.cpu.alert_active`). This data then enriches the context for the `_process_system_analysis` phase.
    *   `PerformanceMonitor` data is also actively pulled and summarized during `_process_system_analysis`.

7.  **Configuration-Driven Behavior:** All key operational parameters (LLM choices for coordinator tasks, SIA script path, timeouts, autonomous loop settings, critical components lists, HITL toggles) are sourced from `mindx.utils.config.Config`.

### Key Data Structures Managed

-   **`agent_registry`**: Dictionary of known agents/components. Metadata can include `script_path` for CLI-callable agents like SIA.
-   **`system_capabilities_cache`**: Output of `_scan_codebase_capabilities` (mapping `mindx.module.ClassOrFunction` to details like file path).
-   **`improvement_backlog`**: Persisted list of improvement suggestions (dictionaries). Each item includes `id` (UUID), `target_component_path`, `suggestion`, `priority`, `is_critical_target`, `status` (e.g., `pending`, `pending_approval`, `in_progress_sia`, `completed_sia_success`, `failed_sia_reported`, `rejected_manual`), `source`, `attempt_count`, `last_attempted_at`, `approved_at`, `rejected_at`, `sia_final_op_status`.
-   **`improvement_campaign_history`**: Persisted log of high-level outcomes of improvement "campaigns" (each typically corresponding to one or more SIA CLI invocations for a specific target).

## Technical Details

-   **Singleton Pattern:** Ensures a single Coordinator instance, accessed via `get_coordinator_agent_mindx_async()` (preferred for async contexts) or `get_coordinator_agent_mindx()`. `reset_instance_async()` is available for testing.
-   **Asynchronous Architecture:** Core logic is `async`, utilizing `asyncio` for managing concurrent operations, subprocesses, and the autonomous loop.
-   **CLI Invocation of SIA:** The `SelfImprovementAgent` is treated as a robust external tool. `asyncio.create_subprocess_exec` is used to run it from its script path.
    -   The `cwd` for the SIA subprocess is set to `PROJECT_ROOT` for consistent relative path handling within SIA if any were to occur (though SIA itself aims for absolute paths).
    -   A `tempfile.NamedTemporaryFile` is used to pass potentially large context strings to SIA via the `--context-file` CLI argument, with cleanup.
-   **Path Resolution (`_resolve_component_path_for_sia`):** Maps various forms of component identifiers (module paths like `mindx.core.belief_system`, registered agent IDs like `self_improve_agent_cli_mindx`, or direct file paths relative to `PROJECT_ROOT`) to the absolute file system path required by the SIA CLI.
-   **Concurrency Control for SIA:** An `asyncio.Semaphore` (`sia_concurrency_limit`) limits the number of concurrent SIA subprocesses to prevent system overload.
-   **Error Handling:** Extensive `try-except` blocks manage potential failures in LLM communications, subprocess execution, file operations, and JSON parsing, aiming to keep the Coordinator stable and provide informative logging.
-   **Persistence:** Backlog and campaign history are saved to JSON files in `PROJECT_ROOT/data/` on modification and during shutdown.

## Usage

The `CoordinatorAgent` is the primary interface for interacting with and managing the self-improvement capabilities of the MindX system. It is typically run via a script like `scripts/run_mindx_coordinator.py`.

**CLI Commands (via `run_mindx_coordinator.py`):**

-   `query <your question>`: For general queries to the Coordinator's LLM.
-   `analyze_system [optional context for analysis focus]`: Triggers a system-wide analysis. Generated improvement suggestions are added to the backlog.
-   `improve <component_id> [optional_context_for_improvement]`: Directly requests an improvement attempt on a specific component.
    -   `<component_id>`: e.g., `mindx.core.belief_system` or `self_improve_agent_cli_mindx`.
-   `backlog`: Displays the current improvement backlog items with their IDs, targets, priorities, and statuses.
-   `process_backlog`: Manually triggers processing of the highest-priority "pending" (and approved, if critical) item from the backlog.
-   `approve <backlog_item_id>`: Changes the status of a `pending_approval` backlog item to `pending`, allowing the autonomous loop (or manual `process_backlog`) to pick it up.
-   `reject <backlog_item_id>`: Changes the status of a `pending_approval` backlog item to `rejected_manual`.
-   `quit` or `exit`: Initiates a graceful shutdown of the Coordinator Agent and its associated tasks.

If the autonomous improvement loop is enabled in the configuration, the Coordinator will periodically perform system analysis and attempt to process items from the improvement backlog, respecting the Human-in-the-Loop flow for critical changes.
