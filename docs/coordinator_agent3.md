# Coordinator Agent (`coordinator_agent.py`) - Production Candidate Final

## Introduction

The `CoordinatorAgent` is the central orchestrator and strategic brain of the MindX system, an Augmentic Project initiative focused on AI self-improvement. It manages system-level operations, conducts analyses to identify improvement opportunities, maintains a persistent backlog of these opportunities, and delegates tactical code modifications to the specialized `SelfImprovementAgent` (SIA) via its Command Line Interface (CLI). A key feature is its autonomous improvement loop, which can proactively work through the backlog, incorporating Human-in-the-Loop (HITL) checks for changes to critical system components.

## Explanation

### Core Responsibilities & Workflow

1.  **Interaction Management (`Interaction` class, `handle_user_input`):**
    *   Serves as the primary interface for users or other agents to interact with MindX.
    *   Manages `Interaction` objects, each representing a unit of work (e.g., query, analysis request, improvement command). `InteractionType` includes `QUERY`, `SYSTEM_ANALYSIS`, `COMPONENT_IMPROVEMENT`, `APPROVE_IMPROVEMENT`, `REJECT_IMPROVEMENT`, and `ROLLBACK_COMPONENT`.
    *   Tracks the lifecycle (`InteractionStatus`) and history of each interaction.

2.  **System-Level Analysis (`_process_system_analysis`):**
    *   Can be triggered on demand or periodically by the autonomous loop.
    *   **Comprehensive Data Aggregation:**
        *   Scans the MindX codebase (`_scan_codebase_capabilities`) using Python's Abstract Syntax Trees (AST) to map out modules, classes, and functions.
        *   Integrates data from `ResourceMonitor` (CPU, memory, disk alerts) and `PerformanceMonitor` (LLM success rates, latencies, costs).
        *   Considers recent `improvement_campaign_history` for context.
    *   **LLM-Driven Strategic Insight:** Constructs a detailed prompt for its internal LLM (`self.llm_handler`). The LLM analyzes the aggregated system data to suggest 1-3 high-impact improvement areas.
    *   **Structured Suggestions:** LLM output is parsed as JSON, expecting suggestions with `target_component_path` (full Python module path like `mindx.learning.self_improve_agent`), a `suggestion` description, `priority` (1-10), and `is_critical_target` (boolean).
    *   Generated suggestions are added to the `improvement_backlog` using `add_to_improvement_backlog`.

3.  **Improvement Backlog Management:**
    *   `self.improvement_backlog`: A list of dictionaries, where each represents an improvement suggestion.
    *   `add_to_improvement_backlog()`: Adds new suggestions, assigns a unique ID, sets initial status to `PENDING`, and performs basic de-duplication against existing pending items. The backlog is sorted to prioritize: 1. Items `PENDING_APPROVAL`, 2. Highest `priority`, 3. Oldest `added_at`.
    *   **Persistence:** The backlog is loaded from and saved to `PROJECT_ROOT/data/improvement_backlog.json`.
    *   **Manual Approval/Rejection:** `_process_backlog_approval` handles `APPROVE_IMPROVEMENT` and `REJECT_IMPROVEMENT` interactions (e.g., from CLI commands `approve <id>` / `reject <id>`) to change the status of items marked `PENDING_APPROVAL`.

4.  **Delegation to SelfImprovementAgent (SIA) via CLI (`_process_component_improvement_cli`):**
    *   Handles `COMPONENT_IMPROVEMENT` interactions.
    *   **Path Resolution:** Uses `_resolve_component_path_for_sia` to map the `target_component` identifier (module path or specific agent ID) to an absolute file system path for the SIA.
    *   **SIA CLI Invocation:** Constructs and executes a command line call to `self_improve_agent.py` using `asyncio.create_subprocess_exec`.
        *   Passes: target file path, improvement context (from `analysis_context` in metadata, potentially via a temporary file if large), max SIA cycles, and any LLM/parameter overrides for SIA.
        *   Crucially includes `--output-json` for parseable results.
        *   Uses an `asyncio.Semaphore` (`sia_concurrency_limit`) to manage concurrent SIA processes.
    *   **Result Parsing:** Robustly parses the JSON output from SIA's `stdout`. The SIA CLI guarantees JSON output with top-level `status`, `message`, and `data` (containing detailed SIA cycle results). Checks SIA process `returncode` and logs `stderr`.
    *   **Critical Update Awareness:** If SIA's response data indicates `code_updated_requires_restart: true` (meaning SIA updated itself or the Coordinator), a high-confidence belief (`system.restart_required.reason`) is added to the `BeliefSystem`, and a prominent warning is logged.
    *   The outcome is logged in `improvement_campaign_history`.

5.  **Component Rollback (`_process_component_rollback_cli`):**
    *   Handles `ROLLBACK_COMPONENT` interactions.
    *   Currently designed to instruct the SIA to roll back *itself* using SIA's `--rollback [N]` CLI feature.
    *   Logs the outcome and signals that a system restart is likely required.

6.  **Autonomous Improvement Loop (`_autonomous_improvement_worker`):**
    *   If enabled (`coordinator.autonomous_improvement.enabled`), this background `asyncio.Task` runs periodically.
    *   **Cycle Steps:**
        1.  **Analyze:** Calls `_process_system_analysis` to refresh suggestions.
        2.  **Resource Check:** Queries `ResourceMonitor`. Defers SIA tasks if system CPU is high.
        3.  **Select Task:** Picks the highest-priority "pending" (and approved, if critical) item from the backlog not in a cool-down period.
        4.  **Human-in-the-Loop (HITL):** If `require_human_approval_for_critical` is true and the selected target is critical (from `critical_components_for_approval` config or `is_critical_target` flag from analysis), its backlog status becomes `PENDING_APPROVAL`. It won't be processed autonomously until approved.
        5.  **Execute:** If actionable, initiates a `COMPONENT_IMPROVEMENT` interaction (invoking SIA).
        6.  **Update Backlog:** Updates the backlog item's status (e.g., `COMPLETED_SUCCESS`, `FAILED_SIA_REPORTED`) and cool-down timer based on SIA's result.
    *   Includes error handling and adaptive sleep intervals.

7.  **Monitoring Integration & Callbacks:**
    *   Registers internal async callbacks with `ResourceMonitor` to update the `BeliefSystem` on high resource usage (`system_health.<resource>.alert_active`) or resolution events. This data informs `_process_system_analysis`.
    *   `PerformanceMonitor` data is actively pulled and summarized during `_process_system_analysis`.

8.  **Configuration-Driven:** Behavior is extensively controlled by settings in `mindx.utils.config.Config`.

## Technical Details

-   **Asynchronous Architecture:** Built on `asyncio`.
-   **CLI for Tactical Layer:** Uses `SelfImprovementAgent` CLI for robust, decoupled code modification.
-   **Path Management:** Relies on `PROJECT_ROOT` from `Config` for consistent file system access.
-   **Concurrency:** `asyncio.Semaphore` for SIA calls. Async file I/O in monitors and belief system.
-   **Persistence:** Backlog and campaign history are saved to JSON files in `PROJECT_ROOT/data/`.

## Usage

The `CoordinatorAgent` is started via `scripts/run_mindx_coordinator.py`.

**CLI Commands (for `run_mindx_coordinator.py`):**

-   `query <question>`
-   `analyze_system [focus_hint]`
-   `improve <component_id_or_module_path> [improvement_goal_for_SIA]`
-   `backlog` (View improvement backlog)
-   `process_backlog` (Manually run one pending item)
-   `approve <backlog_item_id>`
-   `reject <backlog_item_id>`
-   `rollback self_improve_agent_cli_mindx [N]` (Tell SIA to rollback itself to Nth backup)
-   `help`
-   `quit` / `exit`

Autonomous operations proceed if enabled in the configuration. The `CoordinatorAgent` serves as the intelligent "mission control" for MindX's self-evolution.
