# MindX System - Technical Deep Dive (Augmentic Project)

##  Core Architecture

MindX is designed as a multi-agent system with a hierarchical control structure aimed at achieving autonomous self-improvement. The key architectural layers and components are:

**A. Utility Layer:**
    -   **`mindx.utils.Config`**: Centralized configuration management loading from defaults, JSON, `.env`, and environment variables. Defines `PROJECT_ROOT`.
    -   **`mindx.utils.logging_config`**: Standardized logging setup for console and rotating files.
    -   **`mindx.llm.llm_factory` & `LLMHandler`**: Provides an abstraction layer for interacting with various Large Language Models (LLMs). Specific handlers (e.g., `OllamaHandler`, `GeminiHandler`) implement the `LLMInterface` (conceptual) and are created by the factory.
    -   **`mindx.core.belief_system.BeliefSystem`**: A shared, persistent knowledge base for agents. Beliefs are namespaced and include confidence, source, and optional TTL. Uses JSON for persistence.

**B. Monitoring Layer:**
    -   **`mindx.monitoring.ResourceMonitor`**: Asynchronously tracks system CPU, memory, and multi-path disk usage using `psutil`. Triggers callbacks on threshold breaches and resolutions, with alert debouncing. Data from this feeds into strategic analysis.
    -   **`mindx.monitoring.PerformanceMonitor`**: Tracks performance metrics (latency, success rate, token counts, cost, error types) for LLM calls, contextualized by model, task type, and initiating agent. Persists data to JSON and provides reporting.

**C. Tactical Execution Layer (Code Modification):**
    -   **`mindx.learning.SelfImprovementAgent` (SIA)**: The specialized "code surgeon."
        -   **CLI Interface:** Its primary interaction point. Accepts a target file path (or "self"), an improvement context/goal, and LLM/operational overrides. Always outputs structured JSON.
        -   **Internal Cycle:**
              **Analyze (`analyze_target`):** Uses its LLM to analyze the target file and context to propose a specific improvement description.
              **Implement (`implement_improvement`):** Uses its LLM to generate the complete new code for the target file based on the description. Generates a diff.
              **Evaluate (`evaluate_improvement`):**
                *   Syntax check (Python `compile()`).
                *   **Self-Tests (for self-modification):** If modifying its own code, it copies the modified code to an *iteration directory*, then executes this copy as a subprocess with a `--self-test-mode` flag. The self-test mode (within SIA's `main_cli`) performs basic sanity checks and reports success/failure via JSON. This is the critical "proof-of-work."
                *   **LLM Critique:** Uses its LLM to score the change against the original goal.
              **Outcome & Promotion (for self-modification):**
                *   If evaluation fails, changes in the iteration directory are discarded (or reverted if an external file was modified directly, though SIA primarily works on copies for self-mod).
                *   If successful self-modification:
                    *   Backs up the current main SIA script (versioned backups with a manifest).
                    *   Promotes the validated code from the iteration directory to replace the main SIA script.
                    *   Sets a `code_updated_requires_restart: true` flag in its output JSON.
        -   **Rollback:** Supports a `--rollback [N]` CLI command to revert its own script to the Nth latest backup.

**D. Strategic Layer (Improvement Campaign Management):**
    -   **`mindx.learning.strategic_evolution_agent.StrategicEvolutionAgent` (SEA)**: (Formerly AGISelfImprovementAgent)
        -   **Campaign Management:** Orchestrates long-term, multi-step self-improvement campaigns based on high-level objectives (e.g., "Improve system robustness").
        -   **Internal `BDIAgent`:** Uses an instance of `mindx.core.bdi_agent.BDIAgent` for its internal strategic planning and execution loop. The BDI agent's plan consists of strategic actions.
        -   **Internal `SystemAnalyzerTool`:** Used as a BDI action to scan the codebase and prompt an LLM for specific improvement suggestions related to the current campaign goal.
        -   **Delegation for Tactical Work:** When its BDI plan determines a specific code modification is needed, the SEA makes a `COMPONENT_IMPROVEMENT` request to the `CoordinatorAgent`.
        -   **Outcome Evaluation:** Evaluates the results of tactical SIA executions (as reported by the Coordinator) in the context of its overall campaign goal, potentially using its LLM for higher-level critique.

**E. Orchestration Layer (System-Wide Coordination):**
    -   **`mindx.orchestration.coordinator_agent.CoordinatorAgent`**: The central hub.
        -   **Interaction Handling:** Manages user and inter-agent requests (`Interaction` objects).
        -   **System Analysis:** Can perform its own system-wide analysis (`_process_system_analysis`) by combining codebase scans, monitor data, and LLM insights to generate improvement suggestions.
        -   **Improvement Backlog:** Maintains a persistent, prioritized backlog of improvement suggestions (from its own analysis or other sources like the SEA).
        -   **Autonomous Improvement Loop:** Periodically reviews the backlog and can autonomously initiate `COMPONENT_IMPROVEMENT` tasks.
        -   **Human-in-the-Loop (HITL):** For improvements targeting critical components (configurable list), it can set backlog items to `PENDING_APPROVAL`, requiring manual CLI approval before proceeding.
        -   **SIA CLI Invocation:** For `COMPONENT_IMPROVEMENT` tasks, it resolves the target component to a file path and invokes the `SelfImprovementAgent` CLI as a subprocess, managing context files and parsing the SIA's JSON output.
        -   **Rollback Trigger:** Can invoke the SIA's `--rollback` CLI command if, for example, post-update monitoring by the Coordinator indicates a problem.
        -   **Restart Signaling:** Recognizes the `code_updated_requires_restart` flag from SIA's output and updates beliefs/logs accordingly.

##  Agent Interaction Options & Data Flow

This outlines primary interaction patterns, focusing on self-improvement.

**A. User/External System Initiated Improvement (via Coordinator CLI):**

  **User Input:** `improve mindx.utils.config Enhance error logging for missing keys` (sent to `scripts/run_mindx_coordinator.py`).
  **`run_mindx_coordinator.py`:** Parses input, calls `CoordinatorAgent.handle_user_input()` with `InteractionType.COMPONENT_IMPROVEMENT` and metadata:
    ```json
    {
        "target_component": "mindx.utils.config",
        "analysis_context": "Enhance error logging for missing keys." 
    }
    ```
  **`CoordinatorAgent.process_interaction()` -> `_process_component_improvement_cli()`:**
    a.  Resolves `"mindx.utils.config"` to `/path/to/project/mindx/utils/config.py`.
    b.  Constructs SIA CLI command: `python .../self_improve_agent.py /path/to/config.py --context "Enhance..." --output-json ...`
    c.  (If context is large) Writes context to a temporary file, passes `--context-file` to SIA.
    d.  Limits concurrent SIA calls using `sia_concurrency_limit`.
    e.  Executes SIA CLI via `asyncio.create_subprocess_exec`.
  **`SelfImprovementAgent.main_cli()` (Separate Process):**
    a.  Parses its CLI args.
    b.  Instantiates `SelfImprovementAgent`.
    c.  Calls `agent.improve_external_target(Path("/path/to/config.py"), ...)` (since target is not "self").
    d.  This runs internal `run_self_improvement_cycle()`:
        i.  `analyze_target()`: LLM proposes detailed change.
        ii. `implement_improvement()`: LLM generates new code for `config.py`, diff is created, new code saved (to `config.py` directly as it's external target).
        iii. `evaluate_improvement()`: Syntax check, LLM critique. (No self-tests for external target).
    e.  `main_cli()` formats the detailed result from `improve_external_target` into the standard JSON output:
        ```json
        // stdout from SIA CLI
        {
            "status": "SUCCESS", // or "FAILURE"
            "message": "SIA operation summary...",
            "data": { /* full result from improve_external_target/improve_self */ }
        }
        ```
  **`CoordinatorAgent._process_component_improvement_cli()` (Resumes):**
    a.  Reads SIA's `stdout` and `stderr`, checks `returncode`.
    b.  Parses the JSON from SIA's `stdout`.
    c.  Updates `Interaction` object with SIA's result.
    d.  Logs to `improvement_campaign_history`.
    e.  Checks `sia_result["data"].get("code_updated_requires_restart")`. If true (e.g., if SIA improved itself or the Coordinator), logs warning and sets `system.restart_required` belief.
6.  **`run_mindx_coordinator.py`:** Prints Coordinator's final interaction result (which contains SIA's result).

**B. Coordinator's Autonomous Improvement Loop:**

  **`CoordinatorAgent._autonomous_improvement_worker()` (Periodic Task):**
    a.  Calls `self._process_system_analysis()`:
        i.  `_scan_codebase_capabilities()`: AST scan.
        ii. Fetches summaries from `ResourceMonitor` & `PerformanceMonitor`.
        iii. Fetches recent `improvement_campaign_history`.
        iv. Prompts Coordinator's LLM for improvement suggestions.
        v.  LLM returns `{"improvement_suggestions": [...]}`.
        vi. `add_to_improvement_backlog()`: Adds new suggestions to `self.improvement_backlog` and saves it.
    b.  Checks `ResourceMonitor` (e.g., CPU usage). If high, defers.
    c.  Selects highest priority "pending" item from `self.improvement_backlog` (not in cool-down).
    d.  **HITL Check:** If item targets a critical component (from `critical_components_for_approval` config or `is_critical_target` flag in suggestion) AND `require_human_approval_for_critical` is true AND item is not yet approved:
        i.  Sets backlog item status to `PENDING_APPROVAL`.
        ii. Logs warning. Skips to next cycle.
    e.  If item is actionable (pending & approved if critical):
        i.  Updates backlog item status to `IN_PROGRESS`.
        ii. Creates a `COMPONENT_IMPROVEMENT` `Interaction` (metadata includes backlog item ID, suggestion as context for SIA).
        iii. Calls `self.process_interaction()` for this new interaction (which triggers `_process_component_improvement_cli` as above).
        iv. Updates the backlog item's status (e.g., `COMPLETED_SUCCESS`, `FAILED_SIA_REPORTED`), cool-down timer based on SIA's outcome.
        v.  Saves backlog.

**C. Human Approving/Rejecting a Backlog Item (via Coordinator CLI):**

  User: `approve <backlog_item_id>`
  `run_mindx_coordinator.py` -> `CoordinatorAgent.handle_user_input()` with `InteractionType.APPROVE_IMPROVEMENT`.
  `CoordinatorAgent.process_interaction()` -> `_process_backlog_approval(item_id, approve=True)`:
    a.  Finds item in `self.improvement_backlog`.
    b.  If status is `PENDING_APPROVAL`, changes to `PENDING` and sets `approved_at`.
    c.  Saves backlog. Returns success message.
    d.  (Autonomous loop will now be able to pick it up).

**D. StrategicEvolutionAgent (SEA) Initiating an Improvement:**
    *(This assumes SEA is registered with and called by the Coordinator, or runs as a peer)*

  **SEA (`manage_improvement_campaign`):**
    a.  Sets a high-level campaign goal for its internal `BDIAgent`.
    b.  Runs its BDI agent.
  **SEA's Internal `BDIAgent` (`run` cycle):**
    a.  **Plan Execution:** Its plan might include an action like:
        `{"type": "REQUEST_SIA_MODIFICATION", "params": {"target_component_path": "mindx.utils.config", "improvement_goal_for_sia": "Add new config validation method.", ...}}`
    b.  This action is dispatched to `SEA._sea_action_request_coordinator_for_sia_execution()`.
  **`SEA._sea_action_request_coordinator_for_sia_execution()`:**
    a.  Constructs `metadata` for a `COMPONENT_IMPROVEMENT` interaction.
    b.  Calls `self.coordinator_agent.handle_user_input(...)` (acting as the `agent_id` for this interaction).
    c.  This triggers Flow A (steps 3-5) within the `CoordinatorAgent`.
  **SEA (Resumes):**
    a.  Receives the SIA's outcome (via Coordinator's response) from the BDI action handler.
    b.  Its BDI agent updates beliefs and decides on the next strategic step (e.g., evaluate outcome, pick another target, conclude campaign).

**E. Post-Update Failure & Rollback (Conceptual, SIA focused):**

  **External Monitoring (Coordinator or Human):** After SIA reports `SUCCESS_PROMOTED` for *itself* (i.e., `self_improve_agent.py` was updated), the system might become unstable on the *next* run of SIA or if Coordinator tries to use a (now broken) SIA.
  **Triggering Rollback:**
    *   **Manual:** User types `rollback self_improve_agent_cli_mindx [N]` into Coordinator's CLI.
    *   **Automated (Advanced):** Coordinator's `ResourceMonitor` or `PerformanceMonitor` (or a dedicated watchdog) detects persistent critical errors specifically after an SIA self-update belief. It then programmatically creates a `ROLLBACK_COMPONENT` interaction.
  **`CoordinatorAgent.process_interaction()` -> `_process_component_rollback_cli()`:**
    a.  Resolves target to `self_improve_agent.py`'s absolute path.
    b.  Constructs SIA CLI command: `python .../self_improve_agent.py self --rollback [N] --output-json ...`
    c.  Executes SIA CLI.
  **`SelfImprovementAgent.main_cli()` (Separate Process, but now running the *promoted, potentially faulty* code):**
    a.  Parses `--rollback` argument.
    b.  Calls `agent._revert_to_nth_fallback(N)`.
        i.  SIA first calls `_backup_current_self` (to save the current faulty version for forensics).
        ii. Reads its `backup_manifest.jsonl` to find the Nth latest *good* backup.
        iii. Copies that good backup over `SELF_AGENT_ABSOLUTE_PATH`.
    c.  `main_cli()` outputs JSON: `{"status": "SUCCESS", "message": "Rollback initiated. Restart REQUIRED.", ...}`.
  **`CoordinatorAgent` (Resumes):**
    a.  Receives SIA's rollback success message.
    b.  Logs critical warning: "SIA self-rollback executed. MAIN SYSTEM RESTART or at least next SIA call will use reverted code."
    c.  Sets strong `system.restart_required.reason` belief.

##  Data Flow & Persistence

-   **Configuration:** `Config` loads from multiple sources, provides unified access.
-   **Beliefs:** `BeliefSystem` stores dynamic agent knowledge, persisted to `data/mindx_beliefs.json`. Used for inter-agent communication and contextual reasoning.
-   **SIA History:** `data/self_improvement_work_sia/<agent_name>/archive/improvement_history.jsonl` logs every SIA file modification cycle attempt.
-   **SIA Fallbacks:** `data/self_improvement_work_sia/<agent_name>/fallback_versions/` stores previous working versions of the SIA script itself. `backup_manifest.jsonl` tracks these.
-   **Coordinator Backlog:** `data/improvement_backlog.json` stores prioritized improvement suggestions.
-   **Coordinator Campaign History:** `data/improvement_campaign_history.json` logs high-level outcomes of SIA campaigns dispatched by Coordinator.
-   **SEA Campaign History:** `data/sea_campaign_history_<sea_agent_id>.json` logs campaigns managed by a specific SEA instance.
-   **Performance Metrics:** `data/performance_metrics.json` stores LLM call performance data.
-   **Application Logs:** `data/logs/mindx_system.log` for general system logging.

##  Key Algorithms & Heuristics

-   **SIA Self-Test (`_run_self_test_suite`):** Subprocess execution of the modified SIA script with a `--self-test-mode` flag. The self-test logic within `main_cli` performs basic sanity checks. *This is a critical "proof-of-work" for self-updates.*
-   **SIA LLM Critique (`evaluate_improvement`):** LLM scores generated code against the original improvement goal.
-   **Coordinator System Analysis (`_process_system_analysis`):** LLM analyzes codebase structure, monitor data, and history to suggest improvements. Relies heavily on prompt engineering for quality suggestions.
-   **Coordinator Backlog Prioritization:** Currently sorts by `PENDING_APPROVAL` status, then numeric `priority`, then `added_at` timestamp.
-   **Coordinator Autonomous Loop Cool-down:** Prevents rapid retries of failed improvements on the same component.
-   **BDI/SEA Planning (LLM-driven):** The BDI agent (used internally by SEA) prompts an LLM to generate plans (sequences of strategic actions) based on its current goal and beliefs. This is a highly flexible but LLM-dependent planning approach.
-   **Dynamic Parameter Resolution (SEA/BDI):** Action parameters in SEA's BDI plans can reference prior action results or beliefs (e.g., `"$belief.key"`, `"$last_action_result.field"`), making plans adaptive.

##  Scalability & Future Considerations

-   **LLM as a Bottleneck/Single Point of Failure:** Heavy reliance on LLMs means their availability, cost, and potential biases directly impact the system.
-   **Complexity of State:** Managing distributed state (beliefs, backlog, histories) and ensuring consistency can become challenging.
-   **True Autonomy & Emergence:** While the system can autonomously identify and apply improvements, the "intelligence" for *what* constitutes a good improvement or a good plan still largely comes from the LLMs and the engineered prompts. True emergent strategies are a long-term goal.
-   **Multi-File Changes:** Current SIA focuses on single-file changes. Complex refactoring across multiple files is a significant future challenge.
-   **System Restart Management:** A robust external supervisor mechanism is needed for truly seamless recovery and application of updates to core running agents like the Coordinator or a continuously running SIA.
