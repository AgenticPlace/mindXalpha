# Self-Improvement Agent (SIA) (`self_improve_agent.py`) - Production Candidate v3.x (Finalized Core)

## Introduction

The `SelfImprovementAgent` (SIA) is the core tactical execution engine for code modifications within the MindX system (Augmentic Project). It is engineered to operate on individual Python files, including its own source code, to analyze, generate, evaluate, and safely apply improvements. This version represents a mature iteration, emphasizing robust CLI interaction, comprehensive safety mechanisms for self-modification (iteration directories, self-tests, versioned backups, Nth fallback), and detailed JSON output for seamless integration with orchestrators like the `CoordinatorAgent`.

## Explanation

### Core Responsibilities & Workflow

The SIA performs a cycle of operations for each improvement attempt:

1.  **Initialization (`__init__`):**
    *   Accepts overrides for LLM provider/model, max improvement cycles, self-test timeout, and critique score threshold, primarily passed from CLI arguments. These take precedence over global `Config` settings.
    *   Initializes its `LLMHandler` based on the resolved configuration.
    *   Determines its own absolute script path (`SELF_AGENT_ABSOLUTE_PATH`) and filename (`SELF_AGENT_FILENAME`).
    *   Establishes its dedicated working directory (`SELF_IMPROVEMENT_BASE_DIR` under `PROJECT_ROOT/data/`) for `archive` (history logs) and `fallback_versions` (backups of its own script).
    *   The `is_iteration_instance` flag prevents recursive self-improvement if an instance is already running from a temporary iteration directory.

2.  **Target Analysis (`analyze_target`):**
    *   Takes an absolute path to a target Python file, optional additional text context, log snippets, and an `improvement_goal_hint`.
    *   Reads file content (handling new/unreadable files, truncating if very large).
    *   Constructs a detailed prompt for its LLM to propose *one specific, actionable improvement description*.
    *   Returns the textual description or an error message.

3.  **Improvement Implementation (`implement_improvement`):**
    *   Takes the target file path, the improvement description, and original file content.
    *   Prompts its LLM to generate the *complete, new Python code* for the entire target file.
    *   Robustly cleans LLM output (e.g., removing markdown code fences).
    *   Calculates a `diff` patch using `difflib.unified_diff`.
    *   Saves the new code to the target file (which is a copy in an iteration directory for self-improvement).
    *   Returns `(success_bool, new_code_or_error_str, diff_patch_str_or_None)`.

4.  **Evaluation (`evaluate_improvement`):**
    *   Assesses the quality and correctness of the implemented change.
    *   **Syntax Check:** Attempts to `compile()` the new code.
    *   **Self-Tests (for self-improvement only):** If modifying its own code, invokes `_run_self_test_suite` on the script copy in the iteration directory. This executes the script with `--self-test-mode`.
    *   **LLM Self-Critique:** If prior checks pass, prompts an LLM to review the change (old vs. new code snippets against the goal) and provide a quality score (0.0-1.0) and justification.
    *   Returns a dictionary of evaluation results.

5.  **Improvement Cycle Orchestration (`run_self_improvement_cycle`):**
    *   Manages a single Analyze -> Implement -> Evaluate sequence.
    *   **Self-Identification:** Uses `SELF_AGENT_ABSOLUTE_PATH` to determine if the target is its own script.
    *   **Iteration Directory Workflow (for self-improvement):**
        *   Creates a unique iteration directory.
        *   Copies its own current source code into this directory.
        *   All modifications and evaluations for this cycle happen on this copy.
    *   **Decision & Reversion (of working file):** If implementation fails, or evaluation is poor (syntax error, failed self-tests, critique score < `self.critique_threshold`), changes to the working file are reverted.
    *   **Promotion (for successful self-improvement):**
        1.  If self-improvement and evaluation passed: Backs up the current *main* agent script (using `_backup_current_self` with a descriptive reason).
        2.  Copies the successfully modified code from the iteration directory to replace the main agent script.
        3.  Sets `code_updated_requires_restart: true` in the result.
        *   If promotion fails, attempts to restore the main script from the immediate backup, then from pre-cycle content.
    *   Records the outcome in `IMPROVEMENT_LOG_FILE`.

6.  **Campaign Management Methods (`improve_self`, `improve_external_target`):**
    *   Orchestrate multiple `run_self_improvement_cycle` calls, up to `self.max_self_improve_cycles`.
    *   Manage context flow between cycles.
    *   Return a summary of the multi-cycle operation.

### Versioning and Rollback Mechanisms

-   **Versioned Backups (`_backup_current_self`):**
    *   Called before promoting a self-update or before executing a rollback.
    *   Creates a timestamped backup file (e.g., `self_improve_agent.py.YYYYMMDD_HHMMSS_reason.bak`) in `fallback_versions/`.
    *   Logs backup metadata (path, timestamp, reason, current agent version) to `fallback_manifest.jsonl`.
-   **Nth Fallback (`_get_latest_fallback_version(N)`, `_revert_to_nth_fallback(N, reason)`):**
    *   Allows reverting the main agent script to the Nth latest backup recorded in the manifest.
    *   `_revert_to_nth_fallback` first backs up the current (potentially faulty) state before overwriting with the older version.
    *   This mechanism is exposed via the `--rollback [N]` CLI argument.

### CLI Interface (`main_cli`)

The SIA's `main_cli` function is its primary public interface.

-   **Comprehensive Arguments:**
    -   `target_file`: Path or `"self"`.
    -   `--context`/`--context-file`: For analysis context.
    -   `--logs`: Paths to log files.
    -   `--llm-provider`, `--llm-model`, `--cycles`, `--self-test-timeout`, `--critique-threshold`: Override configured SIA parameters.
    -   `--rollback [N]`: New argument to trigger rollback of 'self' to the Nth backup (default N=1).
    -   `--output-json`: Minified JSON output (default is indented).
    -   `--self-test-mode`: Internal. Executes basic sanity checks and outputs JSON status.
-   **Guaranteed JSON Output:**
    *   The CLI *always* prints a single JSON object to `stdout`.
    *   Top-level structure: `{"status": "SUCCESS" | "FAILURE", "message": "string", "data": {...detailed_result...}}`.
    *   Even if internal Python exceptions occur, `main_cli` catches them and formats them into this JSON structure.
-   **Exit Codes:** `0` for overall `SUCCESS`, `1` for `FAILURE`.

## Technical Details

-   **Paths:** `pathlib.Path` used. `SELF_AGENT_ABSOLUTE_PATH` and `SELF_IMPROVEMENT_BASE_DIR` are key.
-   **Configuration:** `mindx.utils.config.Config` used, with CLI args having highest precedence for SIA's operational parameters.
-   **Subprocess for Self-Tests:** `asyncio.create_subprocess_exec` with `sys.executable`.
-   **Diffs:** `difflib.unified_diff`.
-   **JSONL Archive:** `improvement_history.jsonl` for cycle-by-cycle attempt logging.
-   **Error Handling:** Robust `try-except` blocks in `main_cli` and core methods to ensure structured JSON output and graceful failure reporting.

## Usage

Primarily via CLI, invoked by `CoordinatorAgent` or for manual testing/recovery.

**CLI Examples:**

1.  **Improve external file:**
    ```bash
    python mindx/learning/self_improve_agent.py path/to/module.py --context "Optimize for speed." --output-json
    ```
2.  **Self-improve SIA:**
    ```bash
    python mindx/learning/self_improve_agent.py self --context "Improve clarity of log messages during evaluation." --cycles 1
    ```
3.  **Rollback SIA to the latest backup:**
    ```bash
    python mindx/learning/self_improve_agent.py self --rollback
    ```
4.  **Rollback SIA to the second latest backup:**
    ```bash
    python mindx/learning/self_improve_agent.py self --rollback 2 --output-json
    ```

The SIA is a sophisticated worker agent, providing a safe, verifiable, and auditable process for modifying Python code, forming the tactical execution backbone of MindX's self-evolution strategy.
