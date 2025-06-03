# Self-Improvement Agent (SIA) (`self_improve_agent.py`) - Production Candidate v2

## Introduction

The `SelfImprovementAgent` (SIA) is a core worker component of the MindX system (Augmentic Project), specifically engineered for the tactical execution of code improvements. It can target any Python file within the project, including its own source code, for enhancement. The SIA operates with a strong emphasis on safety and verifiability, employing mechanisms like iteration directories for changes, self-tests as proof-of-work for self-updates, and fallback capabilities. It is designed to be callable via a Command Line Interface (CLI), allowing other agents (like the `CoordinatorAgent`) or developers to request specific improvement tasks. This version represents a production candidate with robust error handling and clearly defined interfaces.

## Explanation

### Core Responsibilities & Workflow

The SIA performs a cycle of operations for each improvement attempt, which can be part of a larger campaign of multiple cycles:

1.  **Initialization (`__init__`):**
    *   Sets up its unique `agent_id` and loads global configurations via `mindx.utils.config.Config`.
    *   Accepts overrides for LLM provider/model, maximum improvement cycles, self-test timeout, and critique score threshold. These overrides are typically passed from CLI arguments and take precedence over `Config` values.
    *   Initializes an `LLMHandler` for communication with the chosen LLM.
    *   Determines its own absolute script path (`SELF_AGENT_ABSOLUTE_PATH`) and filename (`SELF_AGENT_FILENAME`), which are critical for identifying and managing self-improvement tasks.
    *   Establishes a dedicated working directory structure (`SELF_IMPROVEMENT_BASE_DIR`, typically `PROJECT_ROOT/data/self_improvement_work_sia/<agent_script_name_stem>/`) containing subdirectories for `archive` (improvement history) and `fallback_versions` (backups of its own code).
    *   The `is_iteration_instance` flag (internal, not CLI) prevents recursive self-improvement attempts if an instance is already running from a temporary iteration directory.

2.  **Target Analysis (`analyze_target`):**
    *   Takes an absolute path to a target Python file, optional additional text context, log snippets, and an `improvement_goal_hint`.
    *   Reads the target file content (handling new or unreadable files gracefully). Content is truncated if it exceeds a configured length (`self_improvement_agent.analysis.max_code_chars`) to fit LLM context windows.
    *   Constructs a detailed prompt for an LLM, including the (potentially truncated) file content and all provided context.
    *   The LLM's task is to analyze this information and propose *one specific, actionable improvement description*.
    *   Returns the textual improvement description or an error message string if analysis fails.

3.  **Improvement Implementation (`implement_improvement`):**
    *   Takes the target file path, the improvement description, and (optionally) the original file content.
    *   Constructs a prompt instructing an LLM to generate the *complete, new Python code* for the entire target file that incorporates the described improvement.
    *   The LLM is explicitly instructed to return *only raw Python code* without explanations or markdown fences.
    *   Robustly cleans the LLM's output to remove common artifacts (e.g., \`\`\`python ... \`\`\`).
    *   Calculates a `diff` patch between the original and new content using `difflib.unified_diff`.
    *   Saves the new code to the target file. For self-improvement, this target file is a *copy of the agent's script within a temporary iteration directory*. For external targets, it's the actual specified file.
    *   Returns a tuple: `(success_boolean, new_code_string_or_error_message_string, diff_patch_string_or_None)`.

4.  **Evaluation (`evaluate_improvement`):**
    *   This crucial step assesses the quality and correctness of the implemented change.
    *   **Syntax Check:** First, it attempts to `compile()` the new code to catch basic Python syntax errors. If this fails, evaluation stops, and the change is considered failed.
    *   **Self-Tests (for self-improvement only):** If `is_self_improvement` is true (meaning the target was the agent's own code, now modified in an iteration directory), it invokes `_run_self_test_suite`. This executes the modified agent script (from the iteration directory) as a subprocess with a special `--self-test-mode` flag. The self-test mode (implemented within `main_cli`) performs predefined sanity checks. If these self-tests fail, the self-improvement attempt is rejected.
    *   **LLM Self-Critique:** If prior checks pass, it prompts an LLM to review the change (snippets of old vs. new code against the improvement goal) and provide a quality score (0.0-1.0) and a textual justification.
    *   Returns a dictionary of evaluation results including `passed_syntax_check`, `passed_self_tests`, `llm_self_critique_score`, and `notes`.

5.  **Improvement Cycle Orchestration (`run_self_improvement_cycle`):**
    *   Manages a single iteration of the Analyze -> Implement -> Evaluate sequence.
    *   **Self-Identification:** Robustly determines if the `target_file_path_conceptual` refers to its own script (`SELF_AGENT_ABSOLUTE_PATH`).
    *   **Iteration Directory Workflow (for self-improvement):**
        *   If it's a self-improvement attempt and this instance is not already an iteration instance, it creates a unique timestamped iteration directory.
        *   It copies its own current source code into this iteration directory.
        *   All subsequent analysis, implementation, and evaluation for this cycle happen on this *copy*. The main agent script remains untouched during these stages.
    *   **Decision & Reversion (of working file):** If implementation fails, or if evaluation is poor (syntax error, failed self-tests for self-improvement, LLM critique score below `self.critique_threshold`), the changes made to the *working file* (the copy in the iteration directory for self-improvement, or the actual external target file) are reverted to its pre-implementation state.
    *   **Promotion (for successful self-improvement):**
        1.  If it was a self-improvement attempt AND the evaluation of the code in the iteration directory passed all checks:
        2.  The current *main* agent script (`SELF_AGENT_ABSOLUTE_PATH`) is backed up.
        3.  The successfully modified and evaluated code from the iteration directory is copied over, replacing the main agent script ("promotion").
        4.  A flag `code_updated_requires_restart: true` is set in the cycle result to signal the `CoordinatorAgent`.
        *   If promotion itself fails, it attempts to restore the main agent script from the immediate backup or its pre-cycle content.
    *   **Recording:** The outcome of the entire cycle is recorded in `IMPROVEMENT_LOG_FILE`.

6.  **Campaign Management Methods (`improve_self`, `improve_external_target`):**
    *   These orchestrate multiple `run_self_improvement_cycle` calls, up to `self.max_self_improve_cycles`.
    *   They manage the flow of context between cycles.
    *   Return a summary dictionary of the overall multi-cycle operation.

### CLI Interface (`main_cli`)

The SIA's `main_cli` function provides a robust command-line interface.

-   **Arguments:** Comprehensive arguments allow fine-grained control:
    -   `target_file`: Path to the target Python file or the special string `"self"`.
    -   `--context`, `--context-file`: Provides textual context for analysis.
    -   `--logs`: Paths to log files for additional context.
    -   `--llm-provider`, `--llm-model`: Overrides LLM configuration.
    -   `--cycles`: Overrides the number of improvement cycles.
    -   `--self-test-timeout`, `--critique-threshold`: Overrides specific SIA evaluation parameters.
    -   `--output-json`: If present, outputs minified JSON; otherwise, indented JSON.
    -   `--self-test-mode`: Internal flag. When the SIA script is run with this flag, it executes predefined sanity checks and exits with a JSON status. This is used by the `_run_self_test_suite` method.
-   **Output:** *Critically, the CLI always outputs a single JSON object to `stdout`*, regardless of internal success or failure. This JSON object has a consistent top-level structure:
    -   `"status": "SUCCESS" | "FAILURE"`: Overall status of the CLI command's execution.
    -   `"message": "Human-readable summary"`
    -   `"data": {...}`: Contains the detailed dictionary result from the `improve_self` or `improve_external_target` method. This includes per-cycle results, diffs, evaluation data, and promotion status for self-improvements.
-   **Exit Codes:** Exits with `0` if the overall operation was successful (e.g., `SUCCESS_PROMOTED` or `SUCCESS_EVALUATED`). Exits with `1` for any failure condition. This structured output and exit code behavior are essential for reliable programmatic parsing by the `CoordinatorAgent`.

## Technical Details

-   **Path Management:** Uses `pathlib.Path` for robust file system operations. `SELF_AGENT_ABSOLUTE_PATH` (its own script path) and `SELF_IMPROVEMENT_BASE_DIR` (its dedicated data directory, typically under `PROJECT_ROOT/data/`) are key anchors. All critical paths are resolved to be absolute.
-   **LLM Interaction:** Uses `LLMHandler` (via `create_llm_handler` from `mindx.llm.llm_factory`) for all LLM communications.
-   **Configuration:** Relies on `mindx.utils.config.Config` for its settings, with CLI arguments providing overrides.
-   **Subprocess for Self-Tests:** `_run_self_test_suite` uses `asyncio.create_subprocess_exec` with `sys.executable` to run the modified agent script in a separate Python process for testing.
-   **Diff Generation:** Employs `difflib.unified_diff` to create standard textual patches.
-   **JSONL Archive:** `improvement_history.jsonl` stores a log of all improvement attempts (each cycle result is a JSON object per line).
-   **Error Handling:** Extensive `try-except` blocks ensure that errors are caught and reported within the final JSON output of the CLI, rather than crashing the script.

## Usage

The SIA is primarily designed for CLI invocation, typically by the `CoordinatorAgent`.

**Example CLI Invocations:**

1.  **Improve an external file:**
    ```bash
    python mindx/learning/self_improve_agent.py /path/to/project/mindx/core/some_module.py \
        --context "Refactor the process_data function for speed by using NumPy vectorization if possible." \
        --cycles 1 \
        --llm-model "deepseek-coder:33b-instruct" \
        --output-json 
    ```

2.  **Trigger self-improvement for the agent itself:**
    ```bash
    python mindx/learning/self_improve_agent.py self \
        --context "The LLM critique prompt in evaluate_improvement needs to be more specific about expecting only JSON output." \
        --critique-threshold 0.7 \
        --output-json
    ```

The `SelfImprovementAgent` is a powerful, safety-conscious component that forms the executive arm of MindX's self-evolution capabilities. Its well-defined CLI and robust internal mechanisms enable complex, autonomous code modification.
