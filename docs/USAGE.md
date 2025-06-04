# MindX System Usage Guide (Augmentic Project)

This guide provides instructions on how to set up, configure, and use the MindX self-improving AI system.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

-   **Python:** Version 3.9 or higher.
-   **pip:** Python package installer (usually comes with Python).
-   **Git:** For cloning the repository (if applicable).
-   **(Recommended for Local LLM Use) Ollama:** If you plan to use local models via Ollama:
    -   Install Ollama from [ollama.ai](https://ollama.ai/).
    -   Pull the models you intend to use. For MindX development, these are good starting points:
        ```bash
        ollama pull deepseek-coder:6.7b-instruct 
        ollama pull nous-hermes2:latest 
        # or other models like llama3, gemma, etc.
        ```
    -   Ensure the Ollama server is running (usually `ollama serve` or started by the Ollama application).
-   **(Optional) Google Gemini API Key:** If you plan to use Google Gemini models, obtain an API key from Google AI Studio.

## 2. Setup and Installation

1.  **Clone the Repository (if you have one):**
    ```bash
    git clone <repository_url>
    cd augmentic_mindx 
    ```
    If you received the code as a directory, navigate into the `augmentic_mindx` root directory.

2.  **Create and Activate a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate   # On Windows PowerShell
    ```

3.  **Install Dependencies:**
    The project uses `pyproject.toml` to define dependencies. Install them using pip:
    ```bash
    pip install -e .[dev]
    ```
    -   The `-e .` installs the `mindx` package in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.
    -   `[dev]` installs both runtime and development dependencies (like `pytest` for testing and `ruff` for linting/formatting). If you only need to run MindX, you can omit `[dev]`: `pip install .`
    
    Alternatively, if you have a `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    # (And potentially pip install -r requirements-dev.txt for development tools)
    ```

## 3. Configuration

MindX uses a layered configuration system (`mindx/utils/config.py`). Settings are loaded with the following precedence (later sources override earlier ones):
    1. Initial code defaults.
    2. `mindx_config.json` file (optional, in project root or `data/config/`).
    3. `.env` file(s) in project root and then current working directory.
    4. Actual environment variables prefixed with `MINDX_`.

**Setting up `.env` (Most Common Configuration Method):**

1.  In the project root directory (`augmentic_mindx/`), create a file named `.env`.
    *(If an `.env.example` file exists, you can copy it to `.env` as a template).*
2.  Edit the `.env` file to specify your settings. This is where you should put **secrets like API keys**.

    **Example `.env` content:**
    ```env
    # --- General Logging ---
    MINDX_LOG_LEVEL="INFO" # Recommended: DEBUG, INFO, WARNING, ERROR, CRITICAL

    # --- Default LLM Provider for the whole system (can be overridden by specific agents) ---
    MINDX_LLM__DEFAULT_PROVIDER="ollama" # Options: "ollama", "gemini" (add more in llm_factory.py)

    # --- Ollama Specific Configuration (if default_provider or any agent uses ollama) ---
    MINDX_LLM__OLLAMA__DEFAULT_MODEL="nous-hermes2:latest" # General purpose model
    MINDX_LLM__OLLAMA__DEFAULT_MODEL_FOR_CODING="deepseek-coder:6.7b-instruct" # Model good at code
    # MINDX_LLM__OLLAMA__BASE_URL="http://localhost:11434" # Usually default

    # --- Gemini Specific Configuration (if default_provider or any agent uses gemini) ---
    # IMPORTANT: Get your API key from Google AI Studio
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE" # Non-prefixed, might be used by SDK directly if not found by MindX prefix
    MINDX_LLM__GEMINI__API_KEY="YOUR_GEMINI_API_KEY_HERE" # MindX prefixed
    MINDX_LLM__GEMINI__DEFAULT_MODEL="gemini-1.5-flash-latest"
    MINDX_LLM__GEMINI__DEFAULT_MODEL_FOR_CODING="gemini-1.5-pro-latest" # Or flash if pro is too slow/costly
    
    # --- SelfImprovementAgent (SIA) LLM Configuration ---
    # Specifies the LLM the SIA uses for its internal analysis and code generation.
    MINDX_SELF_IMPROVEMENT_AGENT__LLM__PROVIDER="ollama" 
    MINDX_SELF_IMPROVEMENT_AGENT__LLM__MODEL="deepseek-coder:6.7b-instruct"
    MINDX_SELF_IMPROVEMENT_AGENT__DEFAULT_MAX_CYCLES="1" # How many improvement iterations SIA runs per call
    MINDX_SELF_IMPROVEMENT_AGENT__CRITIQUE_THRESHOLD="0.6" # Min LLM critique score for a change to be good

    # --- CoordinatorAgent LLM Configuration ---
    # LLM used by Coordinator for tasks like system-wide analysis.
    MINDX_COORDINATOR__LLM__PROVIDER="ollama"
    MINDX_COORDINATOR__LLM__MODEL="nous-hermes2:latest"
    MINDX_COORDINATOR__SIA_CLI_TIMEOUT_SECONDS="900.0" # 15 minutes timeout for SIA subprocess call

    # --- Coordinator's Autonomous Improvement Loop ---
    MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__ENABLED="false" # Set to "true" to enable autonomous mode
    MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__INTERVAL_SECONDS="3600" # Check every 1 hour
    MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__REQUIRE_HUMAN_APPROVAL_FOR_CRITICAL="true"
    # Critical components list is in mindx/utils/config.py _set_final_derived_defaults, can be overridden by JSON config
    # Example: MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__CRITICAL_COMPONENTS='["mindx.learning.self_improve_agent", "mindx.orchestration.coordinator_agent"]' (JSON string list)

    # --- Monitoring ---
    MINDX_MONITORING__RESOURCE__ENABLED="true" # Set to true to activate resource monitor
    MINDX_MONITORING__RESOURCE__INTERVAL="15.0" # Check resources every 15 seconds
    MINDX_MONITORING__PERFORMANCE__ENABLE_PERIODIC_SAVE="true" # Enable periodic save for perf metrics
    MINDX_MONITORING__PERFORMANCE__PERIODIC_SAVE_INTERVAL_SECONDS="300" # Save perf metrics every 5 mins
    ```

## 4. Running the MindX System

The primary way to interact with and run the MindX system is through the `CoordinatorAgent`'s Command Line Interface (CLI).

**Start the Coordinator Agent:**

Navigate to the project root directory (`augmentic_mindx/`) in your terminal (with the virtual environment activated) and run:

```bash
python scripts/run_mindx_coordinator.py
Use code with caution.
Markdown
You should see log messages indicating initialization and then the MindX CLI > prompt.
Interacting via the MindX CLI:
Type help at the prompt to see available commands. Key commands include:
query <your question>
Sends a general query to the Coordinator's configured LLM.
Example: MindX CLI > query Explain the concept of Belief-Desire-Intention architecture.
analyze_system [optional context for analysis focus]
Triggers the Coordinator to perform a system-wide analysis using its LLM, codebase scan, and data from monitors.
Generated improvement suggestions are added to the Coordinator's internal improvement backlog.
Example: MindX CLI > analyze_system Focus on improving the error handling in the LLM interaction modules.
improve <component_id> [optional improvement goal/context for SIA]
Directly requests the Coordinator to initiate an improvement task for a specific component using the SelfImprovementAgent (SIA).
<component_id>: Can be a full Python module path (e.g., mindx.utils.config) or the special registered agent ID self_improve_agent_cli_mindx (instructs the SIA to attempt to improve its own code).
[optional improvement goal/context]: Textual guidance for the SIA's analysis and code generation.
Examples:
MindX CLI > improve mindx.core.belief_system Add a method to retrieve all belief keys matching a regex pattern.
MindX CLI > improve self_improve_agent_cli_mindx Enhance the detail in the JSON output for failed self-tests.
Use code with caution.
backlog
Displays the current list of improvement suggestions stored in the CoordinatorAgent's backlog, including their ID (first 8 characters), target component, priority, current status, and a snippet of the suggestion.
process_backlog
Manually tells the CoordinatorAgent to attempt to process the highest-priority actionable item from its improvement backlog. An item is actionable if its status is "PENDING" and, if it targets a critical component and HITL is enabled, it has been approved.
approve <backlog_item_id>
Approves a specific improvement item in the backlog that is currently in PENDING_APPROVAL status. Use the ID shown by the backlog command. This allows the autonomous loop (or manual process_backlog) to proceed with the critical improvement.
Example: MindX CLI > approve goal_abc123xy
reject <backlog_item_id>
Rejects a specific PENDING_APPROVAL item, changing its status to rejected_manual.
Example: MindX CLI > reject goal_def456uv
rollback <target_component_id> [N]
Instructs the Coordinator to request the SelfImprovementAgent (SIA) to roll back a component.
Currently, this is primarily designed for the SIA to roll back itself.
<target_component_id>: Should be self_improve_agent_cli_mindx or self.
[N]: Optional integer. The Nth latest backup to roll back to (1 is the most recent). Defaults to 1.
Note: After a successful rollback of SIA or Coordinator, the respective process needs a manual restart for the reverted code to take effect.
Example: MindX CLI > rollback self_improve_agent_cli_mindx 2 (Rollback SIA to 2nd latest backup)
quit or exit: Gracefully shuts down the CoordinatorAgent (and its background tasks like autonomous improvement and monitors) and exits the CLI.
Autonomous Mode
If MINDX_COORDINATOR__AUTONOMOUS_IMPROVEMENT__ENABLED="true" is set in your .env file, the CoordinatorAgent will:
Periodically run a SYSTEM_ANALYSIS.
Add any identified improvement suggestions to its backlog.
Periodically attempt to process the highest-priority actionable item from the backlog, respecting the Human-in-the-Loop (HITL) flow for critical components.
5. Standalone SelfImprovementAgent (SIA) CLI (for testing/direct use)
The SelfImprovementAgent can also be invoked directly via its own CLI. This is useful for testing its core code modification capabilities or for very targeted, manual improvement tasks. The CoordinatorAgent uses this same CLI interface internally.
Run the SIA CLI:
python mindx/learning/self_improve_agent.py <target_file_or_"self"> [options]
Use code with caution.
Bash
Key SIA CLI Options (see python mindx/learning/self_improve_agent.py --help for all):
target_file: Path to the Python file to improve, or the special string "self" to target the SIA's own script.
--context "<text>": Textual context for the improvement goal.
--context-file <path/to/context.txt>: Path to a file containing the context.
--cycles <N>: Number of Analyze-Implement-Evaluate cycles for the SIA to run.
--output-json: Output results as a single minified JSON object (default is indented).
--rollback [N]: (Only if target_file is "self") Rolls back the SIA script to its Nth latest backup (default N=1).
--llm-provider <name>, --llm-model <name>: Override LLM for this SIA run.
Example SIA CLI calls:
# Ask SIA to improve a utility function, providing context in a file
python mindx/learning/self_improve_agent.py mindx/utils/some_util.py --context-file my_improvement_goal.txt --output-json

# Ask SIA to try to improve itself based on a context string
python mindx/learning/self_improve_agent.py self --context "Improve the LLM prompt used for code critique in the evaluate_improvement method."
Use code with caution.
Bash
The SIA CLI will always output a detailed JSON object to stdout summarizing its operation and result.
6. Data and Logs
Logs: Application logs are stored in data/logs/mindx_system.log. Log level is controlled by MINDX_LOG_LEVEL in .env.
SIA Data: The SelfImprovementAgent stores its operational data (iteration directories, fallbacks, history) under data/self_improvement_work_sia/<agent_script_name_stem>/.
Coordinator Data: The CoordinatorAgent stores its improvement_backlog.json and improvement_campaign_history.json in the data/ directory.
Performance Metrics: PerformanceMonitor saves its data to data/performance_metrics.json.
7. Development & Testing
Linting/Formatting: Use ruff check . and ruff format . (after pip install ruff).
Type Checking: Use mypy mindx/ (after pip install mypy).
Tests: (Placeholder) Unit and integration tests would reside in the tests/ directory and be run with pytest.
This guide should help you get started with running and interacting with the MindX self-improvement system. Remember that this is an experimental project, and direct code modification by AI carries inherent risks. Proceed with awareness and appropriate safeguards, especially if running with broad permissions or on critical codebases.
