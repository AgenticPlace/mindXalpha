# mindX Augmentic Intelligence System - Instructions & Overview (RC1)

## 1. Introduction

Welcome to the mindX system, an Augmentic Intelligence platform designed for autonomous operation, self-improvement, and strategic evolution. This document provides an overview of its architecture, how to set up your environment, run the system, and key ways to interact with it.

The core principle of mindX is to create an AI system that can not only perform tasks but also understand its own structure, identify areas for improvement, and even modify or extend its own codebase and capabilities over time.

## 2. Core Architecture Overview

MindX is built as a multi-agent system with a hierarchical control structure:

*   **Utility Layer:** Provides foundational services:
    *   `utils.Config`: Centralized configuration management loading from defaults, JSON files, `.env` files, and direct environment variables. Defines `PROJECT_ROOT`.
    *   `utils.logging_config`: Standardized logging setup for console and rotating files, configurable via `Config`.
    *   `llm.llm_factory` & `llm.LLMHandlerInterface`: An abstraction layer for interacting with various Large Language Models (LLMs). Specific handlers (e.g., `GeminiHandler`, `OllamaHandler`, `GroqHandler`) implement the interface.
    *   `core.belief_system.BeliefSystem`: A shared, persistent knowledge base for agents.
*   **Monitoring Layer:**
    *   `monitoring.ResourceMonitor`: Tracks system CPU, memory, and disk usage.
    *   `monitoring.PerformanceMonitor`: Tracks LLM call performance metrics.
*   **Tactical Execution Layer (Code Modification):**
    *   `learning.SelfImprovementAgent (SIA)`: The "code surgeon" invoked via CLI by the `CoordinatorAgent` to make specific code changes. It includes analysis, implementation, evaluation (self-tests, LLM critique), and rollback features.
*   **Strategic Layer (Improvement Campaign Management):**
    *   `learning.strategic_evolution_agent.StrategicEvolutionAgent (SEA)`: (Conceptual for RC1, to be fully integrated) Manages long-term self-improvement campaigns using an internal `BDIAgent`.
*   **Orchestration Layer (System-Wide Coordination):**
    *   `orchestration.CoordinatorAgent`: The central hub for managing interactions, an improvement backlog, autonomous improvement cycles (with Human-in-the-Loop for critical changes), and SIA CLI invocation.
    *   `orchestration.MastermindAgent`: The apex strategic agent. Oversees system evolution, manages the `official_tools_registry.json`, initiates strategic campaigns via its internal `BDIAgent`, and utilizes `BaseGenAgent` for codebase analysis.
*   **Tools Layer (`tools` package):**
    *   Contains reusable tools (e.g., `NoteTakingTool`, `SummarizationTool`, `WebSearchTool`, `BaseGenAgent`). BDIAgents load and use these based on configuration and the official tool registry.
*   **Identity Management (`core.id_manager_agent`):**
    *   Manages cryptographic identities for agents/tools, supporting future security and integration needs.

## 3. Setup and Prerequisites

### 3.1. Python Environment

*   Ensure Python 3.9+ is installed.
*   It is **highly recommended** to use a Python virtual environment to manage dependencies for this project.
    ```bash
    # Navigate to your project root (e.g., /home/luvai/mindX)
    cd /path/to/your/mindX_project_root

    # Create a virtual environment (e.g., named 'venv' or 'mindX_venv')
    python3 -m venv venv

    # Activate the virtual environment
    # On Linux/macOS:
    source venv/bin/activate
    # On Windows:
    # venv\Scripts\activate
    ```
    You should see `(venv)` (or your chosen name) at the beginning of your terminal prompt.

### 3.2. Install Dependencies

With your virtual environment activated, install the required Python packages:
```bash
pip install -r requirements.txt
Use code with caution.
Markdown
Ensure your requirements.txt file includes at least:
python-dotenv (for .env file handling)
httpx (for async HTTP requests, used by some LLM handlers and tools)
psutil (for system resource monitoring)
google-generativeai (if using Gemini LLM)
groq (if using Groq LLM)
ollama (if using the Ollama Python client for local LLMs)
pathspec (for BaseGenAgent to process .gitignore files)
eth-account (for IDManagerAgent cryptographic functions)
If requirements.txt is not comprehensive, you can install them individually:
pip install python-dotenv httpx psutil google-generativeai groq pathspec eth-account
# Add 'ollama' if you plan to use it
Use code with caution.
Bash
3.3. Configuration Files
mindX uses a layered configuration approach.
A. .env File (for Secrets and High-Level Overrides)
This is the primary place for API keys and sensitive information. It can also override settings from JSON configuration files.
In your project root (/home/luvai/mindX/), copy the sample environment file:
cp .env.sample .env
Use code with caution.
Bash
Edit the newly created .env file with your specific settings.
API Keys are CRUCIAL. Fill these in based on the LLM providers you intend to use.
Logging settings here will take precedence. Set MINDX_LOGGING_FILE_ENABLED="true" to create local log files.
Example .env content (refer to your .env.sample for all options):
# --- API Keys ---
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
# GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
# GOOGLE_SEARCH_API_KEY="YOUR_GOOGLE_CSE_API_KEY"
# GOOGLE_SEARCH_ENGINE_ID="YOUR_GOOGLE_CSE_ID"

# --- Logging Configuration ---
MINDX_LOGGING_LEVEL="DEBUG" # For verbose logs during development
MINDX_LOGGING_FILE_ENABLED="true"
MINDX_LOGGING_FILE_DIRECTORY="data/logs"
MINDX_LOGGING_FILE_NAME="mindx_debug.log"
MINDX_LOGGING_FILE_LEVEL="DEBUG"

# --- Agent LLM Preferences (can override JSON defaults) ---
# MINDX_LLM_DEFAULT_PROVIDER="gemini"
# MINDX_MASTERMIND_AGENT_LLM_PROVIDER="gemini"
# MINDX_MASTERMIND_AGENT_LLM_MODEL="gemini-1.5-pro-latest"
# MINDX_COORDINATOR_LLM_PROVIDER="gemini"
# MINDX_COORDINATOR_LLM_MODEL="gemini-1.5-flash-latest"
# MINDX_BDI_DEFAULT_LLM_PROVIDER="gemini"
# MINDX_BDI_DEFAULT_LLM_MODEL="gemini-1.5-flash-latest"
Use code with caution.
Env
B. JSON Configuration Files (in data/config/)
These files provide more detailed and structured configuration. Settings in .env (with the MINDX_ prefix) can override values from these JSON files.
data/config/basegen_config.json:
This is intended as the main system-wide configuration file.
It should contain default settings for logging, LLM providers (which model to use for Gemini, Groq, etc.), agent-specific parameters (like autonomous loop enablement for Coordinator/Mastermind), BDI agent defaults (LLM, tools), monitoring settings, and BaseGenAgent settings.
Ensure this file is valid JSON. Check for syntax errors like missing quotes around keys or trailing commas. The system will report errors if it cannot parse this file.
data/config/llm_factory_config.json:
This file allows fine-tuning of the LLMFactory's behavior, especially the preference order for LLM providers when no specific provider is requested.
It can also specify default models for the factory to use when constructing handlers if no other configuration provides one.
To ensure Gemini is used as the primary LLM for testing:
{
  "default_provider_preference_order": [
    "gemini",
    "groq",
    "ollama"
  ],
  "gemini_settings_for_factory": {
    "default_model_override": "gemini-1.5-flash-latest"
  },
  "provider_specific_handler_config": {
    "gemini": {
      "default_model_for_api_call": "gemini-1.5-flash-latest"
    }
  }
}
Use code with caution.
Json
data/config/official_tools_registry.json:
This file, managed by MastermindAgent, serves as the canonical list of all recognized tools in the mindX ecosystem.
It details each tool's ID, description, module path, class name, capabilities (with input/output schemas), dependencies, status, version, and other metadata relevant for both agent use and potential future integrations (like NFTs).
You can pre-populate this with known tools or let Mastermind develop and register tools over time.
3.4. Directory Structure and __init__.py Files
Python relies on __init__.py files to recognize directories as packages. Ensure your project root (/home/luvai/mindX/) has the following structure, and that each listed directory that is part of your application code contains an (even if empty) __init__.py file:
/home/luvai/mindX/
├── core/
│   ├── __init__.py
│   └── (bdi_agent.py, belief_system.py, id_manager_agent.py, etc.)
├── data/
│   ├── config/
│   │   ├── basegen_config.json
│   │   ├── llm_factory_config.json
│   │   └── official_tools_registry.json
│   ├── logs/ (will be created by logger)
│   └── (other data directories like mastermind_work, temp_sia_contexts, etc.)
├── learning/
│   ├── __init__.py
│   └── (goal_management.py, plan_management.py, self_improve_agent.py, etc.)
├── llm/
│   ├── __init__.py
│   └── (llm_factory.py, llm_interface.py, gemini_handler.py, mock_llm_handler.py, etc.)
├── monitoring/
│   ├── __init__.py
│   └── (performance_monitor.py, resource_monitor.py)
├── orchestration/
│   ├── __init__.py
│   └── (coordinator_agent.py, mastermind_agent.py)
├── scripts/
│   └── run_mindx.py
├── tools/
│   ├── __init__.py
│   └── (base_gen_agent.py, note_taking_tool.py, summarization_tool.py, web_search_tool.py, etc.)
├── utils/
│   ├── __init__.py
│   └── (config.py, logging_config.py)
├── mindX/ (This is your Python Virtual Environment - Venv)
│   ├── bin/
│   ├── lib/
│   └── ...
├── .env
├── .env.sample
├── requirements.txt
└── README.md
Use code with caution.
4. Running the System
Activate your virtual environment (if not already active).
Navigate to the project root directory:
cd /home/luvai/mindX
Use code with caution.
Bash
Run the main script:
python3 scripts/run_mindx.py
Use code with caution.
Bash
Upon successful startup, you will see initialization logs from various components and finally the prompt: mindX (Mastermind) >
5. Command-Line Interface (CLI) Commands
At the mindX (Mastermind) > prompt, type help to see the most current list of available commands.
Key Mastermind-Level Commands:
evolve <directive>: The primary way to give high-level strategic tasks to MastermindAgent. It will use its internal BDIAgent and LLM (Gemini) to decompose, plan, and execute.
Examples:
evolve Develop a comprehensive test suite for the core.belief_system module.
evolve Assess current tool suite and conceptualize a new tool for advanced data visualization.
evolve Analyze the tools.web_search_tool for potential security vulnerabilities and propose fixes.
mastermind_status: Shows Mastermind's current objectives and campaign history.
show_tool_registry: Displays the content of official_tools_registry.json.
analyze_codebase <path_to_code> [focus_prompt]: Uses BaseGenAgent to generate a Markdown summary of the specified codebase, then uses an LLM to interpret that summary based on the focus. The results are stored in Mastermind's beliefs.
Example: analyze_codebase ./orchestration "Identify dependencies between CoordinatorAgent and MastermindAgent."
Coordinator-Level Commands (Prefixed with coord_)
These allow more direct interaction with the CoordinatorAgent's functionalities.
coord_query <your question>: Ask a question to the Coordinator's LLM.
coord_analyze [optional context]: Trigger Coordinator's system analysis to generate improvement suggestions for the backlog.
coord_improve <component_id> [optional context]: Directly task the Coordinator to attempt an improvement on a component using SIA.
<component_id> is a Python module path like utils.config or learning.self_improve_agent.
coord_backlog: View the Coordinator's improvement backlog.
coord_process_backlog: Tell the Coordinator to try processing one item from its backlog.
coord_approve <backlog_item_id> / coord_reject <backlog_item_id>: For Human-in-the-Loop approval/rejection of improvements for critical components.
General Commands:
help: Displays this list of commands.
quit / exit: Gracefully shuts down the mindX system and exits the CLI.
6. Observing System Behavior (Logs)
Console: By default, logs at the level specified in your config/.env (e.g., INFO or DEBUG) are printed.
Log File: If MINDX_LOGGING_FILE_ENABLED="true" in .env (or logging.file.enabled: true in JSON config), logs are written to the file specified (e.g., data/logs/mindx_debug.log).
DEBUG Level: For development and deep insight into agent reasoning (LLM prompts, generated plans, belief updates), set MINDX_LOGGING_LEVEL="DEBUG" in your .env or logging.level: "DEBUG" in basegen_config.json. This will make the log file very verbose and useful.
7. Understanding Autonomous Self-Building & Tool Extrapolation
Vision: MastermindAgent drives the evolution of the mindX system, including its own toolset.
Workflow for Creating/Improving a Tool (Conceptual):
Strategic Goal: Mastermind receives or formulates a high-level goal (e.g., "Enhance data analysis capabilities" or "Integrate external service X as a mindX tool").
Assessment (ASSESS_TOOL_SUITE_EFFECTIVENESS): Mastermind's BDI uses its LLM to analyze the current official_tools_registry.json, system objectives, and (future) performance data. It identifies gaps or tools needing improvement.
Conceptualization (CONCEPTUALIZE_NEW_TOOL):
If a new tool is needed, Mastermind's BDI uses its LLM to generate a detailed specification for this tool (ID, description, capabilities with I/O schemas, module/class name hints, prompt_template_for_llm_interaction, metadata, needs_identity).
For tool extrapolation, the "Identified Need/Gap" fed to this action would include information about the external software (e.g., from an ANALYZE_CODEBASE_FOR_STRATEGY run on the external software's docs or code). The LLM would be prompted to design a mindX wrapper tool, specifying how mindX capabilities map to the external software's API/CLI.
Identity Provisioning (if needs_identity: true): Mastermind's BDI uses REQUEST_NEW_ENTITY_IDENTITY (via IDManagerAgent) to get a cryptographic identity for the new tool. This identity info is added to the tool's concept.
Initiate Coding (INITIATE_TOOL_CODING_CAMPAIGN): Mastermind's BDI tasks the CoordinatorAgent with a COMPONENT_IMPROVEMENT interaction.
Target: The suggested .py file for the new/existing tool (e.g., tools/new_data_analyzer.py).
Context: The full tool specification JSON.
Tactical Code Generation (Coordinator & SIA):
CoordinatorAgent invokes the SelfImprovementAgent (SIA) CLI.
SIA uses its LLM to write/modify the Python code for the tool, aiming to implement the specified capabilities and BaseTool interface. It runs tests (if applicable).
Review & Registry Update (REGISTER_OR_UPDATE_TOOL_IN_REGISTRY):
SIA returns its results (success/failure, paths, diffs) to the Coordinator, which relays them to Mastermind.
Mastermind's BDI analyzes SIA's output. If successful, it calls REGISTER_OR_UPDATE_TOOL_IN_REGISTRY to add/update the tool's entry in official_tools_registry.json, marking its status (e.g., "experimental" or "active").
Usage & Monitoring: Other BDIAgents can now discover (by reading the registry, potentially via Coordinator) and use the new/updated "active" tool. PerformanceMonitor tracks its usage. The cycle repeats.
This iterative process allows mindX to strategically expand its internal capabilities by reasoning about needs, designing solutions, and orchestrating their implementation.
