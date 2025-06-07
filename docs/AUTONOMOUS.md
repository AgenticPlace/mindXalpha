Phase 1: Enabling the Autonomous Loops in Configuration
The first step is to tell the agents they are allowed to operate autonomously. This is done in your main configuration file (e.g., /home/luvai/mindX/data/config/basegen_config.json).
Enable Coordinator's Autonomous Improvement Loop:
This loop makes the CoordinatorAgent periodically:
Analyze the system (using its LLM, codebase scans, monitor data) to generate new improvement suggestions for its backlog.
Process existing items in its improvement_backlog, potentially tasking the SelfImprovementAgent (SIA) for code changes.
Handle Human-in-the-Loop (HITL) for critical changes.
In basegen_config.json:
```.json
{
  // ... other configurations ...
  "coordinator": {
    "llm": {
      "provider": "gemini", // Or your preferred
      "model": "gemini-1.5-flash-latest"
    },
    "autonomous_improvement": {
      "enabled": true,  // <--- SET THIS TO true
      "interval_seconds": 3600, // e.g., check backlog/analyze every 1 hour
      "cooldown_seconds_after_failure": 7200, // Wait 2 hours before retrying a failed component
      "max_cpu_before_sia": 85.0,
      "critical_components": [
        "learning.self_improve_agent",
        "orchestration.coordinator_agent",
        "orchestration.mastermind_agent",
        "core.bdi_agent",
        "utils.config"
      ],
      "require_human_approval_for_critical": true // Keep true for safety
    }
    // ... other coordinator settings ...
  }
  // ...
}
```
# Enable Mastermind's Autonomous Strategic Loop:
This loop makes the MastermindAgent periodically:
Execute its BDI agent with a default high-level directive (e.g., "Proactively monitor and evolve mindX...").
This can lead to actions like ASSESS_TOOL_SUITE_EFFECTIVENESS, CONCEPTUALIZE_NEW_TOOL, ANALYZE_CODEBASE_FOR_STRATEGY, or formulating new strategic campaign goals that might then task the Coordinator.
In basegen_config.json:
```json
{
  // ... other configurations ...
  "mastermind_agent": {
    "default_agent_id": "mastermind_prime_augmentic",
    "llm": {
      "provider": "gemini",
      "model": "gemini-1.5-pro-latest"
    },
    "tools_registry_path": "data/config/official_tools_registry.json",
    "autonomous_loop": {
        "enabled": true, // <--- SET THIS TO true
        "interval_seconds": 14400, // e.g., every 4 hours
        "default_directive": "Proactively monitor mindX, assess tool suite effectiveness, identify strategic evolutionary opportunities for components and tools, and initiate campaigns to enhance overall system health, capabilities, and efficiency based on current state and long-term goals."
    }
  }
  // ...
}
```
# Phase 2: Starting the System
Ensure API Keys and Dependencies:
Your GEMINI_API_KEY (and any other necessary keys like Google Search for WebSearchTool) are in your .env file.
All Python dependencies (pip install -r requirements.txt) are installed in your active virtual environment.
Run the Main Script:
cd /home/luvai/mindX
python3 scripts/run_mindx.py
Use code with caution.
Bash
What Happens When You Start (with autonomous loops enabled):
Initialization:
All agents (MastermindAgent, CoordinatorAgent, BDIAgent instances, Monitors, CodeBaseGenerator, IDManagerAgent, etc.) will initialize as before.
You'll see logs like:
Coordinator: Autonomous improvement loop started. Interval: 3600s.
Mastermind (mastermind_prime_augmentic of mindX): mindX autonomous loop started. Interval: 14400s.
PerformanceMonitor: Periodic performance metrics saver started...
Coordinator's Autonomous Loop in Action:
First Cycle (after interval_seconds):
It will call _process_system_analysis. This involves using its LLM (Gemini) to analyze the current system state (based on codebase scan data, monitor summaries, etc.) and generate improvement_suggestions.
These suggestions are added to data/improvement_backlog.json.
It will then look at the backlog. If there are actionable PENDING items (and not in cool-down, and approved if critical), it will:
Update the item's status to IN_PROGRESS.
Create a COMPONENT_IMPROVEMENT interaction.
Call self.process_interaction(), which invokes _process_component_improvement_cli.
This, in turn, runs the SelfImprovementAgent (SIA) CLI as a subprocess to attempt the code modification.
After SIA finishes, the Coordinator updates the backlog item's status based on SIA's success or failure.
Subsequent Cycles: The Coordinator will repeat this process: analyze, check backlog, process actionable items.
Mastermind's Autonomous Loop in Action:
First Cycle (after its interval_seconds):
It will call self.manage_mindx_evolution() with its default_directive.
Its internal BDIAgent will start a reasoning cycle:
Goal: The default_directive.
Decomposition: The BDI's LLM (Gemini) will break this directive into subgoals.
Planning: For each subgoal, the BDI's LLM will generate a plan of actions (e.g., OBSERVE_MINDX_SYSTEM_STATE, ASSESS_TOOL_SUITE_EFFECTIVENESS, CONCEPTUALIZE_NEW_TOOL, ANALYZE_CODEBASE_FOR_STRATEGY, INITIATE_TOOL_CODING_CAMPAIGN, REGISTER_OR_UPDATE_TOOL_IN_REGISTRY, or tasking the Coordinator via LAUNCH_IMPROVEMENT_CAMPAIGN_VIA_COORDINATOR).
Execution: The BDI agent executes these planned actions.
If an action involves LAUNCH_IMPROVEMENT_CAMPAIGN_VIA_COORDINATOR, this effectively puts a high-level task/suggestion into the Coordinator's sphere (either directly or via an analysis that populates the backlog).
If it's INITIATE_TOOL_CODING_CAMPAIGN, it tasks the Coordinator to run SIA for tool code generation.
If it's REGISTER_OR_UPDATE_TOOL_IN_REGISTRY, Mastermind directly modifies its official_tools_registry.json.
Subsequent Cycles: Mastermind will repeat its strategic cycle.
Interaction Between Mastermind and Coordinator Loops:
Mastermind as a Source for Coordinator's Backlog: When Mastermind's BDI decides a component needs improvement or a new tool needs to be developed by SIA, it will typically do so by creating an interaction that the CoordinatorAgent processes. This interaction (e.g., from LAUNCH_IMPROVEMENT_CAMPAIGN_VIA_COORDINATOR or INITIATE_TOOL_CODING_CAMPAIGN) might lead to the Coordinator adding specific, actionable items to its own improvement_backlog.
Coordinator as Tactical Executor: The Coordinator's autonomous loop then picks up these items (or items from its own analysis) and manages the tactical execution with SIA.
Feedback: The results of SIA campaigns managed by the Coordinator are logged. Mastermind's BDI, through its OBSERVE_MINDX_SYSTEM_STATE (which includes history summaries) and REVIEW_CAMPAIGN_OUTCOMES actions, can become aware of these results to inform its next strategic cycle.
Phase 3: "Seeding" or Guiding Initial Autonomous Evolution (Optional)
While the autonomous loops will eventually start generating their own tasks, you can "kickstart" or guide the initial direction by:
Providing a specific, high-impact evolve directive via the CLI:
mindX (Mastermind) > evolve Strategically enhance the system's error handling and reporting capabilities across all core modules.
Use code with caution.
or
mindX (Mastermind) > evolve Analyze the current 'tools' package and propose a new tool to automate daily system health checks, then initiate its development and register it.
Use code with caution.
This gives Mastermind a strong initial focus.
Manually adding high-priority items to data/improvement_backlog.json:
If you have specific small improvements you want the Coordinator to tackle autonomously, you can pre-populate its backlog. The Coordinator's loop will then pick these up.
Monitoring the Process:
Log Files: Set MINDX_LOGGING_LEVEL="DEBUG" and MINDX_LOGGING_FILE_ENABLED="true" in your .env. The data/logs/mindx_debug.log will be your primary window into what each agent is thinking and doing. Look for:
BDI agent goal decomposition and planning steps.
LLM prompts and responses.
Coordinator's backlog processing.
SIA CLI invocations and their output.
Tool registry updates.
CLI Commands: Periodically use mastermind_status, show_tool_registry, and coord_backlog to check the state of the system.
Persistent Files: Examine data/improvement_backlog.json, data/mastermind_objectives.json, data/config/official_tools_registry.json, and any Markdown files generated by BaseGenAgent.
Important Considerations for Autonomous Operation:
LLM Costs and Rate Limits: Autonomous loops making frequent LLM calls can incur costs and hit rate limits. Monitor this.
Prompt Quality: The effectiveness of the entire system heavily depends on the quality of the prompts given to the LLMs for analysis, planning, and code generation. This will be an ongoing area of refinement.
Safety and HITL: Keep require_human_approval_for_critical enabled, especially in early stages. Carefully define your critical_components.
Testing and Validation: The SIA's self-tests are vital. For other components, you'll need external testing strategies.
Resource Consumption: The ResourceMonitor will help, but complex operations can be resource-intensive.
By enabling these autonomous loops and potentially seeding Mastermind with an initial directive, you set mindX on its path to self-directed evolution. Be prepared to monitor, debug, and refine!
