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

# start mastermind mindX Augmentic Intelligence in autonomous mode
```
```bash
python3 -m venv mindX
```
```bash
source mindX/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
python3 scripts/mindX.py
```
