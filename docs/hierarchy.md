# mindX Augmentic Intelligence: System Hierarchy and Workflow

This document outlines the command and control structure of the `mindX` system, detailing the roles of its core agents and the workflow for a typical self-improvement campaign. The architecture is designed to be a robust, hierarchical, and modular multi-agent system.

## Summary of Agent Hierarchy

The system operates on a clear chain of command, with each agent fulfilling a specialized role.

*   **`MastermindAgent` - The Commander-in-Chief**
    *   The highest-level agent and the primary singleton for the entire system. It is the main interface for human operators or external triggers. Its role is to set broad, strategic directives for the `mindX` system, such as "Evolve the system" or "Develop a new tool for code analysis."

*   **`BDIAgent` - The Tactical Officer**
    *   An instance of a Belief-Desire-Intention agent that is owned and commanded by the `MastermindAgent`. It receives high-level directives from the Mastermind and uses its LLM-powered planning capabilities to break them down into concrete, single-step actions. It does not execute complex campaigns itself but knows which specialized agent to command for the task.

*   **`StrategicEvolutionAgent` (SEA) - The Field General**
    *   A powerful, specialized tool that is commanded by the `BDIAgent`. The SEA is not an autonomous agent itself but a "Campaign Manager." When invoked, it executes a complete, end-to-end evolution campaign for a specific goal. It has its own internal `PlanManager` to orchestrate the multiple steps required for a strategic change, from analysis to final evaluation.

*   **`CoordinatorAgent` - The Operations Officer**
    *   A low-level service agent responsible for managing the execution of tactical tasks and maintaining system-wide state. It acts as the bridge between strategic requests and ground-level execution, handling tasks like calling the code-modification script, managing the improvement backlog, and monitoring system resources.

*   **`SelfImprovementAgent` (SIA) - The Ground Unit**
    *   A robust and safety-oriented script that is the system's "tool" for performing code modifications. It is invoked by the `CoordinatorAgent` via a command-line interface. Its sole purpose is to safely analyze, modify, and test a single target file based on a specific directive, complete with backups and self-testing protocols.

## Detailed Workflow: A Self-Improvement Campaign

This workflow details the sequence of events when a high-level directive like "Evolve the system" is given.

*   **Phase 1: Directive and Strategic Planning**
    - A human operator or an automated trigger issues a directive to the `**MastermindAgent**`.
    - The `**MastermindAgent**` accepts the directive and sets it as a high-priority goal for its internal `**BDIAgent**`.
    - The `**BDIAgent**` begins its deliberation cycle. It sees the abstract goal and consults its LLM-powered planning module.
    - Based on its new doctrine, the planner determines that the correct action for such a goal is `EXECUTE_STRATEGIC_EVOLUTION_CAMPAIGN`. It creates a simple, one-step plan containing this action.

*   **Phase 2: Campaign Execution Command**
    - The `**BDIAgent**` executes its plan.
    - The `_execute_strategic_evolution_campaign` action handler is called.
    - This handler invokes the `run_evolution_campaign` method on the `**StrategicEvolutionAgent**` instance, passing the campaign goal.

*   **Phase 3: Strategic Plan Execution (Internal to SEA)**
    - The `**StrategicEvolutionAgent**` (SEA) receives the command and takes control of the campaign.
    - It uses its internal LLM handler to generate its own multi-step, strategic plan. This plan includes actions like:
        - `REQUEST_SYSTEM_ANALYSIS`
        - `SELECT_IMPROVEMENT_TARGET`
        - `FORMULATE_SIA_TASK_GOAL`
        - `REQUEST_COORDINATOR_FOR_SIA_EXECUTION`
        - `EVALUATE_SIA_OUTCOME`
    - The SEA uses its internal `PlanManager` to begin executing this strategic plan step-by-step.

*   **Phase 4: Tactical Delegation**
    - When the SEA's plan reaches the `REQUEST_COORDINATOR_FOR_SIA_EXECUTION` action, the SEA's action dispatcher calls the `**CoordinatorAgent**`.
    - The request contains the specific file to be modified and a precise description of the change needed (e.g., "Add error handling to the `_save_beliefs` method in `core/belief_system.py`").

*   **Phase 5: Ground-Level Implementation**
    - The `**CoordinatorAgent**` receives the tactical request.
    - It resolves the component path to an absolute file path.
    - It constructs and executes a command-line call to the `self_improve_agent.py` script.
    - The `**SelfImprovementAgent**` (SIA) script runs independently. It creates backups, analyzes the code, generates the proposed changes, runs syntax and self-tests, and, if successful, replaces the original file.
    - The SIA script outputs a final JSON result summarizing its success or failure.

*   **Phase 6: Reporting and Conclusion**
    - The `**CoordinatorAgent**` captures the JSON output from the SIA script and returns it as the result of the interaction.
    - The `**StrategicEvolutionAgent**` receives this result from the Coordinator. It updates its internal beliefs and continues its plan, potentially moving to the `EVALUATE_SIA_OUTCOME` step.
    - Once the SEA's entire internal strategic plan is complete, its `run_evolution_campaign` method returns a final campaign summary object.
    - This summary is returned to the `**BDIAgent**` as the successful result of its `EXECUTE_STRATEGIC_EVOLUTION_CAMPAIGN` action.
    - The `**BDIAgent**` marks its goal as complete and reports its success to the `**MastermindAgent**`.
    - The `**MastermindAgent**` logs the final outcome of the entire evolution campaign, concluding the process.
