# Logic Engine (`logic_engine.py`) - v3 Enhanced for MindX Agent Reasoning
mindX/utils/logic_engine.py

## Introduction

The `logic_engine.py` module provides an enhanced suite of tools for incorporating formal logical reasoning within MindX agents (Augmentic Project). This version (v3) significantly improves upon basic expression evaluation by offering:

-   A more robust `SafeExpressionEvaluator` with expanded capabilities.
-   A `LogicalRule` class for defining IF-THEN rules or consistency constraints.
-   A `LogicEngine` class that manages rules and default assumptions, performs basic forward-chaining inference on beliefs from the shared `BeliefSystem`, checks for consistency against constraints, and can leverage a Large Language Model (LLM) for Socratic questioning to challenge beliefs and assumptions.

This engine aims to help MindX agents make more principled, verifiable, and "logical" choices, particularly useful when operating with smaller LLMs (in fallback modes) or when strict adherence to predefined operational rules is paramount.

## Explanation

### Core Components

-   **`SafeExpressionEvaluator` Class:**
    -   **Purpose:** Safely evaluates Python-like expressions from strings by parsing them into an Abstract Syntax Tree (AST) and processing only whitelisted operations and node types. This is the cornerstone for preventing arbitrary code execution risks inherent in direct `eval()`.
    -   **Key Enhancements:**
        *   **Expanded Operators (`ALLOWED_OPERATORS`):** Now includes basic arithmetic (`+`, `-`, `*`, `/`, `%`, `**`), unary operators (`+/-`), and floor division (`//`) in addition to boolean and comparison operators.
        *   **Expanded Node Types (`ALLOWED_NODE_TYPES`):** Supports more AST nodes, including `ast.BinOp` (for arithmetic), `ast.IfExp` (ternary operator: `val_if_true if condition else val_if_false`), literal collections (`List`, `Tuple`, `Dict`, `Set`), `ast.Subscript` (indexing/slicing like `my_list[0]`, `my_dict['key']`), and `ast.Attribute` (safe attribute access like `my_obj.name`).
        *   **Allowed Builtins:** A curated list of safe Python built-in functions (`len`, `str`, `int`, `float`, `bool`, `all`, `any`, `round`, `abs`, `min`, `max`, `sum`, `sorted`, collection constructors, `isinstance`, `type`, `getattr`, `hasattr`) can be used within expressions.
        *   **Method Calls:** Basic, safe method calls on primitive types (strings, lists, dicts) are now supported (e.g., `my_string.startswith('prefix')`, `my_list.count(item)`).
    -   **Context (`context_vars`):** Must be initialized with a dictionary of allowed variable names and their current Python values for use during evaluation.
    -   **Allowed Functions (`allowed_functions`):** Can be provided with a dictionary of Python callables that are safe to invoke from expressions.

-   **`LogicalRule` Class:**
    -   **Purpose:** Represents a formal logical rule (e.g., IF `condition_expr` THEN `effects`) or a consistency constraint.
    -   **Attributes:**
        -   `id`: Unique rule identifier.
        -   `condition_expr`: A string representing the logical condition (antecedent), evaluated by `SafeExpressionEvaluator`. Validated for syntax on rule creation.
        -   `description`: Human-readable explanation.
        -   `effects`: (For derivation rules) A list of dictionaries describing actions if the condition is true (e.g., `{"set_belief": "derived_fact_key", "value": True, "confidence": 0.9}`). Effect values can reference context variables (e.g., `{"value": "$belief_A"}`).
        -   `is_constraint`: Boolean. If `True`, this rule's condition *must* evaluate to `True` for the current belief state to be considered consistent.
        -   `priority`: Integer. Used to order rule application in inference.
    -   **`evaluate_condition(context_vars, allowed_functions)`:** Evaluates the rule's condition string. Returns `True` or `False`, gracefully handling evaluation errors by typically returning `False` and logging a warning.

-   **`LogicEngine` Class:**
    -   **Purpose:** Orchestrates logical operations for an agent. It manages rules, default assumptions, and interacts with the agent's shared `BeliefSystem`.
    -   **Initialization:** Takes a `BeliefSystem` instance, an optional `LLMHandler` (specifically for Socratic questioning), and an `agent_id_namespace` (for namespacing its own metadata or rules if stored in the `BeliefSystem`).
    -   **Rule Management (`add_rule`, `load_rules_from_belief_system`):** Allows adding `LogicalRule` objects. Can also load rule definitions if they are stored as `Belief` objects in the `BeliefSystem`.
    -   **Default Assumptions (`add_default_assumption`):** Defines beliefs assumed true by default (e.g., `"network.status": "stable"`) with a specified confidence. These are used in evaluation contexts unless overridden by more confident beliefs from the `BeliefSystem`.
    -   **Evaluation Context (`_get_evaluation_context`):** Prepares the `context_vars` dictionary for `SafeExpressionEvaluator`. It intelligently merges:
        1.  Relevant beliefs fetched from the shared `BeliefSystem` (scoped by `agent_belief_prefix_for_context`).
        2.  Active default assumptions (those not overridden by higher-confidence beliefs from the shared system).
        3.  Any `additional_context` explicitly passed to the evaluation method.
    -   **Forward Chaining Inference (`forward_chain`):**
        *   Takes an `agent_belief_prefix_for_context` (to fetch initial beliefs) and a `rule_priority_threshold`.
        *   Iteratively applies derivation rules (non-constraint rules with priority >= threshold) whose conditions are met by the current `working_beliefs` (initial + newly derived).
        *   Rules are sorted and applied by priority.
        *   When a rule fires, its `effects` are processed. `set_belief` effects update the `working_beliefs` snapshot. Placeholder values in effects (e.g., `"$some_var_in_context"`) are resolved.
        *   The process continues until no new beliefs are derived or `max_iterations` is hit.
        *   **Returns:** A tuple containing `(final_working_beliefs_snapshot, list_of_derived_effect_dicts)`. The calling agent is responsible for adding these derived effects as new beliefs into the shared `BeliefSystem`.
    -   **Consistency Checking (`check_consistency`):**
        *   Takes an `agent_belief_prefix_for_context`.
        *   Evaluates all rules in its set marked as `is_constraint=True` (or a specific list of constraint rule IDs) against the current context.
        *   A constraint's condition is expected to evaluate to `True`.
        *   Returns a list of `LogicalRule` objects that were **violated** (their condition evaluated to `False`).
    -   **Socratic Question Generation (`generate_socratic_questions`):**
        *   If an `LLMHandler` is configured for the `LogicEngine`, this method takes a `topic_or_goal` and an `agent_belief_prefix_for_context`.
        *   It constructs a prompt asking the LLM to generate insightful Socratic questions to challenge assumptions, explore alternatives, or probe deeper based on the topic and a summary of relevant current beliefs.
        *   Expects a JSON list of question strings from the LLM.

### How it Enhances Agent Intelligence and Safety

-   **Safer Dynamic Logic:** `SafeExpressionEvaluator` is paramount for allowing agents to evaluate conditions that might be dynamically generated (e.g., by an LLM during planning) or come from configurable rule sets, without the full risk of `eval()`.
-   **Explicit Rule-Based Behavior:** Enables agents to operate based on clearly defined logical rules, making their behavior more predictable, verifiable, and explainable for certain tasks.
-   **Knowledge Derivation (Inference):** `forward_chain` allows agents to deduce new information not explicitly present in their immediate beliefs, potentially uncovering implications that a simpler reactive system or a less capable LLM might miss.
-   **Maintaining Consistency:** `check_consistency` provides a mechanism to detect states that violate fundamental constraints of the agent or its domain, prompting corrective action or replanning.
-   **Challenging Biases & Assumptions (Socratic Questioning):** The Socratic questioning feature, driven by an LLM, encourages critical reflection. An agent can use these questions to:
    -   Re-evaluate its current beliefs or plan.
    -   Formulate more nuanced prompts for other LLM tasks.
    -   Identify areas needing more information gathering.
    This can act as a "governor" on purely LLM-driven choices, especially from smaller models, by forcing consideration of alternatives or unstated premises.
-   **Augmenting Smaller LLMs:** When an agent (like `BDIAgent` or `StrategicEvolutionAgent`) has to rely on a less powerful LLM (e.g., due to resource constraints, cost, or as a fallback):
    -   The `LogicEngine` can enforce critical operational rules as hard constraints that the LLM's output must satisfy.
    -   Derived facts from `forward_chain` can enrich the context provided to the smaller LLM, helping it make better inferences with less direct reasoning capability.
    -   Socratic questions can guide the smaller LLM's "thought process" by breaking down a problem or highlighting important aspects to consider.

## Technical Details

-   **AST for Safety:** `SafeExpressionEvaluator` relies on Python's `ast` module.
-   **Contextual Evaluation:** All rule and expression evaluations are performed against a `context_vars` dictionary provided at runtime. For integration with MindX, this context is built by `LogicEngine._get_evaluation_context` by fetching relevant beliefs from the shared `BeliefSystem` (scoped by `agent_belief_prefix_for_context`) and merging them with active default assumptions.
-   **Modularity:** `LogicalRule` and `SafeExpressionEvaluator` are reusable. `LogicEngine` provides the higher-level orchestration.
-   **LLM Dependency (Optional):** The Socratic questioning feature requires an `LLMHandler`. Other features are self-contained.

## Agent Usage Examples (`BDIAgent` or `StrategicEvolutionAgent`)

An agent would typically instantiate `LogicEngine` during its initialization, potentially loading a set of domain-specific rules.

**1. Defining Rules and Assumptions:**

```python
# --- In an Agent's __init__ or setup phase ---
# self.logic_engine = LogicEngine(self.belief_system, self.llm_handler, agent_id=self.agent_id)

# Example rules an agent might define:
await self.logic_engine.add_rule(LogicalRule(
    rule_id="R001_task_eligibility",
    condition_expr="task_priority >= 8 and required_tool_available == True and system_load_metric < 0.75",
    description="High priority tasks are eligible if required tool is available and system load is low.",
    effects=[{"set_belief": "task_is_eligible_for_immediate_execution", "value": True, "confidence": 0.9}]
))
await self.logic_engine.add_rule(LogicalRule(
    rule_id="C001_data_integrity",
    condition_expr="input_data_checksum == expected_checksum",
    description="Input data checksum must match expected checksum.",
    is_constraint=True 
))

#####################3
# AGENT USAGE
# --- In an agent's decision-making loop (e.g., BDI's deliberate or execute step) ---

# a. Prepare the evaluation context from current beliefs
#    The agent_belief_prefix helps scope beliefs relevant to this agent/task.
agent_context_prefix = f"bdi.{self.domain}.beliefs.knowledge" # Example prefix
current_snapshot_for_logic = await self.logic_engine._get_evaluation_context(
    agent_belief_prefix_for_context=agent_context_prefix,
    additional_context={"current_task_id": "task123"} # Any other dynamic vars
)

# b. Perform Inference
# updated_snapshot, derived_effects = await self.logic_engine.forward_chain(current_snapshot_for_logic)
# if derived_effects:
#     logger.info(f"Agent {self.agent_id}: Logic engine derived {len(derived_effects)} new facts.")
#     for effect in derived_effects:
#         # Add these new facts back to the agent's shared BeliefSystem
#         await self.belief_system.add_belief(
#             f"{agent_context_prefix}.{effect['key']}", # Namespace appropriately
#             effect['value'], effect['confidence'], effect['source'], effect.get('metadata')
#         )
#     # Refresh current_snapshot_for_logic if subsequent rules depend on these new facts in the same cycle
#     current_snapshot_for_logic = updated_snapshot


# c. Check Preconditions for an Action (using a specific rule)
#    Assume 'action_to_take' has a 'precondition_rule_id' field.
# precondition_rule = self.logic_engine.rules.get(action_to_take.precondition_rule_id)
# if precondition_rule:
#     if precondition_rule.evaluate_condition(current_snapshot_for_logic, self.logic_engine.allowed_eval_functions):
#         logger.info(f"Agent {self.agent_id}: Preconditions MET for action {action_to_take.id}.")
#         # ... proceed to execute action ...
#     else:
#         logger.warning(f"Agent {self.agent_id}: Preconditions NOT MET for action {action_to_take.id}. Rule: {precondition_rule.id}")
#         # ... handle precondition failure (e.g., replan, wait) ...

# d. Check System Consistency
# violated_constraints = await self.logic_engine.check_consistency(current_snapshot_for_logic)
# if violated_constraints:
#     logger.error(f"Agent {self.agent_id}: SYSTEM INCONSISTENCY DETECTED! Violated rules: {[r.id for r in violated_constraints]}")
#     # ... take corrective action, alert, or halt ...


# e. Socratic Questioning to refine understanding or LLM prompt
# if some_condition_for_deeper_reflection:
#     socratic_qs = await self.logic_engine.generate_socratic_questions(
#         topic_or_goal="Deciding on optimal strategy for resource allocation",
#         agent_belief_prefix_for_context=agent_context_prefix # To fetch relevant beliefs
#     )
#     logger.info(f"Agent {self.agent_id}: Socratic Questions to consider for resource allocation:\n{socratic_qs}")
#     # These questions could be:
#     # - Logged for human review.
#     # - Used to formulate a new, more detailed query to an LLM.
#     # - Used by the agent to update its own goals or search for more information.
await self.logic_engine.add_default_assumption("network_api.status", "responsive", confidence=0.6)
