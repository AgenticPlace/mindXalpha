# mindx/utils/logic_engine.py
"""
Enhanced Logic Engine for mindX Augmentic Intelligence 

Provides utilities for representing logical rules, evaluating conditions safely,
performing basic forward-chaining inference, and conceptual Socratic questioning
to enhance agent decision-making, especially when dealing with smaller LLMs or
needing to validate choices.
"""
import logging
import sys
import ast
import operator
from typing import Dict, Any, Callable, List, Union, Set, Optional, Tuple
import uuid
import copy # For deepcopying context

from mindx.utils.logging_config import get_logger
from mindx.core.belief_system import BeliefSystem, Belief, BeliefSource 
from mindx.llm.llm_factory import LLMHandler, create_llm_handler # For Socratic questioning
from mindx.utils.config import Config # For configuring LLM for Socratic part

logger = get_logger(__name__)

# --- Safe Expression Evaluation ---
ALLOWED_OPERATORS = {
    ast.And: operator.and_, ast.Or: operator.or_, ast.Not: operator.not_,
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y,
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}
ALLOWED_NODE_TYPES_BASE = ( # Base types always allowed
    ast.Expression, ast.Name, ast.Constant, ast.Attribute,
    ast.BoolOp, ast.UnaryOp, ast.Compare, ast.BinOp,
    ast.Call, ast.IfExp, 
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Slice, ast.Subscript,
    ast.Load, 
)
ALLOWED_NODE_TYPES = ALLOWED_NODE_TYPES_BASE
if sys.version_info < (3,8): # pragma: no cover
    ALLOWED_NODE_TYPES += (ast.NameConstant, ast.Num, ast.Str, ast.Bytes, ast.Ellipsis)


class SafeExpressionEvaluator: # pragma: no cover
    """Safely evaluates Python-like expressions by parsing to AST and allowing only whitelisted operations."""
    def __init__(self, context_vars: Optional[Dict[str, Any]] = None, 
                 allowed_functions: Optional[Dict[str, Callable]] = None):
        self.context_vars = context_vars or {}
        self.allowed_functions = allowed_functions or {}
        self.allowed_builtins = {
            "len": len, "str": str, "int": int, "float": float, "bool": bool, 
            "all": all, "any": any, "round": round, "abs": abs, 
            "min": min, "max": max, "sum": sum, "sorted": sorted,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "isinstance": isinstance, "type": type, # type() can be useful for checking
            "getattr": getattr, # Use getattr carefully, ensure objects in context are safe
            "hasattr": hasattr,
        }

    def _eval_node(self, node: ast.AST) -> Any:
        node_type = type(node)

        if node_type not in ALLOWED_NODE_TYPES: # pragma: no cover
            raise TypeError(f"SafeEvaluator: Disallowed AST node type: {node_type.__name__}")

        if isinstance(node, ast.Constant): return node.value
        elif sys.version_info < (3,8) and isinstance(node, (ast.Num, ast.Str)): return node.n if isinstance(node, ast.Num) else node.s # type: ignore
        elif sys.version_info < (3,8) and isinstance(node, ast.NameConstant): return node.value # type: ignore
        
        elif isinstance(node, ast.Name):
            if node.id in self.context_vars: return self.context_vars[node.id]
            if node.id in self.allowed_builtins: return self.allowed_builtins[node.id]
            raise NameError(f"SafeEvaluator: Name '{node.id}' is not defined in allowed context or builtins.")
        
        elif isinstance(node, ast.UnaryOp):
            op_func = ALLOWED_OPERATORS.get(type(node.op))
            if not op_func: raise TypeError(f"SafeEvaluator: Disallowed unary op: {type(node.op).__name__}")
            return op_func(self._eval_node(node.operand))

        elif isinstance(node, ast.BoolOp):
            op_func = ALLOWED_OPERATORS[type(node.op)]
            if isinstance(node.op, ast.And):
                for value_node in node.values:
                    if not self._eval_node(value_node): return False
                return True
            elif isinstance(node.op, ast.Or):
                for value_node in node.values:
                    if self._eval_node(value_node): return True
                return False
        
        elif isinstance(node, ast.BinOp):
            op_func = ALLOWED_OPERATORS.get(type(node.op))
            if not op_func: raise TypeError(f"SafeEvaluator: Disallowed binary op: {type(node.op).__name__}")
            return op_func(self._eval_node(node.left), self._eval_node(node.right))

        elif isinstance(node, ast.Compare):
            left_val = self._eval_node(node.left)
            for i, op_node in enumerate(node.ops):
                op_type = type(op_node); op_func = ALLOWED_OPERATORS.get(op_type)
                if not op_func: raise TypeError(f"SafeEvaluator: Disallowed comparison op: {op_type.__name__}")
                right_val = self._eval_node(node.comparators[i])
                if not op_func(left_val, right_val): return False
                left_val = right_val 
            return True
        
        elif isinstance(node, ast.Call):
            func_to_call: Callable
            if isinstance(node.func, ast.Name): # Direct function call e.g. my_func() or len()
                func_name = node.func.id
                if func_name in self.allowed_functions: func_to_call = self.allowed_functions[func_name]
                elif func_name in self.allowed_builtins: func_to_call = self.allowed_builtins[func_name] # type: ignore
                else: raise NameError(f"SafeEvaluator: Function call to '{func_name}' not allowed.")
            elif isinstance(node.func, ast.Attribute): # Method call e.g. my_str.startswith('p')
                obj_val = self._eval_node(node.func.value)
                method_name = node.func.attr
                # Allow common, safe, side-effect-free methods on basic types
                safe_methods_on_types = {
                    str: {"startswith", "endswith", "lower", "upper", "strip", "isdigit", "isalpha", "isalnum", "isspace", "find", "count", "replace", "split", "join", "format", "capitalize", "title"},
                    list: {"count", "index"}, # Read-only list methods
                    dict: {"get", "keys", "values", "items"},
                    set: {"issubset", "issuperset", "union", "intersection", "difference", "isdisjoint"},
                    # int, float, bool don't have many callable methods that make sense here
                }
                obj_type = type(obj_val)
                if obj_type in safe_methods_on_types and method_name in safe_methods_on_types[obj_type]:
                    func_to_call = getattr(obj_val, method_name)
                elif hasattr(obj_val, method_name) and callable(getattr(obj_val, method_name)) and not method_name.startswith("_"): # pragma: no cover # More general but riskier
                    logger.warning(f"SafeEvaluator: Allowing call to method '{method_name}' on object of type '{obj_type.__name__}'. Ensure this is safe.")
                    func_to_call = getattr(obj_val, method_name)
                else: raise NameError(f"SafeEvaluator: Method call '{method_name}' on type '{obj_type.__name__}' not allowed or doesn't exist.")
            else: raise TypeError(f"SafeEvaluator: Unsupported function call type: {type(node.func).__name__}")

            args_val = [self._eval_node(arg) for arg in node.args]
            kwargs_val = {kw.arg: self._eval_node(kw.value) for kw in node.keywords if kw.arg is not None}
            return func_to_call(*args_val, **kwargs_val)
        
        elif isinstance(node, ast.IfExp): return self._eval_node(node.body) if self._eval_node(node.test) else self._eval_node(node.orelse)
        elif isinstance(node, ast.List): return [self._eval_node(e) for e in node.elts]
        elif isinstance(node, ast.Tuple): return tuple(self._eval_node(e) for e in node.elts)
        elif isinstance(node, ast.Set): return {self._eval_node(e) for e in node.elts}
        elif isinstance(node, ast.Dict): return { (self._eval_node(k) if k is not None else None) : self._eval_node(v) for k,v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Subscript): return self._eval_node(node.value)[self._eval_node(node.slice)]
        elif isinstance(node, ast.Slice): return slice(self._eval_node(node.lower) if node.lower else None, self._eval_node(node.upper) if node.upper else None, self._eval_node(node.step) if node.step else None)
        elif isinstance(node, ast.Attribute): # obj.attr
            obj = self._eval_node(node.value)
            attr_name = node.attr
            # More controlled attribute access: only allow if obj is a dict and attr_name is a key,
            # or if obj is a simple data object and attr_name is not private.
            if isinstance(obj, dict): # Allow dict key access via attribute if it's a valid key
                if attr_name in obj: return obj[attr_name]
                else: raise AttributeError(f"SafeEvaluator: Dictionary object does not have key '{attr_name}'.")
            # For other objects, only allow non-private attributes
            if not attr_name.startswith('_'):
                if hasattr(obj, attr_name): return getattr(obj, attr_name)
            raise AttributeError(f"SafeEvaluator: Accessing attribute '{attr_name}' on object of type '{type(obj).__name__}' is restricted or attribute does not exist.")
        else: raise TypeError(f"SafeEvaluator: Unsupported AST node: {type(node).__name__}") # pragma: no cover

    def evaluate(self, expression_string: str) -> Any:
        if not isinstance(expression_string, str) or not expression_string.strip():
            logger.debug("SafeEvaluator: Empty/non-string expression -> False.")
            return False
        try:
            ast_tree = ast.parse(expression_string, mode='eval')
            if not isinstance(ast_tree, ast.Expression) or not hasattr(ast_tree, 'body'): # pragma: no cover
                 raise ValueError("Expression must be a single valid Python expression.")
            return self._eval_node(ast_tree.body)
        except SyntaxError as e: logger.error(f"SafeEvaluator: Syntax error parsing '{expression_string}': {e}"); raise
        except (TypeError, NameError, AttributeError, IndexError, KeyError) as e_eval: # pragma: no cover
            logger.warning(f"SafeEvaluator: Error evaluating '{expression_string}': {type(e_eval).__name__} - {e_eval}. Context (keys): {list(self.context_vars.keys())}. Allowed funcs: {list(self.allowed_functions.keys())}")
            raise # Re-raise evaluation errors for LogicalRule to handle
        except Exception as e_unexp: # pragma: no cover
            logger.error(f"SafeEvaluator: Unexpected error for '{expression_string}': {e_unexp}", exc_info=True)
            raise RuntimeError(f"Unexpected evaluation error: {e_unexp}") from e_unexp

class LogicalRule: # pragma: no cover
    """Represents a logical rule (IF condition THEN effects) or a consistency constraint."""
    def __init__(self, rule_id: str, condition_expr: str, 
                 description: Optional[str] = None, 
                 effects: Optional[List[Dict[str, Any]]] = None, 
                 is_constraint: bool = False,
                 priority: int = 0): # Higher number = higher priority for firing
        self.id = rule_id; self.condition_expr = condition_expr; self.description = description
        self.effects = effects or []; self.is_constraint = is_constraint; self.priority = priority
        try: ast.parse(self.condition_expr, mode='eval')
        except SyntaxError as e: raise ValueError(f"Invalid syntax in rule '{rule_id}' condition: '{self.condition_expr}'. Error: {e}") from e

    def evaluate_condition(self, context_vars: Dict[str, Any], 
                           allowed_functions: Optional[Dict[str, Callable]] = None) -> bool:
        evaluator = SafeExpressionEvaluator(context_vars=context_vars, allowed_functions=allowed_functions)
        try:
            result = evaluator.evaluate(self.condition_expr)
            if not isinstance(result, bool): logger.warning(f"Rule '{self.id}' non-boolean result: {type(result).__name__} '{result}'. Treating as False."); return False
            return result
        except Exception: return False # Default to False on any evaluation error, already logged by SafeEvaluator
    def __repr__(self): return f"<LogicalRule id='{self.id}' prio={self.priority} condition='{self.condition_expr}' constraint={self.is_constraint}>"
    def to_dict(self) -> Dict[str, Any]: return {"id": self.id, "condition_expr": self.condition_expr, "description": self.description, "effects": self.effects, "is_constraint": self.is_constraint, "priority": self.priority}
    @classmethod
    def from_dict(cls, data: Dict[str,Any]) -> 'LogicalRule': return cls(**data)


class LogicEngine: # pragma: no cover
    """
    Provides foundational logic capabilities for MindX agents.
    Manages logical rules, default assumptions, performs forward chaining inference,
    checks consistency, and can use an LLM for Socratic questioning.
    """
    def __init__(self, belief_system: BeliefSystem, 
                 llm_handler_for_socratic: Optional[LLMHandler] = None, # Specifically for Socratic part
                 agent_id_namespace: str = "mindx_default_logic_engine"):
        self.belief_system = belief_system
        self.llm_handler_socratic = llm_handler_for_socratic
        self.agent_id_ns = agent_id_namespace # For namespacing beliefs *about* rules/defaults if persisted
        self.log_prefix = f"LogicEngine ({self.agent_id_ns}):"
        self.config = Config()
        
        self.rules: Dict[str, LogicalRule] = {} 
        self.default_assumptions: Dict[str, Dict[str, Any]] = {} # key -> {"value": val, "confidence": float, "overridden_by_belief_key": Optional[str]}
        
        self.allowed_eval_functions: Dict[str, Callable] = {
            # These allow rules to *query the provided context_vars* during evaluation,
            # NOT to directly query the global BeliefSystem to maintain purity of evaluation.
            "context_has_key": lambda key, ctx: key in ctx,
            "context_get_value": lambda key, default_val, ctx: ctx.get(key, default_val),
            # Example for an agent to pass its own safe functions:
            # "is_weather_good": lambda weather_belief_val: weather_belief_val in ["sunny", "clear"]
        }
        logger.info(f"{self.log_prefix} Initialized. LLM for Socratic: {'Yes' if self.llm_handler_socratic else 'No'}")

    async def load_rules_from_belief_system(self, rule_prefix: Optional[str] = None):
        """Loads rules stored as Belief objects into the engine."""
        rule_belief_prefix = rule_prefix or f"logic_engine.{self.agent_id_ns}.rules."
        logger.info(f"{self.log_prefix} Loading rules from BeliefSystem with prefix: {rule_belief_prefix}")
        rule_beliefs = await self.belief_system.query_beliefs(partial_key=rule_belief_prefix)
        count = 0
        for belief in rule_beliefs:
            try:
                if isinstance(belief.value, dict):
                    rule = LogicalRule.from_dict(belief.value)
                    self.rules[rule.id] = rule
                    count += 1
                else: logger.warning(f"{self.log_prefix} Belief {belief.key} value is not a dict, cannot load as rule.")
            except Exception as e: logger.error(f"{self.log_prefix} Failed to load rule from belief {belief.key}: {e}")
        logger.info(f"{self.log_prefix} Loaded {count} rules from BeliefSystem.")


    async def add_rule(self, rule: LogicalRule, persist: bool = False):
        if not isinstance(rule, LogicalRule): raise TypeError("Rule must be LogicalRule instance.")
        self.rules[rule.id] = rule
        logger.info(f"{self.log_prefix} Added rule: {rule.id} - '{rule.condition_expr}'")
        if persist: await self.belief_system.add_belief(f"logic_engine.{self.agent_id_ns}.rules.{rule.id}", rule.to_dict(), 1.0, BeliefSource.EXTERNAL_INPUT)

    async def add_default_assumption(self, key: str, value: Any, confidence: float = 0.5, persist: bool = False):
        self.default_assumptions[key] = {"value": value, "confidence": confidence, "overridden_by_belief_key": None}
        logger.info(f"{self.log_prefix} Added default assumption: {key} = {str(value)[:50]} (Conf: {confidence})")
        if persist: await self.belief_system.add_belief(f"logic_engine.{self.agent_id_ns}.defaults.{key}", self.default_assumptions[key], 1.0, BeliefSource.EXTERNAL_INPUT)

    async def _get_evaluation_context(self, agent_belief_prefix: str, additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Prepares context for SafeExpressionEvaluator.
        Merges agent's relevant beliefs from shared BeliefSystem with active defaults and additional context.
        """
        eval_context: Dict[str, Any] = {}
        
        # 1. Fetch relevant beliefs from the shared BeliefSystem for the specific agent/domain
        agent_beliefs = await self.belief_system.query_beliefs(partial_key=agent_belief_prefix)
        for belief in agent_beliefs:
            # Use simplified key (suffix after prefix) in eval_context for easier rule writing
            simple_key = belief.key.replace(agent_belief_prefix, "").lstrip(".")
            # Higher confidence beliefs from BeliefSystem override lower confidence ones if keys clash (rare with good prefixing)
            if simple_key not in eval_context or belief.confidence > (eval_context.get(f"{simple_key}_confidence", -1.0)):
                eval_context[simple_key] = belief.value
                eval_context[f"{simple_key}_confidence"] = belief.confidence # Store confidence too if needed by rules

        # 2. Overlay active default assumptions
        # A belief from BeliefSystem with the same key as a default assumption overrides the default
        # if its confidence is higher than the default's confidence.
        for def_key, def_data in self.default_assumptions.items():
            corresponding_belief_val = eval_context.get(def_key)
            corresponding_belief_conf = eval_context.get(f"{def_key}_confidence", -1.0)

            if corresponding_belief_val is not None and corresponding_belief_conf >= def_data["confidence"]:
                def_data["overridden_by_belief_key"] = f"{agent_belief_prefix.rstrip('.')}.{def_key}" # Mark as overridden
                def_data["is_active"] = False
                # The value from BeliefSystem (already in eval_context) takes precedence.
            else: # Default is not overridden by a stronger belief
                def_data["is_active"] = True
                def_data["overridden_by_belief_key"] = None
                if def_key not in eval_context: # Add default if no belief existed
                    eval_context[def_key] = def_data["value"]
                    eval_context[f"{def_key}_confidence"] = def_data["confidence"]
        
        # 3. Overlay any additional explicit context (highest precedence for these vars)
        if additional_context:
            eval_context.update(additional_context)
        
        logger.debug(f"{self.log_prefix} Prepared evaluation context with {len(eval_context)} vars (keys snippet: {list(eval_context.keys())[:5]})")
        return eval_context
    
    async def forward_chain(self, agent_belief_prefix_for_context: str,
                            max_iterations: int = 10,
                            rule_priority_threshold: int = 0 # Minimum priority for rules to be considered
                           ) -> Tuple[Dict[str, Any], List[Dict[str,Any]]]:
        """
        Performs forward chaining. Reads from agent_belief_prefix, derives new facts.
        Returns a dictionary of ALL facts (initial + derived) in the snapshot, and a list of derived effect dicts.
        The calling agent is responsible for adding these derived facts to the shared BeliefSystem.
        """
        logger.info(f"{self.log_prefix} Starting forward chaining. Context prefix: {agent_belief_prefix_for_context}")
        # Initial context for the first iteration
        current_eval_context = await self._get_evaluation_context(agent_belief_prefix_for_context)
        
        all_derived_effects_this_run: List[Dict[str,Any]] = []
        
        # Sort rules by priority (descending) for ordered application
        sorted_rules = sorted([r for r in self.rules.values() if not r.is_constraint and r.priority >= rule_priority_threshold], 
                              key=lambda r: -r.priority)

        for i in range(max_iterations):
            made_change_in_iteration = False
            logger.debug(f"{self.log_prefix} Inference iteration {i+1}/{max_iterations}")
            
            for rule in sorted_rules:
                # If all effects of this rule are already in current_eval_context with same values, skip.
                # This is a simple check; a more advanced one would compare values if already present.
                effects_already_met = True
                if rule.effects:
                    for effect in rule.effects:
                        key_to_check = effect.get("set_belief") or effect.get("update_belief")
                        # This check can be complex if effect value is dynamic.
                        # For now, if key exists, assume effect might re-evaluate it.
                        # A simpler check: if all effect keys are already in current_eval_context, maybe skip.
                        # For robust chaining, we re-evaluate rules if any part of context changed.
                        pass # Let rules fire if their conditions are met based on current context.

                if rule.evaluate_condition(context_data=current_eval_context, allowed_functions=self.allowed_eval_functions):
                    logger.debug(f"{self.log_prefix} Rule '{rule.id}' (Prio:{rule.priority}) fired. Condition: '{rule.condition_expr}'")
                    for effect_template in rule.effects:
                        effect = copy.deepcopy(effect_template) # Work on a copy
                        
                        # Resolve any placeholders in effect values from current_eval_context
                        if isinstance(effect.get("value"), str) and effect["value"].startswith("$"):
                            placeholder_key = effect["value"][1:] # Remove '$'
                            effect["value"] = current_eval_context.get(placeholder_key, None) # Resolve from context
                            if effect["value"] is None: logger.warning(f"{self.log_prefix} Rule '{rule.id}' effect placeholder '{placeholder_key}' not found in context. Using None.")

                        if "set_belief" in effect or "update_belief" in effect:
                            key = effect["set_belief"] if "set_belief" in effect else effect["update_belief"]
                            value = effect["value"]
                            
                            if key not in current_eval_context or current_eval_context[key] != value:
                                logger.info(f"{self.log_prefix} Rule '{rule.id}' derived: {key} = {str(value)[:50]}")
                                current_eval_context[key] = value # Update working snapshot for next rules in this iteration
                                # Store confidence if effect specifies it, otherwise default
                                current_eval_context[f"{key}_confidence"] = effect.get("confidence", 0.8) 
                                
                                derived_effect_log = {
                                    "key": key, "value": value, 
                                    "confidence": effect.get("confidence", 0.8),
                                    "source": BeliefSource.INFERRED, # Mark as inferred by logic engine
                                    "metadata": {"derived_by_rule_id": rule.id, "engine_agent_id": self.agent_id, "iteration": i+1}
                                }
                                all_derived_effects_this_run.append(derived_effect_log)
                                made_change_in_iteration = True
            
            if not made_change_in_iteration:
                logger.info(f"{self.log_prefix} Forward chaining stabilized in {i+1} iterations.")
                break
        else: # pragma: no cover # Only if loop completes all max_iterations
            logger.warning(f"{self.log_prefix} Forward chaining reached max {max_iterations} iterations.")
            
        return current_eval_context, all_derived_effects_this_run

    async def check_consistency(self, agent_belief_prefix_for_context: str,
                                constraint_rule_ids: Optional[List[str]] = None) -> List[LogicalRule]: # pragma: no cover
        """Checks beliefs (from prefix) against constraint rules. Returns violated rules."""
        logger.info(f"{self.log_prefix} Checking consistency. Context prefix: {agent_belief_prefix_for_context}")
        violated_constraints: List[LogicalRule] = []
        eval_context = await self._get_evaluation_context(agent_belief_prefix_for_context)

        rules_to_check_list: List[LogicalRule] = []
        if constraint_rule_ids:
            for r_id in constraint_rule_ids:
                rule = self.rules.get(r_id)
                if rule and rule.is_constraint: rules_to_check_list.append(rule)
                elif rule: logger.warning(f"{self.log_prefix} Rule '{r_id}' not a constraint, skipped.")
                else: logger.warning(f"{self.log_prefix} Rule '{r_id}' not found for consistency.")
        else: rules_to_check_list = [r for r in self.rules.values() if r.is_constraint]

        if not rules_to_check_list: logger.info(f"{self.log_prefix} No constraint rules to check."); return []
        for rule in rules_to_check_list:
            if not rule.evaluate_condition(eval_context, self.allowed_eval_functions):
                logger.warning(f"{self.log_prefix} VIOLATION: Constraint '{rule.id}' FALSE. Cond: '{rule.condition_expr}'")
                violated_constraints.append(rule)
        if not violated_constraints: logger.info(f"{self.log_prefix} Beliefs consistent with checked constraints.")
        return violated_constraints

    async def generate_socratic_questions(self, topic_or_goal: str, 
                                          agent_belief_prefix_for_context: str, 
                                          num_questions: int = 3) -> List[str]: # pragma: no cover
        """(LLM-driven) Generates Socratic questions based on topic and current beliefs from prefix."""
        if not self.llm_handler_socratic: return ["Error: LLMHandler for Socratic questioning not configured."]
        logger.info(f"{self.log_prefix} Generating Socratic Qs for: '{topic_or_goal}' using prefix '{agent_belief_prefix_for_context}'")
        
        relevant_beliefs = await self.belief_system.query_beliefs(partial_key=agent_belief_prefix_for_context, min_confidence=0.5)
        belief_summary = "\n".join([f"- {b.key.replace(agent_belief_prefix_for_context,'').lstrip('.')}: {str(b.value)[:80]}" for b in relevant_beliefs[:10]]) # Max 10 beliefs for prompt

        prompt = (
            f"You are the Socratic assistant for mindX Augmentic Intelligence. Current Topic/Goal: '{topic_or_goal}'.\n"
            f"Relevant Beliefs (summary):\n{belief_summary if belief_summary else 'No specific beliefs provided for this topic.'}\n\n"
            f"Generate {num_questions} insightful Socratic questions that challenge assumptions, explore alternatives, or probe for deeper understanding of the topic/goal and related beliefs. "
            f"Aim to stimulate critical thinking and identify potential flaws or unstated premises.\n"
            f"Respond ONLY with a JSON list of question strings. Example: [\"What if our primary assumption about X is incorrect?\", \"Is there an alternative approach to Y that we haven't considered?\"]" )
        try:
            max_q_tokens = self.config.get(f"logic_engine.{self.agent_id_ns}.socratic.max_tokens", 120 * num_questions)
            temp_q = self.config.get(f"logic_engine.{self.agent_id_ns}.socratic.temperature", 0.65)
            response_str = await self.llm_handler_socratic.generate_text(prompt, max_tokens=max_q_tokens, temperature=temp_q, json_mode=True)
            if not response_str or response_str.startswith("Error:"): raise ValueError(f"LLM Socratic Q gen error: {response_str}")
            
            questions: List[str] = [] # ... (Robust JSON parsing for list of strings)
            try: questions = json.loads(response_str)
            except json.JSONDecodeError: match = re.search(r"\[\s*\"[\s\S]*?\"\s*\]", response_str, re.DOTALL);
            if match: questions = json.loads(match.group(0))
            else: raise ValueError("LLM Socratic Qs not valid JSON list.")
            if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions): raise ValueError("LLM Socratic Qs not list of strings.")
            
            logger.info(f"{self.log_prefix} Generated {len(questions)} Socratic questions."); return questions[:num_questions]
        except Exception as e: logger.error(f"{self.log_prefix} Failed to generate Socratic questions: {e}", exc_info=True); return [f"Error generating Socratic Qs: {e}"]
