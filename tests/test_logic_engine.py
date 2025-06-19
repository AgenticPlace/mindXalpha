import pytest
from utils.logic_engine import LogicalRule # Adjusted import path

# Test basic creation and attribute storage
def test_logical_rule_creation_and_attributes():
    rule_id = "test_rule_001"
    condition = "context_var_A > 10 and context_var_B == 'active'"
    description = "A simple test rule."
    effects = [{"set_belief": "derived_fact_X", "value": True, "confidence": 0.9}]
    is_constraint = False
    priority = 5

    rule = LogicalRule(
        rule_id=rule_id,
        condition_expr=condition,
        description=description,
        effects=effects,
        is_constraint=is_constraint,
        priority=priority
    )

    assert rule.id == rule_id
    assert rule.condition_expr == condition
    assert rule.description == description
    assert rule.effects == effects
    assert rule.is_constraint == is_constraint
    assert rule.priority == priority

# Test LogicalRule.to_dict() method
def test_logical_rule_to_dict():
    rule_data = {
        "id": "test_rule_002",
        "condition_expr": "data_point < 100",
        "description": "Rule for data point threshold.",
        "effects": [{"action": "trigger_alert", "level": "high"}],
        "is_constraint": True,
        "priority": 10
    }
    rule = LogicalRule(
        rule_id=rule_data["id"],
        condition_expr=rule_data["condition_expr"],
        description=rule_data["description"],
        effects=rule_data["effects"],
        is_constraint=rule_data["is_constraint"],
        priority=rule_data["priority"]
    )

    assert rule.to_dict() == rule_data

# Test LogicalRule.from_dict() class method
def test_logical_rule_from_dict():
    rule_data = {
        "id": "test_rule_003",
        "condition_expr": "status == 'PENDING' or attempts < 3",
        "description": "Rule for retrying tasks.",
        "effects": [{"action": "retry_task"}],
        "is_constraint": False,
        "priority": 2
    }
    rule = LogicalRule.from_dict(rule_data)

    assert rule.id == rule_data["id"]
    assert rule.condition_expr == rule_data["condition_expr"]
    assert rule.description == rule_data["description"]
    assert rule.effects == rule_data["effects"]
    assert rule.is_constraint == rule_data["is_constraint"]
    assert rule.priority == rule_data["priority"]

# Test that ValueError is raised for invalid condition_expr syntax
def test_logical_rule_invalid_condition_syntax():
    with pytest.raises(ValueError) as excinfo:
        LogicalRule(
            rule_id="invalid_syntax_rule",
            condition_expr="this is not valid python syntax ---",
            description="A rule with invalid condition syntax."
        )
    assert "Invalid syntax in rule" in str(excinfo.value)
    assert "condition" in str(excinfo.value)

# Test that a valid but more complex condition does not raise an error on init
def test_logical_rule_valid_complex_condition():
    try:
        LogicalRule(
            rule_id="valid_complex_rule",
            condition_expr="((var1 > 10 and var2 != 'initial') or var3 in [1,2,3]) and helper_func(var4)",
            description="A rule with a valid, more complex condition."
        )
    except ValueError:
        pytest.fail("ValueError raised for a valid complex condition string on LogicalRule init.")
