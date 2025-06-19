import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call

from mindx.evolution.blueprint_agent import BlueprintAgent, get_blueprint_agent_async
from mindx.utils.config import Config
from mindx.core.belief_system import BeliefSystem, BeliefSource
from mindx.orchestration.coordinator_agent import CoordinatorAgent # Assuming this is the correct path
from mindx.llm.llm_factory import LLMHandler # Assuming this is the correct path

# Test data
MINIMAL_LLM_RESPONSE_STR = json.dumps({
    "blueprint_title": "Test Blueprint",
    "target_mindx_version_increment": "+0.0.1",
    "focus_areas": [{
        "area_title": "Test Area",
        "development_goals": [{
            "target_component_module_path": "mindx.test.module",
            "description": "Test goal",
            "justification": "Because testing."
        }]
    }],
    "key_performance_indicators": ["KPI1"],
    "potential_risks_challenges": ["Risk1"]
})

MALFORMED_JSON_STR = '{"blueprint_title": "Test Blueprint", "focus_areas": [}'
MALFORMED_JSON_WITH_GOOD_EMBEDDED_STR = f'Some text around it {MINIMAL_LLM_RESPONSE_STR} and some more text.'
MALFORMED_JSON_WITH_BAD_EMBEDDED_STR = f'Some text around it {MALFORMED_JSON_STR} and some more text.'


@pytest.fixture
def mock_config():
    """Fixture for a mock Config object."""
    config = Config(test_mode=True)
    # Set some default values that the agent might use during init or calls
    config.set("llm.default_provider", "mock_provider")
    config.set("llm.mock_provider.default_model_for_reasoning", "mock_model")
    config.set("evolution.blueprint_agent.agent_id", "test_bp_agent")
    config.set("evolution.blueprint_agent.llm.provider", "mock_provider")
    config.set("evolution.blueprint_agent.llm.model", "mock_model")
    config.set("evolution.blueprint_agent.llm.max_tokens", 2000)
    config.set("evolution.blueprint_agent.llm.temperature", 0.5)
    config.set("system.agents.coordinator.version", "v_test_coord")
    config.set("system.agents.sia.version", "v_test_sia")
    config.set("system.agents.sea.version", "v_test_sea")
    return config

@pytest.fixture
def mock_belief_system():
    """Fixture for a mock BeliefSystem."""
    bs = MagicMock(spec=BeliefSystem)
    bs.add_belief = AsyncMock()
    return bs

@pytest.fixture
def mock_llm_handler():
    """Fixture for a mock LLMHandler."""
    llm = MagicMock(spec=LLMHandler)
    llm.generate_text = AsyncMock(return_value=MINIMAL_LLM_RESPONSE_STR)
    llm.provider_name = "mock_provider"
    llm.model_name = "mock_model"
    return llm

@pytest.fixture
def mock_coordinator_agent(mock_config):
    """Fixture for a mock CoordinatorAgent."""
    coord = MagicMock(spec=CoordinatorAgent)
    coord.system_capabilities_cache = {}
    coord.improvement_backlog = []
    coord.improvement_campaign_history = []

    # Mock resource monitor
    mock_res_monitor = MagicMock()
    mock_res_monitor.get_resource_usage = MagicMock(return_value={"cpu_percent": 10.0, "memory_percent": 20.0})
    coord.resource_monitor = mock_res_monitor

    # Mock performance monitor
    mock_perf_monitor = MagicMock()
    mock_perf_monitor.get_all_metrics = MagicMock(return_value={
        "llm_key1": {"requests": 100, "success_rate": 0.95},
        "llm_key2": {"requests": 5, "success_rate": 0.6} # low success rate example
    })
    coord.performance_monitor = mock_perf_monitor

    coord._scan_codebase_capabilities = AsyncMock(return_value={
        "cap1": {"module": "module.A"}, "cap2": {"module": "module.B"}
    })
    coord.config = mock_config # Agent might access coordinator's config for some things
    return coord

@pytest.fixture
async def blueprint_agent_instance(mock_belief_system, mock_coordinator_agent, mock_config):
    """Fixture to get a BlueprintAgent instance, ensuring reset for isolation."""
    await BlueprintAgent.reset_instance_async() # Ensure clean state for each test

    # Patch create_llm_handler within the get_blueprint_agent_async factory or BlueprintAgent constructor
    with patch('mindx.evolution.blueprint_agent.create_llm_handler', return_value=mock_llm_handler()) as patched_create_llm:
        agent = await get_blueprint_agent_async(
            belief_system=mock_belief_system,
            coordinator_ref=mock_coordinator_agent,
            config_override=mock_config,
            test_mode=True
        )
        # agent.llm_handler = mock_llm_handler # Replace with mocked one if create_llm_handler isn't patched early enough
    return agent


@pytest.mark.asyncio
async def test_get_blueprint_agent_singleton(mock_belief_system, mock_coordinator_agent, mock_config):
    """Test that get_blueprint_agent_async returns the same instance."""
    with patch('mindx.evolution.blueprint_agent.create_llm_handler', return_value=mock_llm_handler()):
        agent1 = await get_blueprint_agent_async(mock_belief_system, mock_coordinator_agent, mock_config, test_mode=True)
        agent2 = await get_blueprint_agent_async(mock_belief_system, mock_coordinator_agent, mock_config, test_mode=True)
        # In test_mode=True, it should actually create new instances if called again with test_mode=True,
        # or if the factory itself decides based on test_mode.
        # The critical part for singleton is if test_mode=False (default)
    await BlueprintAgent.reset_instance_async()
    with patch('mindx.evolution.blueprint_agent.create_llm_handler', return_value=mock_llm_handler()):
        agent_a = await get_blueprint_agent_async(mock_belief_system, mock_coordinator_agent, mock_config, test_mode=False)
        agent_b = await get_blueprint_agent_async(mock_belief_system, mock_coordinator_agent, mock_config, test_mode=False)
        assert agent_a is agent_b

@pytest.mark.asyncio
async def test_blueprint_agent_reset_instance(mock_belief_system, mock_coordinator_agent, mock_config):
    """Test that reset_instance_async allows creating a new instance."""
    with patch('mindx.evolution.blueprint_agent.create_llm_handler', return_value=mock_llm_handler()) as mock_create_llm:
        agent1 = await get_blueprint_agent_async(mock_belief_system, mock_coordinator_agent, mock_config, test_mode=False)
        await BlueprintAgent.reset_instance_async()
        assert BlueprintAgent._instance is None
        agent2 = await get_blueprint_agent_async(mock_belief_system, mock_coordinator_agent, mock_config, test_mode=False)
        assert agent1 is not agent2
        # Ensure shutdown was called on the old instance if it existed and had shutdown
        # This is implicitly tested by reset_instance_async not erroring out

@pytest.mark.asyncio
async def test_init_reinitialization_prevention(mock_belief_system, mock_coordinator_agent, mock_config):
    """Test that __init__ prevents re-initialization if instance exists and not in test_mode."""
    with patch('mindx.evolution.blueprint_agent.create_llm_handler', return_value=mock_llm_handler()) as mock_create_llm:
        agent1 = BlueprintAgent(belief_system=mock_belief_system, coordinator_ref=mock_coordinator_agent, config_override=mock_config, test_mode=False)
        agent1_llm_handler_id = id(agent1.llm_handler)

        # Attempt to re-initialize by calling __init__ again (which is what get_blueprint_agent_async would do if not careful)
        # This scenario is a bit artificial as direct __init__ is not typical after factory.
        # The factory get_blueprint_agent_async handles this mostly.
        # Let's test the internal _initialized flag's effect.
        BlueprintAgent(belief_system=mock_belief_system, coordinator_ref=mock_coordinator_agent, config_override=mock_config, test_mode=False)
        assert id(agent1.llm_handler) == agent1_llm_handler_id # LLM handler should not have been recreated

        # If we reset and create new, it should be different
        await BlueprintAgent.reset_instance_async()
        agent2 = BlueprintAgent(belief_system=mock_belief_system, coordinator_ref=mock_coordinator_agent, config_override=mock_config, test_mode=False)
        assert id(agent2.llm_handler) != agent1_llm_handler_id

    # Reset for other tests
    await BlueprintAgent.reset_instance_async()


@pytest.mark.asyncio
async def test_gather_system_state_summary_basic(blueprint_agent_instance, mock_coordinator_agent):
    """Test _gather_mindx_system_state_summary with basic populated data."""
    mock_coordinator_agent.system_capabilities_cache = {"cap1": {"module": "module.A"}}
    mock_coordinator_agent.improvement_backlog = [{"status": "PENDING", "priority": 1}]
    mock_coordinator_agent.improvement_campaign_history = [{"target_component_id": "comp1"}]

    summary = await blueprint_agent_instance._gather_mindx_system_state_summary()

    assert summary["num_scanned_capabilities"] == 1
    assert summary["backlog_total_items"] == 1
    assert summary["backlog_pending_items"] == 1
    assert summary["recent_campaigns_count"] == 1
    assert "resource_monitor_snapshot" in summary
    assert "performance_monitor_snapshot" in summary
    assert summary["performance_monitor_snapshot"]["low_success_rate_llms"] == ["llm_key2"]
    assert len(summary["conceptual_known_limitations"]) > 0
    assert "core_agent_versions" in summary
    mock_coordinator_agent._scan_codebase_capabilities.assert_not_called()

@pytest.mark.asyncio
async def test_gather_system_state_summary_empty_caches_and_lists(blueprint_agent_instance, mock_coordinator_agent):
    """Test _gather_mindx_system_state_summary with empty caches and lists."""
    mock_coordinator_agent.system_capabilities_cache = {} # Empty cache
    mock_coordinator_agent.improvement_backlog = []       # Empty backlog
    mock_coordinator_agent.improvement_campaign_history = [] # Empty history

    summary = await blueprint_agent_instance._gather_mindx_system_state_summary()

    assert summary["num_scanned_capabilities"] == 2 # Falls back to _scan_codebase_capabilities
    mock_coordinator_agent._scan_codebase_capabilities.assert_called_once()
    assert summary["backlog_total_items"] == 0
    assert summary["recent_campaigns_count"] == 0
    assert "backlog_top_pending_priorities" not in summary # or is empty list
    assert "last_campaign_outcomes" not in summary # or is empty list

@pytest.mark.asyncio
async def test_gather_system_state_summary_no_monitors(blueprint_agent_instance, mock_coordinator_agent):
    """Test _gather_mindx_system_state_summary when monitors are None."""
    mock_coordinator_agent.resource_monitor = None
    mock_coordinator_agent.performance_monitor = None

    summary = await blueprint_agent_instance._gather_mindx_system_state_summary()

    assert "resource_monitor_snapshot" not in summary
    assert "performance_monitor_snapshot" not in summary


# --- Tests for generate_next_evolution_blueprint ---

@pytest.mark.asyncio
async def test_generate_blueprint_success(blueprint_agent_instance, mock_llm_handler, mock_belief_system):
    """Test successful blueprint generation."""
    current_version = "1.0.0"
    directive = "Focus on stability"

    # Ensure llm_handler on the instance is the mocked one
    blueprint_agent_instance.llm_handler = mock_llm_handler

    blueprint = await blueprint_agent_instance.generate_next_evolution_blueprint(
        current_mindx_version=current_version,
        high_level_directive=directive
    )

    mock_llm_handler.generate_text.assert_called_once()
    call_args = mock_llm_handler.generate_text.call_args
    prompt = call_args[0][0]
    assert current_version in prompt
    assert directive in prompt
    assert "Current MindX System State Summary" in prompt

    expected_blueprint = json.loads(MINIMAL_LLM_RESPONSE_STR)
    assert blueprint == expected_blueprint

    mock_belief_system.add_belief.assert_called_once_with(
        f"mindx.evolution.blueprint.latest",
        expected_blueprint,
        0.95,
        BeliefSource.SELF_ANALYSIS,
        metadata=mock_belief_system.add_belief.call_args[0][4] # time.time() makes it tricky otherwise
    )
    metadata = mock_belief_system.add_belief.call_args[0][4]
    assert metadata["mindx_version_input"] == current_version
    assert metadata["directive"] == directive

@pytest.mark.asyncio
async def test_generate_blueprint_llm_returns_error_string(blueprint_agent_instance, mock_llm_handler):
    """Test blueprint generation when LLM returns a string starting with 'Error:'."""
    blueprint_agent_instance.llm_handler = mock_llm_handler
    mock_llm_handler.generate_text.return_value = "Error: LLM unavailable"

    result = await blueprint_agent_instance.generate_next_evolution_blueprint("1.0.0")

    assert "error" in result
    assert "LLM generation failed or returned error: Error: LLM unavailable" in result["error"]
    assert result["blueprint_title"] == "Error Blueprint - Validation/Input Error"

@pytest.mark.asyncio
async def test_generate_blueprint_llm_returns_empty_string(blueprint_agent_instance, mock_llm_handler):
    """Test blueprint generation when LLM returns an empty string."""
    blueprint_agent_instance.llm_handler = mock_llm_handler
    mock_llm_handler.generate_text.return_value = ""

    result = await blueprint_agent_instance.generate_next_evolution_blueprint("1.0.0")

    assert "error" in result
    assert "LLM generation failed or returned error:" in result["error"] # Empty string after colon
    assert result["blueprint_title"] == "Error Blueprint - Validation/Input Error"


@pytest.mark.asyncio
async def test_generate_blueprint_json_decode_error_and_no_fallback(blueprint_agent_instance, mock_llm_handler):
    """Test JSONDecodeError when LLM response is malformed and regex fallback also fails."""
    blueprint_agent_instance.llm_handler = mock_llm_handler
    mock_llm_handler.generate_text.return_value = MALFORMED_JSON_STR

    result = await blueprint_agent_instance.generate_next_evolution_blueprint("1.0.0")

    assert "error" in result
    assert "Blueprint LLM response was not valid JSON and no JSON object found within." in result["error"]
    assert "position 46" in result["error"] # Check position from error
    assert result["blueprint_title"] == "Error Blueprint - Validation/Input Error" # Outer catch is ValueError

@pytest.mark.asyncio
async def test_generate_blueprint_json_decode_error_with_successful_fallback(blueprint_agent_instance, mock_llm_handler, mock_belief_system):
    """Test JSONDecodeError where initial parse fails but regex fallback succeeds."""
    blueprint_agent_instance.llm_handler = mock_llm_handler
    mock_llm_handler.generate_text.return_value = MALFORMED_JSON_WITH_GOOD_EMBEDDED_STR

    blueprint = await blueprint_agent_instance.generate_next_evolution_blueprint("1.0.0")

    expected_blueprint = json.loads(MINIMAL_LLM_RESPONSE_STR)
    assert blueprint == expected_blueprint
    mock_belief_system.add_belief.assert_called_once() # Should succeed

@pytest.mark.asyncio
async def test_generate_blueprint_json_decode_error_with_malformed_fallback(blueprint_agent_instance, mock_llm_handler):
    """Test JSONDecodeError where regex fallback finds JSON, but it's also malformed."""
    blueprint_agent_instance.llm_handler = mock_llm_handler
    mock_llm_handler.generate_text.return_value = MALFORMED_JSON_WITH_BAD_EMBEDDED_STR

    result = await blueprint_agent_instance.generate_next_evolution_blueprint("1.0.0")

    assert "error" in result
    assert "Blueprint LLM response contained an extractable JSON-like structure, but it was still invalid." in result["error"]
    assert "position 46" in result["error"] # Check position from error within the extracted part
    assert result["blueprint_title"] == "Error Blueprint - Validation/Input Error" # Outer catch

@pytest.mark.asyncio
async def test_generate_blueprint_validation_failure_missing_keys(blueprint_agent_instance, mock_llm_handler):
    """Test blueprint validation failure due to missing essential keys."""
    blueprint_agent_instance.llm_handler = mock_llm_handler
    invalid_blueprint_json = {"blueprint_title": "Only title"} # Missing 'focus_areas'
    mock_llm_handler.generate_text.return_value = json.dumps(invalid_blueprint_json)

    result = await blueprint_agent_instance.generate_next_evolution_blueprint("1.0.0")

    assert "error" in result
    assert "Generated blueprint missing essential keys" in result["error"]
    assert "['blueprint_title']" in result["error"]
    assert result["blueprint_title"] == "Error Blueprint - Validation/Input Error"

@pytest.mark.asyncio
async def test_generate_blueprint_llm_generic_exception(blueprint_agent_instance, mock_llm_handler):
    """Test handling of a generic exception during LLM call."""
    blueprint_agent_instance.llm_handler = mock_llm_handler
    mock_llm_handler.generate_text.side_effect = Exception("Network Error")

    result = await blueprint_agent_instance.generate_next_evolution_blueprint("1.0.0")

    assert "error" in result
    assert "Unexpected blueprint generation exception: Exception: Network Error" in result["error"]
    assert result["blueprint_title"] == "Error Blueprint - Unexpected"

@pytest.mark.asyncio
async def test_shutdown(blueprint_agent_instance):
    """Test that shutdown method runs without error."""
    try:
        await blueprint_agent_instance.shutdown()
    except Exception as e:
        pytest.fail(f"BlueprintAgent.shutdown() raised an exception: {e}")

# TODO: Add tests for the fallback JSONDecodeError handler at the end of generate_next_evolution_blueprint
# This one: except json.JSONDecodeError as jde:
# It's hard to trigger directly if the inner try-except for parsing is robust.
# Could be triggered if json.loads was called on something else that failed, but currently not the case.

# After running tests, identify and prepare a diff for removing pragmas.
# For now, this concludes the test writing phase.
