# mindx/llm/mock_llm_handler.py
"""
Mock LLM Handler for MindX testing and fallback.
"""
import logging
import asyncio
import json
import re
from typing import Optional, Any, Dict

from mindx.utils.logging_config import get_logger
from .llm_interface import LLMHandlerInterface # Relative import

logger = get_logger(__name__)

class MockLLMHandler(LLMHandlerInterface): # pragma: no cover
    """Mock LLM Handler for testing and fallback when real providers are unavailable."""
    def __init__(self, model_name_for_api: Optional[str] = "mock_generic_model", 
                 api_key: Optional[str] = None, base_url: Optional[str] = None): # api_key, base_url ignored by mock
        super().__init__("mock", model_name_for_api, None, None) # Provider is always "mock"
        logger.info(f"MockLLMHandler initialized for model '{self.model_name_for_api}'. This handler returns predefined or simple dynamic responses.")

    async def generate_text(self, prompt: str, model: str, # 'model' arg is passed but mock uses its own configured model_name_for_api
                            max_tokens: Optional[int] = 2048, temperature: Optional[float] = 0.7,
                            json_mode: Optional[bool] = False, **kwargs: Any) -> Optional[str]:
        
        effective_model_name = self.model_name_for_api or model # Use instance's model if available
        logger.debug(f"MockLLM Call: EffectiveModel='{effective_model_name}', JSON={json_mode}, Temp={temperature}, MaxTok={max_tokens}, Prompt(start): {prompt[:150]}...")
        
        # Simulate a short, variable delay
        simulated_latency = 0.01 + (len(prompt) % 100) * 0.0005 # Small delay based on prompt length
        await asyncio.sleep(simulated_latency)

        if json_mode:
            if "system_analysis" in prompt.lower() or "improvement_suggestions" in prompt.lower():
                return json.dumps({
                    "improvement_suggestions": [
                        {"target_component_path": f"mindx.mock.{effective_model_name}.file_alpha", "suggestion": f"Mock suggestion from {effective_model_name} to refactor for better modularity.", "priority": 7, "is_critical_target": False},
                        {"target_component_path": f"mindx.mock.{effective_model_name}.file_beta", "suggestion": f"Mock suggestion from {effective_model_name} to add comprehensive unit tests.", "priority": 6, "is_critical_target": False}
                    ]
                })
            elif "critique" in prompt.lower() or "scale of 0.0" in prompt.lower():
                return json.dumps({"score": 0.82, "justification": f"Mock critique from {effective_model_name}: The proposed change generally aligns with the goal and seems plausible based on the snippets provided."})
            elif "plan" in prompt.lower() and "action" in prompt.lower() and "JSON list" in prompt: # BDI Planning
                return json.dumps([
                    {"id": "mock_act_1", "type": "ANALYZE_DATA", "params": {"source_belief_key": "initial_data_for_goal"}, "description": "Analyze initial state."},
                    {"id": "mock_act_2", "type": "MAKE_DECISION", "params": {"source_belief_key": "action_results.ANALYZE_DATA.mock_topic", "options_list_belief": "available_strategies"}, "dependency_ids": ["mock_act_1"], "description": "Decide next strategy."},
                    {"id": "mock_act_3", "type": "NO_OP", "params": {"message": "Mock plan completed."}, "dependency_ids": ["mock_act_2"]}
                ])
            else: # Generic JSON mock
                return json.dumps({"mock_provider": self.provider_name, "model_used_for_mock": effective_model_name, "message": "This is a JSON mock response.", "prompt_received_snippet": prompt[:70]})

        # Non-JSON mode responses
        if "ONLY the complete, new content" in prompt or "generate the improved code" in prompt.lower() or "Provide ONLY the complete, new Python code" in prompt :
            # Try to find original code in prompt for a slightly more realistic mock modification
            original_code_match = re.search(r"Current code of.*?:\n```python\n(.*?)\n```", prompt, re.DOTALL)
            if original_code_match:
                original_code = original_code_match.group(1)
                return f"# MOCK: LLM modification by {effective_model_name} based on original code.\n{original_code}\n\n# Added by MockLLMHandler: A new comment or minor change for testing purposes.\npass\n"
            return f"# MOCK: Generated Python code by {effective_model_name}\ndef new_mock_function_for_mindx():\n    print('Hello from MindX mock LLM {effective_model_name}!')\n    # This is a mock implementation.\n    return True"
        elif "Proposed Improvement Description:" in prompt or "analyze the provided python code" in prompt.lower():
            return f"Mock improvement suggestion from {effective_model_name}: Suggest refactoring the primary data processing loop for enhanced clarity and adding more specific exception handling around all external API calls to improve robustness."
        
        # Generic text response
        return f"MOCK general text response from {self.provider_name}/{effective_model_name} regarding your query about: '{prompt[:70]}...'. This mock simulates a typical LLM text output."

    async def shutdown(self): # pragma: no cover
        """Perform any cleanup for the MockLLMHandler."""
        logger.info(f"MockLLMHandler for model '{self.model_name_for_api}' shutting down. (No specific actions needed).")
