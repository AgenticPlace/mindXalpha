# mindx/llm/llm_factory.py
from typing import Optional, Any, Dict, List, Tuple
import json
import asyncio
import importlib 
import os
from pathlib import Path

from mindx.utils.config import Config, PROJECT_ROOT # Use PROJECT_ROOT for default config path
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default Cloud Run friendly models if not specified in llm_factory_config.json
# This can be overridden or extended by the JSON config.
DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY = {
    "excellent_fit": [
        {"id": "phi3_py", "ollama_name": "phi3:mini", "type": "general/reasoning", "est_memory_needed_gb": 4},
        {"id": "gemma_2b_py", "ollama_name": "gemma:2b", "type": "general", "est_memory_needed_gb": 3},
        {"id": "codegemma_2b_py", "ollama_name": "codegemma:2b", "type": "coding", "est_memory_needed_gb": 3},
    ],
    "good_fit": [
        {"id": "llama3_8b_py", "ollama_name": "llama3:8b", "type": "general/chat", "est_memory_needed_gb": 8},
        {"id": "mistral_7b_py", "ollama_name": "mistral:7b", "type": "general/chat", "est_memory_needed_gb": 8},
        {"id": "codegemma_7b_py", "ollama_name": "codegemma:7b-it", "type": "coding", "est_memory_needed_gb": 8},
    ],
    "default_coding_preference_order": ["codegemma_7b_py", "codegemma_2b_py", "phi3_py"],
    "default_general_preference_order": ["llama3_8b_py", "mistral_7b_py", "phi3_py", "gemma_2b_py"]
}


class LLMHandlerInterface: # pragma: no cover
    """Abstract base class (conceptual) for LLM Handlers."""
    def __init__(self, provider_name: str, model_name_for_api: Optional[str], api_key: Optional[str], base_url: Optional[str]):
        self.provider_name = provider_name
        self.model_name_for_api = model_name_for_api
        self.api_key = api_key
        self.base_url = base_url
        logger.info(f"LLMHandlerInterface: Initialized for {provider_name}/{model_name_for_api or 'default_from_provider_config'}")

    async def generate_text(self, prompt: str, model: str, 
                            max_tokens: Optional[int] = 2048, temperature: Optional[float] = 0.7,
                            json_mode: Optional[bool] = False, **kwargs: Any) -> Optional[str]:
        raise NotImplementedError("Subclasses must implement generate_text")

class OllamaHandler(LLMHandlerInterface): # pragma: no cover
    # (Same as previous full OllamaHandler - no changes needed here for config loading)
    def __init__(self, model_name_for_api: Optional[str], base_url: Optional[str]):
        super().__init__("ollama", model_name_for_api, None, base_url)
        self.ollama_sdk = None
        try: self.ollama_sdk = importlib.import_module("ollama"); logger.info(f"OllamaHandler: SDK imported for model {self.model_name_for_api or 'provider_default'}.")
        except ImportError: logger.error("OllamaHandler: Ollama SDK not found. `pip install ollama` required.")
    async def generate_text(self, prompt: str, model: str, max_tokens: Optional[int]=2048, temperature: Optional[float]=0.7, json_mode: Optional[bool]=False, **kwargs: Any) -> Optional[str]:
        if not self.ollama_sdk: return f"Error: Ollama SDK not available for model '{model}'."
        logger.debug(f"Ollama Call: Model='{model}', JSON={json_mode}, Temp={temperature}, MaxTok={max_tokens}, Prompt(start): {prompt[:150]}...")
        try:
            client_args = {};
            if self.base_url: client_args['host'] = self.base_url
            async_client = self.ollama_sdk.AsyncClient(**client_args)
            options: Dict[str, Any] = {"temperature": temperature if temperature is not None else 0.7}
            if max_tokens is not None and max_tokens > 0 : options["num_predict"] = max_tokens
            generate_args: Dict[str, Any] = {"model": model, "prompt": prompt, "options": options}
            if json_mode: generate_args["format"] = "json"
            if "stop_sequences" in kwargs: options["stop"] = kwargs["stop_sequences"]
            response = await async_client.generate(**generate_args)
            final_response = response.get('response', '').strip()
            if json_mode and final_response and not (final_response.startswith('{') and final_response.endswith('}')) and not (final_response.startswith('[') and final_response.endswith(']')): logger.warning(f"OllamaHandler: Model '{model}' requested JSON but output non-JSON. Snippet: {final_response[:100]}")
            logger.debug(f"Ollama response for '{model}' (first 100): {final_response[:100]}"); return final_response
        except Exception as e: logger.error(f"OllamaHandler: API call failed for model '{model}': {e}", exc_info=True); return f"Error: Ollama call failed - {type(e).__name__}: {e}"


class GeminiHandler(LLMHandlerInterface): # pragma: no cover
    # (Same as previous full GeminiHandler - no changes needed here for config loading)
    def __init__(self, model_name_for_api: Optional[str], api_key: Optional[str]):
        super().__init__("gemini", model_name_for_api, api_key, None)
        self.genai_sdk = None
        if not self.api_key: logger.error(f"GeminiHandler: API key not provided for model '{self.model_name_for_api}'. Calls will fail."); return
        try: self.genai_sdk = importlib.import_module("google.generativeai"); self.genai_sdk.configure(api_key=self.api_key); logger.info(f"GeminiHandler: SDK configured for model '{self.model_name_for_api}'.")
        except ImportError: logger.error("GeminiHandler: SDK not found. `pip install google-generativeai` required.")
        except Exception as e: logger.error(f"GeminiHandler: Error configuring SDK for '{self.model_name_for_api}': {e}"); self.genai_sdk = None
    async def generate_text(self, prompt: str, model: str, max_tokens: Optional[int]=2048, temperature: Optional[float]=0.7, json_mode: Optional[bool]=False, **kwargs: Any) -> Optional[str]:
        if not self.genai_sdk: return f"Error: Gemini SDK unavailable/unconfigured for model '{model}'."
        logger.debug(f"Gemini Call: Model='{model}', JSON={json_mode}, Temp={temperature}, MaxTok={max_tokens}, Prompt(start): {prompt[:150]}...")
        try:
            gen_config_args: Dict[str, Any] = {"temperature": temperature if temperature is not None else 0.7}
            if max_tokens is not None and max_tokens > 0: gen_config_args["max_output_tokens"] = max_tokens
            effective_prompt = prompt
            if json_mode: # (Gemini JSON mode handling from previous)
                is_pro = "pro" in model.lower(); use_mime = False
                if is_pro and hasattr(self.genai_sdk.types.GenerationConfig(), "response_mime_type"):
                    try: gen_config_args["response_mime_type"] = "application/json"; use_mime = True; logger.info(f"Gemini: Using JSON mime_type for '{model}'.")
                    except: logger.warning(f"Gemini: Failed JSON mime_type for '{model}', using prompt instruction."); use_mime = False
                if not use_mime: effective_prompt = prompt + "\n\nRespond ONLY with a single, valid JSON object."; logger.info("Gemini: Appended JSON instruction to prompt.")
            safety_settings = kwargs.get("safety_settings", [{"category":c,"threshold":"BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT","HARM_CATEGORY_HATE_SPEECH","HARM_CATEGORY_SEXUALLY_EXPLICIT","HARM_CATEGORY_DANGEROUS_CONTENT"]])
            model_instance = self.genai_sdk.GenerativeModel(model, safety_settings=safety_settings)
            response = await model_instance.generate_content_async(effective_prompt, generation_config=self.genai_sdk.types.GenerationConfig(**gen_config_args))
            final_response = "".join(part.text for part in response.parts if hasattr(part, "text")).strip() if response.parts else ""
            if not final_response and response.prompt_feedback and response.prompt_feedback.block_reason: block_reason_msg = f"Blocked by API. Reason: {response.prompt_feedback.block_reason.name}"; logger.warning(f"Gemini response for '{model}' empty due to blocking: {block_reason_msg}"); return f"Error: Gemini response blocked - {block_reason_msg}"
            if json_mode and final_response and not (final_response.startswith('{') and final_response.endswith('}')) and not (final_response.startswith('[') and final_response.endswith(']')): logger.warning(f"Gemini: Model '{model}' requested JSON, output non-JSON. Snippet: {final_response[:100]}")
            logger.debug(f"Gemini response for '{model}' (first 100): {final_response[:100]}"); return final_response
        except Exception as e: logger.error(f"Gemini API call failed for '{model}': {e}", exc_info=True);
        if hasattr(e, 'message') and ("API key" in e.message or "permission" in e.message): return f"Error: Gemini API key/permission issue for '{model}'."
        return f"Error: Gemini call failed for '{model}' - {type(e).__name__}: {e}"

class MockLLMHandler(LLMHandlerInterface): # pragma: no cover
    # (Same as previous MockLLMHandler)
    def __init__(self, model_name: Optional[str] = "mock_generic_model"): super().__init__("mock", model_name, None, None); logger.info(f"MockLLMHandler initialized for model '{self.model_name_for_api}'.")
    async def generate_text(self, prompt: str, model: str, max_tokens: Optional[int]=2048, temperature: Optional[float]=0.7, json_mode: Optional[bool]=False, **kwargs: Any) -> Optional[str]:
        logger.debug(f"MockLLM Call: Model='{self.model_name_for_api}', JSON={json_mode}, Prompt(start): {prompt[:150]}..."); await asyncio.sleep(0.01 + (len(prompt) % 100) * 0.001)
        if json_mode:
            if "system_analysis" in prompt.lower() or "improvement_suggestions" in prompt.lower(): return json.dumps({"improvement_suggestions": [{"target_component_path": f"mindx.mock.{self.model_name_for_api}.file", "suggestion": f"Mock: Add comments.", "priority": 5, "is_critical_target": False}]})
            elif "critique" in prompt.lower(): return json.dumps({"score": 0.78, "justification": f"Mock critique from {self.model_name_for_api}: Plausible."})
            else: return json.dumps({"mock_provider": self.provider_name, "model": self.model_name_for_api, "message": "JSON mock response"})
        if "ONLY the complete, new content" in prompt: return f"# MOCK: Code by {self.model_name_for_api}\ndef new_mock(): pass"
        elif "Proposed Improvement Description:" in prompt.lower(): return f"Mock suggestion from {self.model_name_for_api}: Refactor for clarity."
        return f"MOCK general text response from {self.provider_name}/{self.model_name_for_api} for: {prompt[:70]}..."


# --- LLM Factory Cache and Global Config Loading ---
_llm_handler_cache: Dict[Tuple[str, str, Optional[str], Optional[str]], LLMHandlerInterface] = {}
_llm_handler_cache_lock = asyncio.Lock()
_factory_config_data: Optional[Dict[str, Any]] = None
_factory_config_loaded_flag = False

def _load_llm_factory_config_json() -> Dict[str, Any]: # pragma: no cover
    """Loads the llm_factory_config.json file. Caches result."""
    global _factory_config_data, _factory_config_loaded_flag
    if _factory_config_loaded_flag:
        return _factory_config_data or {}

    default_factory_config_path = PROJECT_ROOT / "mindx" / "data" / "config" / "llm_factory_config.json"
    factory_config_path_str = Config().get("llm.factory_config_path", str(default_factory_config_path))
    factory_config_path = Path(factory_config_path_str)

    if factory_config_path.exists() and factory_config_path.is_file():
        try:
            with factory_config_path.open("r", encoding="utf-8") as f:
                _factory_config_data = json.load(f)
            logger.info(f"LLMFactory: Loaded specific configuration from {factory_config_path}")
        except Exception as e:
            logger.error(f"LLMFactory: Error loading config {factory_config_path}: {e}. Using empty factory config.")
            _factory_config_data = {}
    else:
        logger.info(f"LLMFactory: Specific config file {factory_config_path} not found. Using global Config and internal defaults.")
        _factory_config_data = {}
    
    _factory_config_loaded_flag = True
    return _factory_config_data

async def create_llm_handler(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None, 
    api_key: Optional[str] = None, 
    base_url: Optional[str] = None
) -> LLMHandlerInterface:
    """
    Factory function to create or retrieve a cached LLMHandler instance.
    Loads specific factory config from `llm_factory_config.json` for overrides.
    """
    global_config = Config() # For global llm.* settings
    factory_config = _load_llm_factory_config_json() # Load/get cached factory-specific JSON config

    # 1. Determine Provider
    eff_provider_name = provider_name or \
                        factory_config.get("default_provider_override") or \
                        global_config.get("llm.default_provider", "ollama")
    eff_provider_name = eff_provider_name.lower()

    # 2. Determine Model Name
    eff_model_name = model_name # Direct arg highest precedence
    if not eff_model_name: # Try factory_config specific override for this provider
        eff_model_name = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("default_model_override")
    if not eff_model_name: # Try global config for this provider
        eff_model_name = global_config.get(f"llm.{eff_provider_name}.default_model")
    
    # Cloud Run friendly default for Ollama if still no model name
    if not eff_model_name and eff_provider_name == "ollama":
        logger.info(f"LLMFactory: No Ollama model specified. Selecting Cloud Run friendly default.")
        cr_models_config = factory_config.get("ollama_settings_for_factory", {}).get("cloud_run_friendly_models", DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY)
        
        # Prefer coding models
        for cr_model_id in cr_models_config.get("default_coding_preference_order", []):
            for category_list_key in ["good_fit", "excellent_fit"]:
                model_detail = next((m for m in cr_models_config.get(category_list_key, []) if m["id"] == cr_model_id), None)
                if model_detail: eff_model_name = model_detail["ollama_name"]; break
            if eff_model_name: break
        
        if not eff_model_name: # Fallback to general models
            for cr_model_id in cr_models_config.get("default_general_preference_order", []):
                for category_list_key in ["good_fit", "excellent_fit"]:
                    model_detail = next((m for m in cr_models_config.get(category_list_key, []) if m["id"] == cr_model_id), None)
                    if model_detail: eff_model_name = model_detail["ollama_name"]; break
                if eff_model_name: break
        
        if eff_model_name: logger.info(f"LLMFactory: Defaulting Ollama to Cloud Run friendly: '{eff_model_name}'")
        else: eff_model_name = global_config.get("llm.ollama.default_model", "nous-hermes2:latest"); logger.warning(f"LLMFactory: No Cloud Run preferred model found, Ollama fallback to '{eff_model_name}'.")

    elif not eff_model_name: # For other providers, ultimate fallback if no model name yet
        eff_model_name = f"default_model_for_{eff_provider_name}"


    # 3. Determine API Key
    eff_api_key = api_key
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic"]: # Add other API key providers
        eff_api_key = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("api_key_override")
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic"]:
        eff_api_key = global_config.get(f"llm.{eff_provider_name}.api_key")
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic"]: # Last resort, check non-MINDX_ env var
        eff_api_key = os.getenv(f"{eff_provider_name.upper()}_API_KEY")


    # 4. Determine Base URL (mainly for Ollama)
    eff_base_url = base_url
    if not eff_base_url and eff_provider_name == "ollama":
        eff_base_url = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("base_url_override")
    if not eff_base_url and eff_provider_name == "ollama":
        eff_base_url = global_config.get(f"llm.ollama.base_url")

    # Model name for API call (passed to handler's __init__)
    # This is the name the handler uses internally. It's usually eff_model_name unless overridden by factory_config
    api_model_name_for_handler = factory_config.get("provider_specific_handler_config", {}).get(eff_provider_name, {}).get("default_model_for_api_call", eff_model_name)


    cache_key = (eff_provider_name, api_model_name_for_handler or "unspecified_model", eff_api_key or "no_key", eff_base_url or "no_url")

    async with _llm_handler_cache_lock:
        if cache_key in _llm_handler_cache: # pragma: no cover
            logger.debug(f"LLMFactory: Returning cached LLMHandler for key: {cache_key}")
            return _llm_handler_cache[cache_key]

        handler_instance: LLMHandlerInterface
        if eff_provider_name == "ollama":
            handler_instance = OllamaHandler(model_name_for_api=api_model_name_for_handler, base_url=eff_base_url)
        elif eff_provider_name == "gemini":
            handler_instance = GeminiHandler(model_name_for_api=api_model_name_for_handler, api_key=eff_api_key)
        # Add elif for "openai", "anthropic", etc.
        else: # pragma: no cover
            logger.warning(f"LLMFactory: Unknown or unconfigured LLM provider '{eff_provider_name}'. Using MockLLMHandler for model '{api_model_name_for_handler}'.")
            handler_instance = MockLLMHandler(model_name=api_model_name_for_handler)
        
        _llm_handler_cache[cache_key] = handler_instance
        return handler_instance
