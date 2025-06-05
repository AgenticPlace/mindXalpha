# mindx/llm/llm_factory.py
from typing import Optional, Any, Dict, List, Tuple
import json
import asyncio
import importlib 
import os
from pathlib import Path

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from .llm_interface import LLMHandlerInterface # Abstract base class
# Import concrete handlers from their own files
from .ollama_handler import OllamaHandler # Assuming ollama_handler.py exists
from .gemini_handler import GeminiHandler # Assuming gemini_handler.py exists
from .groq_handler import GroqHandler     # NEW: Assuming groq_handler.py exists
from .mock_llm_handler import MockLLMHandler # Assuming mock_llm_handler.py exists

logger = get_logger(__name__)

# OLLAMA_CLOUD_RUN_MODELS_PY: This constant could also move to llm_factory_config.json
# or be part of OllamaHandler if it's very specific to it. Keeping here for now for visibility.
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

# --- Factory Function Cache and Config Loading ---
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
        logger.info(f"LLMFactory: Specific config file {factory_config_path} not found. Using global Config and internal defaults only.")
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
    Includes logic for selecting a Cloud Run friendly Ollama model if no specific model is given.
    """
    global_config = Config() 
    factory_config = _load_llm_factory_config_json()

    # 1. Determine Effective Provider Name
    eff_provider_name = provider_name or \
                        (factory_config.get("default_provider_preference_order", [])[0] if factory_config.get("default_provider_preference_order") else None) or \
                        global_config.get("llm.default_provider", "ollama")
    eff_provider_name = eff_provider_name.lower()

    # 2. Determine Effective Model Name (model_name_for_api)
    eff_model_name_for_api = model_name 
    if not eff_model_name_for_api: 
        eff_model_name_for_api = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("default_model_override")
    if not eff_model_name_for_api: 
        eff_model_name_for_api = global_config.get(f"llm.{eff_provider_name}.default_model")
    
    if not eff_model_name_for_api and eff_provider_name == "ollama": # pragma: no cover
        logger.info(f"LLMFactory: No specific Ollama model set. Selecting Cloud Run friendly default.")
        cr_models_config = factory_config.get("ollama_settings_for_factory", {}).get("cloud_run_friendly_models", DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY)
        coding_pref = cr_models_config.get("default_coding_preference_order", DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY["default_coding_preference_order"])
        general_pref = cr_models_config.get("default_general_preference_order", DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY["default_general_preference_order"])
        all_cr = cr_models_config.get("good_fit", DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY["good_fit"]) + \
                 cr_models_config.get("excellent_fit", DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY["excellent_fit"])
        for cr_id in coding_pref: model_detail = next((m for m in all_cr if m["id"] == cr_id), None);
        if model_detail: eff_model_name_for_api = model_detail["ollama_name"]; break
        if not eff_model_name_for_api:
            for cr_id in general_pref: model_detail = next((m for m in all_cr if m["id"] == cr_id), None);
            if model_detail: eff_model_name_for_api = model_detail["ollama_name"]; break
        if eff_model_name_for_api: logger.info(f"LLMFactory: Defaulting Ollama to Cloud Run friendly: '{eff_model_name_for_api}'")
        else: eff_model_name_for_api = global_config.get("llm.ollama.default_model", "nous-hermes2:latest"); logger.warning(f"LLMFactory: No Cloud Run preferred model, Ollama fallback to '{eff_model_name_for_api}'.")

    elif not eff_model_name_for_api: 
        eff_model_name_for_api = global_config.get(f"llm.{eff_provider_name}.default_model", f"default_model_for_{eff_provider_name}")

    # 3. Determine API Key
    eff_api_key = api_key
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic", "groq"]:
        eff_api_key = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("api_key_override")
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic", "groq"]:
        eff_api_key = global_config.get(f"llm.{eff_provider_name}.api_key")
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic", "groq"]:
        eff_api_key = os.getenv(f"{eff_provider_name.upper()}_API_KEY") # Fallback to direct env var

    # 4. Determine Base URL
    eff_base_url = base_url
    if not eff_base_url and eff_provider_name == "ollama":
        eff_base_url = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("base_url_override")
    if not eff_base_url and eff_provider_name == "ollama":
        eff_base_url = global_config.get(f"llm.ollama.base_url")

    # This is the model name passed to the Handler's constructor.
    final_model_name_for_handler = factory_config.get("provider_specific_handler_config", {}).get(eff_provider_name, {}).get("default_model_for_api_call", eff_model_name_for_api)

    cache_key = (eff_provider_name, final_model_name_for_handler or "unspecified_model", eff_api_key or "no_key", eff_base_url or "no_url")

    async with _llm_handler_cache_lock:
        if cache_key in _llm_handler_cache: # pragma: no cover
            logger.debug(f"LLMFactory: Returning cached LLMHandler for key: {cache_key}")
            return _llm_handler_cache[cache_key]

        handler_instance: LLMHandlerInterface
        if eff_provider_name == "ollama":
            handler_instance = OllamaHandler(model_name_for_api=final_model_name_for_handler, base_url=eff_base_url)
        elif eff_provider_name == "gemini":
            handler_instance = GeminiHandler(model_name_for_api=final_model_name_for_handler, api_key=eff_api_key)
        elif eff_provider_name == "groq": # ADDED GROQ
            handler_instance = GroqHandler(model_name_for_api=final_model_name_for_handler, api_key=eff_api_key)
        # Add elif for "openai", "anthropic", etc. here
        # elif eff_provider_name == "openai":
        #     from .openai_handler import OpenAIHandler # Example
        #     handler_instance = OpenAIHandler(model_name_for_api=final_model_name_for_handler, api_key=eff_api_key)
        else: # pragma: no cover
            logger.warning(f"LLMFactory: Unknown or unconfigured LLM provider '{eff_provider_name}'. Using MockLLMHandler for model '{final_model_name_for_handler}'.")
            handler_instance = MockLLMHandler(model_name=final_model_name_for_handler)
        
        _llm_handler_cache[cache_key] = handler_instance
        return handler_instance
