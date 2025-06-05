# mindx/llm/llm_factory.py
from typing import Optional, Any, Dict, List, Tuple
import json
import asyncio
import importlib
import os
from pathlib import Path
# Ensure logging is imported if used directly, or use the module's logger instance
import logging # Added for explicitness, though getLogger was used

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from .llm_interface import LLMHandlerInterface # Abstract base class

logger = get_logger(__name__) # Module-level logger

# Dynamically import concrete handlers from their own files
try:
    from .ollama_handler import OllamaHandler
except ImportError as e: # pragma: no cover
    OllamaHandler = None # type: ignore
    logger.warning(f"Could not import OllamaHandler: {e}. Ollama provider will be unavailable.")
try:
    from .gemini_handler import GeminiHandler
except ImportError as e: # pragma: no cover
    GeminiHandler = None # type: ignore
    logger.warning(f"Could not import GeminiHandler: {e}. Gemini provider will be unavailable.")
try:
    from .groq_handler import GroqHandler
except ImportError as e: # pragma: no cover
    GroqHandler = None # type: ignore
    logger.warning(f"Could not import GroqHandler: {e}. Groq provider will be unavailable.")
# Always import MockLLMHandler as a fallback
from .mock_llm_handler import MockLLMHandler


# Default Cloud Run friendly models if not specified in llm_factory_config.json
DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY = {
    "excellent_fit": [
        {"id": "phi3_py", "ollama_name": "phi3:mini", "type": "general/reasoning", "est_memory_needed_gb": 4, "notes": "Default Python list: Microsoft Phi-3 Mini."},
        {"id": "gemma_2b_py", "ollama_name": "gemma:2b", "type": "general", "est_memory_needed_gb": 3, "notes": "Default Python list: Google's 2B model."},
        {"id": "codegemma_2b_py", "ollama_name": "codegemma:2b", "type": "coding", "est_memory_needed_gb": 3, "notes": "Default Python list: Google's 2B code model."},
    ],
    "good_fit": [
        {"id": "llama3_8b_py", "ollama_name": "llama3:8b", "type": "general/chat", "est_memory_needed_gb": 8, "notes": "Default Python list: Meta's 8B model."},
        {"id": "mistral_7b_py", "ollama_name": "mistral:7b", "type": "general/chat", "est_memory_needed_gb": 8, "notes": "Default Python list: Mistral 7B model."},
        {"id": "codegemma_7b_py", "ollama_name": "codegemma:7b-it", "type": "coding", "est_memory_needed_gb": 8, "notes": "Default Python list: Google's 7B code model."},
    ],
    "default_coding_preference_order": ["codegemma_7b_py", "codegemma_2b_py", "phi3_py"], # Uses 'id' from above lists
    "default_general_preference_order": ["llama3_8b_py", "mistral_7b_py", "phi3_py", "gemma_2b_py"]
}

# --- Factory Function Cache and Config Loading ---
_llm_handler_cache: Dict[Tuple[str, str, Optional[str], Optional[str]], LLMHandlerInterface] = {}
_llm_handler_cache_lock = asyncio.Lock() # For async safety of cache access
_factory_config_data: Optional[Dict[str, Any]] = None
_factory_config_loaded_flag = False # To ensure JSON config is loaded only once

def _load_llm_factory_config_json() -> Dict[str, Any]: # pragma: no cover
    """Loads the llm_factory_config.json file. Caches result. Should be called once."""
    global _factory_config_data, _factory_config_loaded_flag
    if _factory_config_loaded_flag:
        return _factory_config_data if _factory_config_data is not None else {}

    # Default path for factory-specific config, relative to PROJECT_ROOT
    # This path can be overridden by a global Config entry 'llm.factory_config_path'
    app_config = Config() # Get a config instance
    default_factory_config_path = PROJECT_ROOT / "mindx" / "data" / "config" / "llm_factory_config.json"
    factory_config_path_str = app_config.get("llm.factory_config_path", str(default_factory_config_path))
    factory_config_path = Path(factory_config_path_str)

    if factory_config_path.exists() and factory_config_path.is_file():
        try:
            with factory_config_path.open("r", encoding="utf-8") as f:
                _factory_config_data = json.load(f)
            logger.info(f"LLMFactory: Loaded specific configuration from {factory_config_path}")
        except Exception as e:
            logger.error(f"LLMFactory: Error loading config file {factory_config_path}: {e}. Using empty factory config.")
            _factory_config_data = {} # Use empty if file is corrupt to prevent crash
    else:
        logger.info(f"LLMFactory: Specific config file {factory_config_path} not found. Using global Config and internal defaults only.")
        _factory_config_data = {} # Empty if not found, defaults will apply

    _factory_config_loaded_flag = True
    return _factory_config_data if _factory_config_data is not None else {}

async def create_llm_handler(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None, # This is the specific model/tag for the API
    api_key: Optional[str] = None, # Direct override for API key
    base_url: Optional[str] = None  # Direct override for base URL (e.g., for Ollama)
) -> LLMHandlerInterface:
    """
    Factory function to create or retrieve a cached LLMHandler instance.
    Configuration is sourced with precedence: Direct args > llm_factory_config.json > Global Config > Code Defaults.
    Includes logic for selecting a Cloud Run friendly Ollama model if no specific model is given.
    """
    global_config = Config()
    factory_config = _load_llm_factory_config_json() # Load/get cached factory-specific JSON config

    # --- 1. Determine Effective Provider Name ---
    eff_provider_name = provider_name # Direct argument
    if not eff_provider_name: # Factory JSON config's preference order
        pref_order = factory_config.get("default_provider_preference_order")
        eff_provider_name = pref_order[0] if pref_order and isinstance(pref_order, list) and len(pref_order) > 0 else None
    if not eff_provider_name: # Global config default
        eff_provider_name = global_config.get("llm.default_provider", "ollama") # Final fallback: ollama
    eff_provider_name = eff_provider_name.lower()


    # --- 2. Determine Effective Model Name (this becomes model_name_for_api for the handler) ---
    eff_model_name_for_api = model_name # Direct argument highest precedence

    if not eff_model_name_for_api: # Try factory_config specific default for this provider
        eff_model_name_for_api = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("default_model_override")

    if not eff_model_name_for_api: # Try global config for this provider's default model
        eff_model_name_for_api = global_config.get(f"llm.{eff_provider_name}.default_model")

    # Special default logic for Ollama if still no model name (e.g., for Cloud Run friendliness)
    if not eff_model_name_for_api and eff_provider_name == "ollama": # pragma: no cover
        logger.info(f"LLMFactory: No specific Ollama model set via args or config. Selecting Cloud Run friendly default.")
        cr_models_config_source = factory_config.get("ollama_settings_for_factory", {}).get("cloud_run_friendly_models")
        if cr_models_config_source is None:
            cr_models_config_source = DEFAULT_OLLAMA_CLOUD_RUN_MODELS_PY
            logger.debug("LLMFactory: Using internal Python list for Ollama Cloud Run friendly models.")
        else:
            logger.debug("LLMFactory: Using llm_factory_config.json for Ollama Cloud Run friendly models.")

        coding_pref_order = cr_models_config_source.get("default_coding_preference_order", [])
        all_cr_models_list = cr_models_config_source.get("good_fit", []) + cr_models_config_source.get("excellent_fit", [])

        for cr_model_id_pref in coding_pref_order:
            model_detail = next((m_dict for m_dict in all_cr_models_list if m_dict.get("id") == cr_model_id_pref), None)
            if model_detail and model_detail.get("ollama_name"):
                eff_model_name_for_api = model_detail["ollama_name"]; break

        if not eff_model_name_for_api:
            general_pref_order = cr_models_config_source.get("default_general_preference_order", [])
            for cr_model_id_pref in general_pref_order:
                model_detail = next((m_dict for m_dict in all_cr_models_list if m_dict.get("id") == cr_model_id_pref), None)
                if model_detail and model_detail.get("ollama_name"):
                    eff_model_name_for_api = model_detail["ollama_name"]; break

        if eff_model_name_for_api:
            logger.info(f"LLMFactory: Defaulting Ollama to Cloud Run friendly model: '{eff_model_name_for_api}'")
        else:
            eff_model_name_for_api = global_config.get("llm.ollama.default_model", "phi3:mini") # A more common small default
            logger.warning(f"LLMFactory: No Cloud Run preferred model found for Ollama, using global/hardcoded Ollama fallback: '{eff_model_name_for_api}'.")

    elif not eff_model_name_for_api: # For providers other than Ollama, if no model_name yet
        eff_model_name_for_api = global_config.get(f"llm.{eff_provider_name}.default_model", f"default_model_for_{eff_provider_name}")


    # --- 3. Determine API Key ---
    eff_api_key = api_key # Direct arg
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic", "groq"]:
        eff_api_key = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("api_key_override")
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic", "groq"]:
        eff_api_key = global_config.get(f"llm.{eff_provider_name}.api_key")
    if not eff_api_key and eff_provider_name in ["gemini", "openai", "anthropic", "groq"]:
        env_var_name = ""
        if eff_provider_name == "gemini": env_var_name = "GEMINI_API_KEY" # Specific to GeminiHandler's expectation
        elif eff_provider_name == "openai": env_var_name = "OPENAI_API_KEY"
        elif eff_provider_name == "anthropic": env_var_name = "ANTHROPIC_API_KEY"
        elif eff_provider_name == "groq": env_var_name = "GROQ_API_KEY"
        if env_var_name: eff_api_key = os.getenv(env_var_name)


    # --- 4. Determine Base URL (primarily for Ollama) ---
    eff_base_url = base_url # Direct arg
    if not eff_base_url and eff_provider_name == "ollama":
        eff_base_url = factory_config.get(f"{eff_provider_name}_settings_for_factory", {}).get("base_url_override")
    if not eff_base_url and eff_provider_name == "ollama":
        eff_base_url = global_config.get(f"llm.ollama.base_url")

    final_model_name_for_handler_constructor = factory_config.get("provider_specific_handler_config", {}).get(eff_provider_name, {}).get("default_model_for_api_call", eff_model_name_for_api)

    cache_key_model_name = final_model_name_for_handler_constructor or f"implicit_default_for_{eff_provider_name}"
    cache_key = (eff_provider_name, cache_key_model_name, eff_api_key or "no_key_for_handler", eff_base_url or "no_url_for_handler")

    async with _llm_handler_cache_lock:
        if cache_key in _llm_handler_cache: # pragma: no cover
            logger.debug(f"LLMFactory: Returning cached LLMHandler for key: {cache_key}")
            return _llm_handler_cache[cache_key]

        handler_instance: LLMHandlerInterface
        model_arg_for_handler = final_model_name_for_handler_constructor # Model name for the handler's constructor

        if eff_provider_name == "ollama":
            if OllamaHandler: handler_instance = OllamaHandler(model_name_for_api=model_arg_for_handler, base_url=eff_base_url)
            else: logger.error("LLMFactory: OllamaHandler not imported, cannot create instance."); handler_instance = MockLLMHandler(model_name=model_arg_for_handler)
        elif eff_provider_name == "gemini":
            if GeminiHandler: handler_instance = GeminiHandler(model_name_for_api=model_arg_for_handler, api_key=eff_api_key)
            else: logger.error("LLMFactory: GeminiHandler not imported, cannot create instance."); handler_instance = MockLLMHandler(model_name=model_arg_for_handler)
        elif eff_provider_name == "groq":
            if GroqHandler: handler_instance = GroqHandler(model_name_for_api=model_arg_for_handler, api_key=eff_api_key)
            else: logger.error("LLMFactory: GroqHandler not imported, cannot create instance."); handler_instance = MockLLMHandler(model_name=model_arg_for_handler)
        # Example for OpenAI:
        # elif eff_provider_name == "openai":
        #     try:
        #         from .openai_handler import OpenAIHandler # Assuming you create this
        #         handler_instance = OpenAIHandler(model_name_for_api=model_arg_for_handler, api_key=eff_api_key)
        #     except ImportError: # pragma: no cover
        #         logger.error("LLMFactory: OpenAIHandler not imported, cannot create instance.")
        #         handler_instance = MockLLMHandler(model_name=model_arg_for_handler)
        else: # pragma: no cover
            logger.warning(f"LLMFactory: Unknown or unconfigured LLM provider '{eff_provider_name}'. Using MockLLMHandler for model '{model_arg_for_handler}'.")
            handler_instance = MockLLMHandler(model_name=model_arg_for_handler)

        _llm_handler_cache[cache_key] = handler_instance
        logger.info(f"LLMFactory: Created and cached LLMHandler for {eff_provider_name} with model '{model_arg_for_handler}' (Cache Key: {cache_key})")
        return handler_instance
