# LLM Factory and Handlers (`llm_factory.py`) - v4
# (Modular Handlers, Groq, External JSON Config)

## Introduction

This module is the cornerstone of MindX's interactions with various Large Language Models (LLMs). It defines a common interface for LLM operations and provides a factory function to create and manage specific LLM provider handlers. This version (v4) emphasizes modularity by expecting concrete handlers (like `OllamaHandler`, `GeminiHandler`, `GroqHandler`) to reside in their own separate files within the `mindx/llm` package. It continues to support an external JSON configuration file (`llm_factory_config.json`) for fine-grained control over factory behavior and model preferences, including defaults suitable for environments like Google Cloud Run.

## Explanation

### Core Components

1.  **`LLMHandlerInterface` (in `llm_interface.py`):**
    *   An Abstract Base Class (or conceptual interface) defining the common contract all specific LLM provider handlers must implement.
    *   Key method: `async def generate_text(self, prompt: str, model: str, max_tokens: Optional[int], temperature: Optional[float], json_mode: Optional[bool], **kwargs: Any) -> Optional[str]`.
        -   `model`: This argument is the **specific model name/tag** that the provider's API expects (e.g., "llama3:8b-instruct-q5_K_M", "gemini-1.5-pro-latest").
        -   `json_mode`: A boolean hint. If `True`, the handler should instruct the LLM to return a valid JSON object string.

2.  **Concrete Handlers (in separate files, e.g., `ollama_handler.py`, `gemini_handler.py`, `groq_handler.py`, `mock_llm_handler.py`):**
    *   Each handler implements `LLMHandlerInterface`.
    *   **`OllamaHandler`**: Interacts with an Ollama server using direct HTTP API calls via `aiohttp` (instead of the `ollama` SDK for more control, as per user-provided code).
    *   **`GeminiHandler`**: Interacts with Google Gemini models via the `google-generativeai` SDK.
    *   **`GroqHandler`**: Interacts with Groq Cloud's API via the `groq` Python SDK.
    *   **`MockLLMHandler`**: A fallback for testing or unconfigured providers.
    *   Each handler is initialized with a `model_name_for_api` (its default model if `generate_text` is called without a specific `model` argument), and provider-specific needs like API keys or base URLs.

3.  **`OLLAMA_CLOUD_RUN_MODELS_PY` Constant (in `llm_factory.py`):**
    *   A default Python dictionary defining "Cloud Run Friendly" Ollama models, categorized by resource fit.
    *   This list can be **overridden or extended** by the `ollama_settings_for_factory.cloud_run_friendly_models` section in `llm_factory_config.json`.
    *   Includes `default_coding_preference_order` and `default_general_preference_order` to guide default model selection for Ollama if no specific model is configured.

4.  **`_load_llm_factory_config_json()` Helper Function (in `llm_factory.py`):**
    *   Loads settings from `PROJECT_ROOT/mindx/data/config/llm_factory_config.json` (path itself configurable via global `Config`). This JSON file allows granular control over factory behavior without code changes.

5.  **`create_llm_handler` Factory Function (in `llm_factory.py`):**
    *   This asynchronous function is the **sole recommended way** for MindX components to obtain an LLM handler.
    *   **Layered Configuration Sourcing (for provider, API model name, API key, base URL):**
        1.  Direct arguments passed to `create_llm_handler`.
        2.  `llm_factory_config.json` (e.g., `ollama_settings_for_factory.default_model_override`).
        3.  Global `Config` (`mindx_config.json` / `.env` / `MINDX_` Env Vars, e.g., `config.get("llm.ollama.default_model")`).
        4.  Internal Python defaults (like `OLLAMA_CLOUD_RUN_MODELS_PY` or hardcoded fallbacks).
    *   **Cloud Run Aware Default for Ollama:** If the provider is "ollama" and no specific model name is resolved through arguments or higher-precedence configs, it intelligently selects a default model from the "Cloud Run Friendly" lists (from JSON config or Python default), prioritizing coding models.
    *   **Handler Instantiation:** Based on the final effective provider name, it instantiates the appropriate handler class (e.g., `OllamaHandler`, `GeminiHandler`, `GroqHandler`). The `model_name_for_api` passed to the handler's constructor (which becomes the handler's internal default model) is also resolved through the configuration layers.
    *   **Caching:** Instantiated handlers are cached using a key derived from all effective construction parameters (provider, API model name, API key, base URL). An `asyncio.Lock` ensures cache safety.

## Technical Details

-   **Modular Handlers:** Each LLM provider's interaction logic is encapsulated in its own handler class in a separate file (e.g., `groq_handler.py`), imported by the factory. This improves organization and extensibility.
-   **External Factory Configuration:** `llm_factory_config.json` offers a dedicated place to fine-tune the factory's behavior and model preferences without altering global application configuration or code.
-   **Dynamic SDK Imports:** SDKs are typically imported within their respective handler files, allowing the factory to function even if not all optional LLM SDKs are installed (falling back to `MockLLMHandler` for missing ones).

## Usage

1.  **Ensure Dependencies:** Install necessary SDKs (e.g., `pip install groq ollama google-generativeai`).
2.  **Configure `.env`:** Include necessary API keys (e.g., `GROQ_API_KEY`, `MINDX_LLM__GEMINI__API_KEY`).
3.  **Create/Configure `PROJECT_ROOT/mindx/data/config/llm_factory_config.json` (Optional but Recommended):**
    Define provider preferences, model overrides, or custom Cloud Run model lists. See the example JSON provided with the Python code for structure.
4.  **Obtain and Use Handlers in MindX Components:**
    ```python
    from mindx.llm.llm_factory import create_llm_handler

    async def example_task_execution(prompt: str, preferred_provider: Optional[str] = None, specific_model: Optional[str] = None):
        try:
            # Factory resolves best handler and model based on args and configs
            handler = await create_llm_handler(provider_name=preferred_provider, model_name=specific_model)
            
            # The model to use for *this specific generate_text call* is handler.model_name_for_api
            # if you want the handler's configured default, or you can override it here.
            # If specific_model was passed to create_llm_handler, handler.model_name_for_api will be that.
            api_call_model = specific_model or handler.model_name_for_api 
            if not api_call_model: # Should not happen if factory logic is sound
                print("Error: No effective model name could be determined.")
                return

            response = await handler.generate_text(
                prompt=prompt,
                model=api_call_model, 
                max_tokens=1024,
                temperature=0.5
            )

            if response and not response.startswith("Error:"):
                print(f"Response from {handler.provider_name}/{api_call_model}:\n{response}")
            else:
                print(f"Error from {handler.provider_name}/{api_call_model}: {response}")

        except Exception as e:
            print(f"An error occurred: {e}")

    # Example calls:
    # asyncio.run(example_task_execution("Write a Python function for quicksort.", preferred_provider="ollama", specific_model="deepseek-coder:6.7b-instruct"))
    # asyncio.run(example_task_execution("Explain the theory of relativity briefly.", preferred_provider="gemini"))
    # asyncio.run(example_task_execution("What's the fastest way to summarize a document?", preferred_provider="groq")) 
    ```

This modular and highly configurable `LLMFactory` provides mindX with a robust and adaptable interface to a variety of Large Language Models. mindX should expand this as necessary.
