# Model Registry (`model_registry.py`)

## Introduction

The `ModelRegistry` class in the MindX LLM module (Augmentic Project) serves as a centralized component for discovering, initializing, and providing access to various Large Language Model (LLM) handlers. It ensures that LLM client instances (which implement the `LLMInterface`) are created once based on global configuration and then cached for reuse throughout the MindX system. This promotes efficient resource use and consistent LLM interaction patterns.

## Explanation

### Core Functionality

1.  **Singleton Pattern:**
    *   The `ModelRegistry` is implemented as a singleton. This means only one instance of the registry exists per application run.
    *   Access is provided via factory functions: `get_model_registry_async()` (for asynchronous contexts) and `get_model_registry()` (for synchronous contexts).
    *   An `asyncio.Lock` (`_lock`) is used within the async factory to ensure thread-safe and task-safe instantiation during concurrent startup scenarios.
    *   A `test_mode` flag in the constructor and factories allows for bypassing the singleton behavior and re-initializing the registry for isolated testing. `reset_instance_async()` is also provided for tests.

2.  **Configuration-Driven Initialization (`_load_and_initialize_configured_providers`):**
    *   On initialization, the `ModelRegistry` reads the `llm.providers` section from the global `Config` object.
    *   This section is expected to be a dictionary where each key is a provider name (e.g., "ollama", "gemini") and the value is a dictionary of provider-specific settings, primarily an `enabled: true/false` flag.
        ```json
        // Example in mindx_config.json or via .env
        "llm": {
            "default_provider": "ollama",
            "providers": {
                "ollama": {
                    "enabled": true,
                    "default_model": "nous-hermes2:latest" // Optional: specific default for registry
                },
                "gemini": {
                    "enabled": true,
                    // API key typically comes from llm.gemini.api_key global config
                }
            },
            "ollama": { /* global ollama settings */ },
            "gemini": { /* global gemini settings */ }
        }
        ```
    *   For each provider marked as `"enabled": true`, the registry calls `mindx.llm.llm_factory.create_llm_handler(provider_name=...)`.
    *   The `create_llm_handler` function is responsible for instantiating the correct `LLMHandler` subclass (e.g., `OllamaHandler`, `GeminiHandler`) using further global configurations (like `llm.<provider_name>.default_model`, `llm.<provider_name>.api_key`, etc.).
    *   If the `llm.providers` section is missing or empty, it attempts to initialize a handler for the `llm.default_provider` as a fallback.
    *   Initialized handlers are stored in the `self.providers` dictionary, keyed by the lowercase provider name.

3.  **Handler Retrieval (`get_handler`):**
    *   Provides a method `get_handler(provider_name: str) -> Optional[LLMInterface]`.
    *   Other components (like `MultiModelAgent`) use this to obtain a ready-to-use LLM handler for a specific provider.
    *   If a handler for the requested provider was not pre-initialized (e.g., not enabled in `llm.providers` but globally configured under `llm.<provider_name>`), `get_handler` will attempt an on-demand creation and caching of the handler.

4.  **Listing Providers (`list_available_providers`):**
    *   Returns a list of provider names for which handlers have been successfully initialized and are available.

## Technical Details

-   **Dependencies:**
    -   `mindx.utils.config.Config`: For accessing system-wide configuration.
    -   `mindx.utils.logging_config.get_logger`: For standardized logging.
    -   `mindx.llm.llm_interface.LLMInterface`: The abstract base class (or protocol) that all LLM handlers must conform to.
    -   `mindx.llm.llm_factory.create_llm_handler`: The factory function that actually instantiates specific `LLMHandler` implementations.
-   **Error Handling:** Includes logging for errors during provider configuration loading or handler instantiation.
-   **Case-Insensitive Provider Names:** Provider names are handled case-insensitively (converted to lowercase internally).

## Usage

The `ModelRegistry` is typically initialized once at the start of the MindX application, often implicitly when `get_model_registry()` or `get_model_registry_async()` is first called. Other components then use the getter to access handlers.

```python
# In an async component like MultiModelAgent or CoordinatorAgent:
# from mindx.llm.model_registry import get_model_registry_async
# from mindx.llm.llm_interface import LLMInterface # For type hint

# async def some_method_in_agent(self):
#     registry = await get_model_registry_async() # Gets the singleton instance
    
#     ollama_handler: Optional[LLMInterface] = registry.get_handler("ollama")
#     if ollama_handler:
#         # Now use the handler. The handler itself knows its default model from factory init.
#         # The 'model' arg to generate is the specific model_name_for_api.
#         response = await ollama_handler.generate(
#             prompt="Translate to French: Hello World", 
#             model="nous-hermes2:latest" # Example specific model for this call
#         )
#         print(f"Ollama says: {response}")
#     else:
#         print("Ollama handler not available.")

#     available_providers = registry.list_available_providers()
#     print(f"Available LLM providers in registry: {available_providers}")
