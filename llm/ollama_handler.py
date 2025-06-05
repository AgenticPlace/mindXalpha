# mindx/llm/ollama_handler.py
"""
LLM Handler for Ollama (Local LLMs).
Interacts directly with an Ollama service API.
"""
import logging
import subprocess
import asyncio
import json # Standard library json
from typing import Dict, Any, Optional, List

# Try to import aiohttp, otherwise log an error
try:
    import aiohttp
except ImportError: # pragma: no cover
    aiohttp = None # type: ignore 
    logging.getLogger(__name__).error(
        "OllamaHandler: aiohttp library not found. Please 'pip install aiohttp'. "
        "Ollama API calls will fail."
    )

from mindx.utils.logging_config import get_logger
from .llm_interface import LLMHandlerInterface # Ensure this is in __init__.py or path correct

logger = get_logger(__name__)

class OllamaHandler(LLMHandlerInterface): # pragma: no cover
    """
    Handles interactions with an Ollama service using direct HTTP API calls.
    Implements the LLMHandlerInterface for integration with MindX LLMFactory.
    """
    def __init__(self, model_name_for_api: Optional[str], # Default model this instance is configured for
                       api_key: Optional[str] = None, # Not used by Ollama
                       base_url: Optional[str] = None):
        
        # model_name_for_api is the default model tag (e.g., "llama3:8b") this handler might use
        # if no specific model is passed to generate_text.
        # base_url is like "http://localhost:11434"
        super().__init__("ollama", model_name_for_api, api_key, base_url)
        
        self.api_base_url = self.base_url if self.base_url else "http://localhost:11434"
        if not self.api_base_url.endswith("/api"):
            self.api_url = f"{self.api_base_url.rstrip('/')}/api"
        else: # pragma: no cover # Should not happen if base_url is clean
            self.api_url = self.api_base_url 
        
        self.http_client_session: Optional[aiohttp.ClientSession] = None
        logger.info(f"OllamaHandler initialized for API base: {self.api_base_url}. Default model (if any): {self.model_name_for_api}")

    async def _get_client_session(self) -> aiohttp.ClientSession:
        """Manages and returns an aiohttp.ClientSession."""
        if self.http_client_session is None or self.http_client_session.closed: # pragma: no cover
            # You might want to configure timeouts, connectors, etc. here
            self.http_client_session = aiohttp.ClientSession(json_serialize=json.dumps)
        return self.http_client_session

    async def generate_text(self, prompt: str, model: str, 
                            max_tokens: Optional[int] = None, # Ollama calls this 'num_predict'
                            temperature: Optional[float] = 0.7,
                            json_mode: Optional[bool] = False, 
                            **kwargs: Any) -> Optional[str]:
        """
        Generates text using the specified Ollama model via HTTP API.

        Args:
            prompt: The input prompt.
            model: The Ollama model tag to use (e.g., "llama3:8b", "codegemma:7b").
            max_tokens: Corresponds to Ollama's 'num_predict'. If None or 0, Ollama might use model default or fill context.
            temperature: Sampling temperature.
            json_mode: If True, requests JSON output from Ollama.
            **kwargs: Can include 'options' dict for Ollama, or 'stop_sequences'.

        Returns:
            The generated text string, or an error string starting with "Error:".
        """
        if not model: # pragma: no cover
            logger.error(f"{self.provider_name}Handler: No model specified for generate_text call.")
            return "Error: No model specified for generation."
        if not aiohttp: # pragma: no cover
             return "Error: aiohttp library not installed for OllamaHandler."

        session = await self._get_client_session()
        generate_endpoint = f"{self.api_url}/generate"
        
        ollama_options: Dict[str, Any] = kwargs.get("options", {})
        if temperature is not None: ollama_options["temperature"] = temperature
        if max_tokens is not None and max_tokens > 0: ollama_options["num_predict"] = max_tokens
        # else: ollama_options["num_predict"] = -1 # Generate until stop or context full, might be very long

        if "stop_sequences" in kwargs and isinstance(kwargs["stop_sequences"], list):
            ollama_options["stop"] = kwargs["stop_sequences"]

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False, # This handler implements non-streaming for generate_text
            "options": ollama_options
        }
        if json_mode:
            payload["format"] = "json"

        logger.debug(f"OllamaHandler: POST to {generate_endpoint}. Model: {model}. JSON Mode: {json_mode}. Payload options: {ollama_options}. Prompt (start): {prompt[:100]}...")
        
        response_content_full = ""
        try:
            async with session.post(generate_endpoint, json=payload) as response:
                if response.status != 200: # pragma: no cover
                    error_text = await response.text()
                    logger.error(f"Ollama API error ({response.status}) for model '{model}': {error_text[:500]}")
                    return f"Error: Ollama API request failed with status {response.status} - {error_text[:200]}"

                # Ollama's non-streaming /api/generate returns a single JSON object
                # when stream=false. If it were stream=true, we'd iterate response.content.
                response_data = await response.json(loads=json.loads) # Use standard json
                
                if "response" in response_data:
                    response_content_full = response_data["response"].strip()
                elif "error" in response_data: # pragma: no cover
                    logger.error(f"Ollama API returned an error in JSON for model '{model}': {response_data['error']}")
                    return f"Error: {response_data['error']}"
                else: # pragma: no cover
                    logger.warning(f"Ollama response for model '{model}' missing 'response' field. Full data: {str(response_data)[:500]}")
                    # Fallback: try to concatenate all string values if it's a stream of JSON objects that was mis-handled
                    if isinstance(response_data, list) and all(isinstance(item, dict) for item in response_data):
                        for item_data in response_data:
                            if "response" in item_data: response_content_full += item_data["response"]
                            if item_data.get("done"): break
                    response_content_full = response_content_full.strip()


            if json_mode and response_content_full and \
               not (response_content_full.startswith('{') and response_content_full.endswith('}')) and \
               not (response_content_full.startswith('[') and response_content_full.endswith(']')): # pragma: no cover
                logger.warning(f"OllamaHandler: Model '{model}' requested JSON but output seems non-JSON. Snippet: {response_content_full[:100]}")

            logger.debug(f"OllamaHandler response for '{model}' (first 100 chars): {response_content_full[:100]}")
            return response_content_full

        except aiohttp.ClientConnectorError as e_conn: # pragma: no cover
            logger.error(f"OllamaHandler: Connection error for model '{model}' at {self.api_base_url}: {e_conn}", exc_info=True)
            return f"Error: Ollama connection error - {e_conn}"
        except Exception as e: # pragma: no cover
            logger.error(f"OllamaHandler: Exception during API call for model '{model}': {e}", exc_info=True)
            return f"Error: Ollama API call exception for '{model}' - {type(e).__name__}: {e}"

    # --- Utility methods (not part of LLMHandlerInterface, but useful for an Ollama tool/agent) ---
    def check_ollama_installation_sync(self) -> bool: # pragma: no cover
        """Synchronously checks if the 'ollama' CLI command is available."""
        command = "ollama --version" # A less intrusive command than 'list' for just checking install
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and "version" in result.stdout.lower():
                logger.info(f"Ollama CLI is installed and accessible. Version: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"Ollama CLI not accessible or version command failed. stderr: {result.stderr.strip()}")
                return False
        except FileNotFoundError: # pragma: no cover
             logger.error("Ollama CLI command 'ollama' not found. Is Ollama installed and in PATH?")
             return False
        except subprocess.TimeoutExpired: # pragma: no cover
             logger.error("Ollama CLI version check timed out.")
             return False
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to check Ollama installation via CLI: {e}")
            return False

    async def list_local_models_api(self) -> List[Dict[str, Any]]: # pragma: no cover
        """Asynchronously lists models available via the Ollama API /api/tags."""
        if not aiohttp: return [{"error": "aiohttp library not installed"}]
        list_endpoint = f"{self.api_url}/tags"
        session = await self._get_client_session()
        models_list = []
        try:
            logger.debug(f"OllamaHandler: GET from {list_endpoint}")
            async with session.get(list_endpoint, timeout=10) as response:
                if response.status == 200:
                    data = await response.json(loads=json.loads)
                    models_list = data.get("models", []) # Structure is {"models": [{"name": "model:tag", ...}]}
                    logger.info(f"OllamaHandler: API listed {len(models_list)} models.")
                else: # pragma: no cover
                    error_text = await response.text()
                    logger.error(f"Ollama API error listing models ({response.status}): {error_text[:500]}")
                    return [{"error": f"API error {response.status}", "details": error_text[:200]}]
        except Exception as e: # pragma: no cover
            logger.error(f"Ollama API error listing models: {e}", exc_info=True)
            return [{"error": f"Exception listing models: {e}"}]
        return models_list

    async def get_model_info_api(self, model_name_tag: str) -> Dict[str, Any]: # pragma: no cover
        """Asynchronously gets detailed information for a specific model via /api/show."""
        if not aiohttp: return {"error": "aiohttp library not installed"}
        show_endpoint = f"{self.api_url}/show"
        session = await self._get_client_session()
        payload = {"name": model_name_tag}
        try:
            logger.debug(f"OllamaHandler: POST to {show_endpoint} for model {model_name_tag}")
            async with session.post(show_endpoint, json=payload, timeout=15) as response:
                if response.status == 200:
                    model_info = await response.json(loads=json.loads)
                    logger.info(f"OllamaHandler: Retrieved info for model '{model_name_tag}'.")
                    return model_info
                else: # pragma: no cover
                    error_text = await response.text()
                    logger.error(f"Ollama API error showing model info ({response.status}) for '{model_name_tag}': {error_text[:500]}")
                    return {"error": f"API error {response.status}", "details": error_text[:200]}
        except Exception as e: # pragma: no cover
            logger.error(f"Ollama API error showing model info for '{model_name_tag}': {e}", exc_info=True)
            return {"error": f"Exception showing model info: {e}"}

    async def pull_model_api(self, model_name_tag: str, insecure: bool = False, stream: bool = False) -> Dict[str, Any]: # pragma: no cover
        """
        Asynchronously pulls a model via the Ollama API /api/pull.
        Returns the final status message. If streaming, this might be complex.
        For simplicity, this stub returns the *last* message if streaming.
        """
        if not aiohttp: return {"error": "aiohttp library not installed", "status": "ERROR"}
        pull_endpoint = f"{self.api_url}/pull"
        session = await self._get_client_session()
        payload = {"name": model_name_tag, "insecure": insecure, "stream": stream}
        logger.info(f"OllamaHandler: Initiating pull for model '{model_name_tag}' via API (stream: {stream})...")
        final_status_message = {"status": "ERROR", "message": "Pull did not complete as expected."}
        try:
            async with session.post(pull_endpoint, json=payload, timeout=None) as response: # Timeout None for long pulls
                if response.status != 200: # pragma: no cover
                    error_text = await response.text()
                    logger.error(f"Ollama API error initiating pull ({response.status}) for '{model_name_tag}': {error_text[:500]}")
                    return {"error": f"API error {response.status}", "details": error_text[:200], "status": "ERROR"}
                
                async for line_bytes in response.content: # Handles streaming output
                    if line_bytes:
                        line_str = line_bytes.decode('utf-8').strip()
                        try:
                            status_data = json.loads(line_str)
                            logger.debug(f"Ollama Pull '{model_name_tag}': {status_data.get('status')} - {status_data.get('digest','')} {status_data.get('total','')} {status_data.get('completed','')}")
                            final_status_message = status_data # Keep last status
                            if "error" in status_data: # pragma: no cover
                                logger.error(f"Ollama Pull '{model_name_tag}' stream error: {status_data['error']}")
                                final_status_message["status"] = "ERROR" # Ensure overall status is error
                                break
                            if status_data.get("status", "").lower() == "success":
                                logger.info(f"Ollama Pull '{model_name_tag}' completed successfully via API.")
                                break
                        except json.JSONDecodeError: # pragma: no cover
                            logger.warning(f"Ollama Pull '{model_name_tag}': Non-JSON line in stream: {line_str}")
            return final_status_message # Return the last message, hopefully "success"
        except Exception as e: # pragma: no cover
            logger.error(f"Ollama API error during pull for '{model_name_tag}': {e}", exc_info=True)
            return {"error": f"Exception during pull: {e}", "status": "ERROR"}


    async def shutdown(self): # pragma: no cover
        """Closes the aiohttp ClientSession if it was created."""
        if self.http_client_session and not self.http_client_session.closed:
            await self.http_client_session.close()
            logger.info("OllamaHandler: Closed aiohttp ClientSession.")
        self.http_client_session = None

    def __del__(self): # pragma: no cover
        # Ensure session is closed if handler is garbage collected, though explicit shutdown is better.
        if self.http_client_session and not self.http_client_session.closed:
            try: # __del__ should not raise exceptions
                # For an async close in a sync context, this is tricky.
                # Better to rely on explicit async shutdown().
                if hasattr(asyncio, 'get_running_loop'): # Check if a loop is available
                    try:
                        loop = asyncio.get_running_loop()
                        if loop.is_running():
                            asyncio.ensure_future(self.http_client_session.close(), loop=loop)
                        else: # Loop not running, cannot await. This is a problem.
                             logger.warning("OllamaHandler.__del__: Event loop not running, cannot close session asynchronously. Potential resource leak.")
                    except RuntimeError: # No loop
                         logger.warning("OllamaHandler.__del__: No event loop, cannot close session asynchronously. Potential resource leak.")
                else: # Older Python
                     logger.warning("OllamaHandler.__del__: Cannot reliably close session from __del__ in this Python version.")

            except Exception as e:
                logger.error(f"OllamaHandler: Error closing session in __del__: {e}")
