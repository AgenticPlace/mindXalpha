# mindx/llm/groq_handler.py
"""
LLM Handler for Groq Cloud API.
Requires GROQ_API_KEY environment variable or configuration.
"""
import os
import logging
import asyncio
import importlib
from typing import Optional, Any, Dict

from mindx.utils.logging_config import get_logger
from .llm_interface import LLMHandlerInterface # Relative import

logger = get_logger(__name__)

class GroqHandler(LLMHandlerInterface): # pragma: no cover
    """LLM Handler for Groq Cloud models."""
    def __init__(self, model_name_for_api: Optional[str], api_key: Optional[str], 
                 base_url: Optional[str] = None): # base_url typically not used for Groq cloud
        super().__init__("groq", model_name_for_api, api_key, base_url)
        self.groq_sdk = None
        self.async_client = None

        if not self.api_key:
            logger.error(f"GroqHandler: API key not provided for model '{self.model_name_for_api}'. Groq calls will fail.")
            return # Handler will be non-functional
        try:
            self.groq_sdk = importlib.import_module("groq")
            # Initialize AsyncGroq client here
            self.async_client = self.groq_sdk.AsyncGroq(api_key=self.api_key)
            logger.info(f"GroqHandler: Groq SDK imported and AsyncGroq client initialized for model '{self.model_name_for_api}'.")
        except ImportError:
            logger.error("GroqHandler: Groq SDK not found. Please install with `pip install groq` to use Groq models.")
        except Exception as e:
            logger.error(f"GroqHandler: Error initializing Groq client for model '{self.model_name_for_api}': {e}", exc_info=True)
            self.groq_sdk = None # Ensure it's None if init fails
            self.async_client = None


    async def generate_text(self, prompt: str, model: str, 
                            max_tokens: Optional[int] = 2048, temperature: Optional[float] = 0.7,
                            json_mode: Optional[bool] = False, **kwargs: Any) -> Optional[str]:
        if not self.async_client or not self.groq_sdk:
            err_msg = f"Groq SDK/client not available for model '{model}' (handler initially configured for '{self.model_name_for_api}')."
            logger.error(f"GroqHandler: {err_msg}")
            return f"Error: {err_msg}"

        logger.debug(f"Groq Call: Model='{model}', JSON={json_mode}, Temp={temperature}, MaxTok={max_tokens}, Prompt(start): {prompt[:150]}...")
        
        # Groq API uses OpenAI's message format for chat completions
        messages = [{"role": "user", "content": prompt}]
        
        # Allow system prompt via kwargs
        if "system_prompt" in kwargs and isinstance(kwargs["system_prompt"], str):
            messages.insert(0, {"role": "system", "content": kwargs["system_prompt"]})

        request_params: Dict[str, Any] = {
            "messages": messages,
            "model": model, # This is the model ID like 'llama3-8b-8192'
            "temperature": temperature if temperature is not None else 0.7,
            # Groq requires max_tokens. If None or 0, use a sensible default or raise error.
            "max_tokens": max_tokens if max_tokens is not None and max_tokens > 0 else 2048, 
        }
        if json_mode:
            request_params["response_format"] = {"type": "json_object"}
            logger.info(f"GroqHandler: Requesting JSON response_format for model '{model}'.")
        
        # Allow passing other ChatCompletionCreateParams like 'stop', 'stream', 'seed', 'tool_choice', 'tools'
        # For simplicity, only explicitly handling stop sequences here.
        if "stop_sequences" in kwargs and isinstance(kwargs["stop_sequences"], list):
            request_params["stop"] = kwargs["stop_sequences"]
        
        # Add other known Groq params if passed in kwargs
        for p_name in ["top_p", "seed", "stream"]: # stream handled differently if True
            if p_name in kwargs:
                request_params[p_name] = kwargs[p_name]
        
        if request_params.get("stream"): # Streaming needs different handling
             logger.warning("GroqHandler: Streaming requested but not fully implemented in this generate_text method. Will attempt non-streamed.")
             request_params.pop("stream", None) # For now, force non-streamed if this method is called

        try:
            chat_completion = await self.async_client.chat.completions.create(**request_params)
            
            if not chat_completion.choices or not chat_completion.choices[0].message: # pragma: no cover
                logger.warning(f"GroqHandler: Response for '{model}' from Groq had no choices or message content.")
                return "Error: Groq response was empty or malformed."

            final_response = chat_completion.choices[0].message.content
            if final_response is None: final_response = "" # Ensure string
            final_response = final_response.strip()
            
            if json_mode and final_response and not (final_response.startswith('{') and final_response.endswith('}')) and not (final_response.startswith('[') and final_response.endswith(']')): # pragma: no cover
                logger.warning(f"GroqHandler: Model '{model}' requested JSON but output seems non-JSON. Snippet: {final_response[:100]}")

            logger.debug(f"Groq response for '{model}' (first 100 chars): {final_response[:100]}")
            return final_response
        except Exception as e: # pragma: no cover
            logger.error(f"Groq API call failed for model '{model}': {e}", exc_info=True)
            # Groq SDK might raise specific exceptions, e.g., groq.APIError
            if hasattr(e, 'message') and ("API key" in str(e.message).lower() or "authentication" in str(e.message).lower()):
                return f"Error: Groq API key/authentication issue for '{model}'. Check configuration."
            return f"Error: Groq call failed for '{model}' - {type(e).__name__}: {e}"
