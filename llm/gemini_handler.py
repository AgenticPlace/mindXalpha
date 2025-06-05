# mindx/llm/gemini_handler.py
"""
LLM Handler for Google Gemini models.
Requires GOOGLE_APPLICATION_CREDENTIALS or GEMINI_API_KEY.
"""
import logging
import asyncio
import importlib
import json # For robust parsing of JSON if needed
import re   # For robust parsing of JSON if needed
from typing import Optional, Any, Dict

from mindx.utils.logging_config import get_logger
from .llm_interface import LLMHandlerInterface

logger = get_logger(__name__)

class GeminiHandler(LLMHandlerInterface): # pragma: no cover
    """LLM Handler for Google Gemini models."""
    def __init__(self, model_name_for_api: Optional[str], api_key: Optional[str],
                 base_url: Optional[str] = None): # base_url not typically used for Gemini cloud
        super().__init__("gemini", model_name_for_api, api_key, base_url)
        self.genai_sdk = None
        self._configured_ok = False

        if not self.api_key and not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            logger.error(
                f"GeminiHandler: API key not provided for model '{self.model_name_for_api}' AND "
                "GOOGLE_APPLICATION_CREDENTIALS not set. Gemini calls will likely fail."
            )
            # Not returning immediately, as GOOGLE_APPLICATION_CREDENTIALS might work implicitly
            # if SDK is imported and finds them.

        try:
            self.genai_sdk = importlib.import_module("google.generativeai")
            if self.api_key: # Prefer explicit API key if provided
                self.genai_sdk.configure(api_key=self.api_key)
                logger.info(f"GeminiHandler: SDK configured with explicit API key for model '{self.model_name_for_api}'.")
            else: # Rely on ADC (Application Default Credentials)
                # No explicit configure() needed if ADC is set up correctly
                logger.info(f"GeminiHandler: SDK imported for model '{self.model_name_for_api}'. Will attempt to use Application Default Credentials.")
            self._configured_ok = True # Assume configured if SDK imported and API key path chosen
        except ImportError:
            logger.error("GeminiHandler: Google Generative AI SDK not found. `pip install google-generativeai` required.")
        except Exception as e: # Catch configuration errors from genai.configure
            logger.error(f"GeminiHandler: Error configuring Gemini SDK for model '{self.model_name_for_api}': {e}", exc_info=True)
            self.genai_sdk = None # Ensure it's None if config fails

    async def generate_text(self, prompt: str, model: str, 
                            max_tokens: Optional[int] = 2048, temperature: Optional[float] = 0.7,
                            json_mode: Optional[bool] = False, **kwargs: Any) -> Optional[str]:
        if not self.genai_sdk or not self._configured_ok:
            return f"Error: Gemini SDK not available or not configured for model '{model}' (handler for '{self.model_name_for_api}')."

        logger.debug(f"Gemini Call: Model='{model}', JSON={json_mode}, Temp={temperature}, MaxTok={max_tokens}, Prompt(start): {prompt[:150]}...")
        try:
            gen_config_args: Dict[str, Any] = {}
            if temperature is not None: gen_config_args["temperature"] = temperature
            if max_tokens is not None and max_tokens > 0: gen_config_args["max_output_tokens"] = max_tokens
            
            effective_prompt = prompt
            # Gemini JSON mode handling: some models support response_mime_type="application/json"
            # For others, rely on prompt engineering.
            if json_mode:
                # Check if the specific model tag indicates a version that might support mime_type (e.g., "1.5-pro")
                # This is a heuristic; official Gemini docs are the source of truth.
                if ("1.5-pro" in model.lower() or "flash" in model.lower()) and \
                   hasattr(self.genai_sdk.types.GenerationConfig(), "response_mime_type"): # Check if SDK version supports it
                    try:
                        gen_config_args["response_mime_type"] = "application/json"
                        logger.info(f"GeminiHandler: Requesting JSON response_mime_type for model '{model}'.")
                    except Exception as e_mime: # pragma: no cover
                        logger.warning(f"GeminiHandler: Attempt to set JSON mime_type for '{model}' failed ('{e_mime}'), falling back to prompt instruction for JSON.")
                        effective_prompt = prompt + "\n\nYour entire response MUST be a single, valid JSON object. Do not include any text outside this JSON structure, including markdown fences."
                else: 
                    effective_prompt = prompt + "\n\nYour entire response MUST be a single, valid JSON object. Do not include any text outside this JSON structure."
                    logger.info("GeminiHandler: Appended JSON mode instruction to prompt for model '{model}'.")
            
            # Safety settings can be passed via kwargs or configured defaults
            safety_settings = kwargs.get("safety_settings", self.config.get(f"llm.gemini.safety_settings"))
            # Default safety settings if not provided by config or kwargs
            if safety_settings is None: # pragma: no cover
                 safety_settings = [
                    {"category": hc.name, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} 
                    for hc in self.genai_sdk.types.HarmCategory if hc.name not in ["HARM_CATEGORY_UNSPECIFIED"] # Exclude unspecified
                 ]


            model_instance = self.genai_sdk.GenerativeModel(model_name=model, safety_settings=safety_settings)
            
            # Gemini API typically uses a list of "Content" objects (prompt parts)
            # For simple text generation, a single string prompt is often fine.
            # If prompt is already structured for chat (list of dicts), adapt.
            # For now, assume prompt is a single string for generate_content_async.
            
            response = await model_instance.generate_content_async(
                effective_prompt,
                generation_config=self.genai_sdk.types.GenerationConfig(**gen_config_args)
            )
            
            final_response = ""
            if response.parts: # Process parts to construct full text
                final_response = "".join(part.text for part in response.parts if hasattr(part, "text")).strip()
            
            # Check for blocking due to safety filters or other reasons
            if not final_response and response.prompt_feedback and response.prompt_feedback.block_reason: # pragma: no cover
                block_reason_name = getattr(response.prompt_feedback.block_reason, 'name', str(response.prompt_feedback.block_reason))
                block_reason_msg = f"Blocked by API. Reason: {block_reason_name}"
                if response.prompt_feedback.safety_ratings: 
                    block_reason_msg += f" SafetyRatings: {[(sr.category.name, sr.probability.name) for sr in response.prompt_feedback.safety_ratings]}"
                logger.warning(f"GeminiHandler response for '{model}' was empty due to blocking: {block_reason_msg}")
                return f"Error: Gemini response blocked - {block_reason_msg}"

            # Post-check JSON format if json_mode was requested
            if json_mode and final_response and \
               not (final_response.startswith('{') and final_response.endswith('}')) and \
               not (final_response.startswith('[') and final_response.endswith(']')): # pragma: no cover
                logger.warning(f"GeminiHandler: Model '{model}' requested JSON but output seems non-JSON. Snippet: {final_response[:100]}")

            logger.debug(f"GeminiHandler response for '{model}' (first 100 chars): {final_response[:100]}")
            return final_response
        except Exception as e: # pragma: no cover
            logger.error(f"GeminiHandler: API call failed for model '{model}': {e}", exc_info=True)
            err_msg = str(e)
            if "API key" in err_msg.lower() or "permission" in err_msg.lower() or "authentication" in err_msg.lower():
                return f"Error: Gemini API key/permission/authentication issue for '{model}'. Check configuration and credentials."
            return f"Error: Gemini call failed for '{model}' - {type(e).__name__}: {err_msg}"

    async def shutdown(self): # pragma: no cover
        """Perform any cleanup for the GeminiHandler."""
        logger.info(f"GeminiHandler for model '{self.model_name_for_api}' shutting down. (No specific async cleanup action implemented).")
        # The google.generativeai SDK client does not typically require an explicit close().
