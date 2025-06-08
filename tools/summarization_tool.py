# mindx/tools/summarization_tool.py
"""
SummarizationTool for mindX agents.
Utilizes an LLM to summarize provided text based on context and length constraints.
This version is integrated into the BaseTool architecture.
"""
from typing import Dict, Any, Optional, TYPE_CHECKING

from utils.config import Config
from utils.logging_config import get_logger
from llm.llm_interface import LLMHandlerInterface
from core.bdi_agent import BaseTool

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from core.bdi_agent import BDIAgent

logger = get_logger(__name__)

class SummarizationTool(BaseTool):
    """
    Tool for summarizing text using a Large Language Model.
    It inherits from BaseTool and integrates with the BDIAgent's tool system.
    It can take into account topic context, desired summary length, and output format.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None, 
                 llm_handler: Optional[LLMHandlerInterface] = None,
                 bdi_agent_ref: Optional['BDIAgent'] = None):
        """
        Initializes the summarization tool.
        
        Args:
            config: Optional Config instance.
            llm_handler: LLMHandler instance provided by the agent.
            bdi_agent_ref: Optional reference to the owning BDI agent.
        """
        super().__init__(config, llm_handler, bdi_agent_ref=bdi_agent_ref)
        
        if not self.llm_handler:
            raise ValueError("SummarizationTool requires a valid LLMHandlerInterface instance passed during initialization.")
        
        logger.info(f"SummarizationTool initialized. Using LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name_for_api or 'default'}")
    
    async def execute(self, **kwargs) -> str:
        """
        Summarizes the provided text using an LLM, driven by parameters from the agent's plan.
        
        Args:
            **kwargs: A dictionary of parameters. Expected keys include:
                - text_to_summarize (str): The text content to be summarized.
                - topic_context (Optional[str]): Context about the topic of the text.
                - max_summary_words (Optional[int]): Approx. max words for the summary. Defaults to 150.
                - output_format (Optional[str]): "paragraph" or "bullet_points". Defaults to "paragraph".
                - temperature (Optional[float]): LLM generation temperature.
                - custom_instructions (Optional[str]): Additional instructions for the LLM.
            
        Returns:
            The generated summary as a string, or an error message if summarization fails.
        """
        text_to_summarize = kwargs.get("text_to_summarize")
        topic_context = kwargs.get("topic_context")
        max_summary_words = kwargs.get("max_summary_words", 150)
        output_format = kwargs.get("output_format", "paragraph")
        temperature = kwargs.get("temperature")
        custom_instructions = kwargs.get("custom_instructions")
        
        self.logger.info(f"Executing summarization. Topic: '{topic_context or 'N/A'}', MaxWords: {max_summary_words}, Format: {output_format}")
        
        if not text_to_summarize or not isinstance(text_to_summarize, str) or not text_to_summarize.strip():
            self.logger.warning("No valid 'text_to_summarize' provided in kwargs.")
            return "Error: No text provided for summarization."
        
        # Truncate very long input text to avoid exceeding LLM context limits.
        # A more advanced implementation might use map-reduce for very long texts.
        max_input_chars = self.config.get("tools.summarization.max_input_chars", 30000)
        if len(text_to_summarize) > max_input_chars:
            self.logger.warning(f"Input text length ({len(text_to_summarize)} chars) exceeds max ({max_input_chars}). Truncating.")
            text_to_summarize = (text_to_summarize[:max_input_chars//2] + 
                                 f"\n... (TEXT TRUNCATED DUE TO LENGTH) ...\n" + 
                                 text_to_summarize[-(max_input_chars//2):])
        
        prompt = self._build_summarization_prompt(
            text_to_summarize, topic_context, max_summary_words, output_format, custom_instructions
        )
        
        eff_temperature = temperature if temperature is not None else self.config.get("tools.summarization.llm.temperature", 0.2)
        # Rough estimate: words to tokens (generous buffer)
        max_tokens_for_summary = int(max_summary_words * 2.5)
                         
        try:
            self.logger.debug(f"Sending prompt to LLM (first 200 chars): {prompt[:200]}...")
            summary_result = await self.llm_handler.generate_text(
                prompt=prompt,
                max_tokens=max_tokens_for_summary,
                temperature=eff_temperature
            )
            
            if summary_result and not summary_result.lower().startswith("error:"):
                self.logger.info(f"Summary generated successfully for topic '{topic_context or 'N/A'}'.")
                return summary_result.strip()
            else:
                self.logger.error(f"LLM generation for summary failed or returned error: {summary_result}")
                return f"Error: LLM failed to generate summary - {summary_result}"

        except Exception as e:
            self.logger.error(f"Exception during summarization LLM call: {e}", exc_info=True)
            return f"Error: Exception during summarization - {type(e).__name__}: {e}"

    def _build_summarization_prompt(self, text: str, topic: Optional[str], 
                                   max_words: int, format_type: str,
                                   custom_instr: Optional[str]) -> str:
        """Constructs the prompt for the summarization LLM."""
        
        prompt_lines = [
            "You are an expert summarization AI. Please summarize the following text accurately and concisely."
        ]
        if topic:
            prompt_lines.append(f"The text is about: {topic}.")
        
        prompt_lines.append(f"The summary should be approximately {max_words} words or less.")
        
        if format_type and format_type.lower() == "bullet_points":
            prompt_lines.append("Present the summary as a series of key bullet points, each starting with a hyphen '-'.")
        else:
            prompt_lines.append("Present the summary as a coherent paragraph.")
            
        prompt_lines.extend([
            "Focus on extracting the most critical information and main ideas.",
            "Maintain a neutral and objective tone.",
            "Ensure factual accuracy with respect to the original text."
        ])

        if custom_instr:
            prompt_lines.append(f"\nAdditional specific instructions for this summary: {custom_instr}")
            
        prompt_lines.append("\nText to Summarize:\n---BEGIN TEXT---")
        prompt_lines.append(text)
        prompt_lines.append("---END TEXT---\n\nConcise Summary:")
        
        return "\n".join(prompt_lines)
