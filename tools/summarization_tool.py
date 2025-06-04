# mindx/tools/summarization_tool.py
"""
SummarizationTool for MindX agents.
Utilizes an LLM to summarize provided text based on context and length constraints.
"""
import logging
import asyncio
from typing import Dict, Any, Optional

# from .base import BaseTool # Conceptual: if BaseTool exists
from mindx.utils.config import Config
from mindx.utils.logging_config import get_logger
from mindx.llm.llm_factory import create_llm_handler, LLMHandler # To get its own LLM

logger = get_logger(__name__)

class SummarizationTool: # Replace with `class SummarizationTool(BaseTool):` if BaseTool is defined
    """
    Tool for summarizing text using a Large Language Model.
    It can take into account topic context and desired summary length.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None, 
                 llm_handler: Optional[LLMHandler] = None, # Can be provided, or will create its own
                 bdi_agent_ref: Optional[Any] = None): # For BaseTool compatibility
        """
        Initialize the summarization tool.
        
        Args:
            config: Optional Config instance.
            llm_handler: Optional LLMHandler instance. If None, one will be created
                         based on tool-specific or default LLM configurations.
            bdi_agent_ref: Optional reference to the owning BDI agent.
        """
        # super().__init__(config, llm_handler, bdi_agent_ref=bdi_agent_ref) # If inheriting BaseTool
        self.config = config or Config()
        
        if llm_handler:
            self.llm_handler = llm_handler
        else:
            # Configure LLM specific to this tool if needed, or use defaults
            tool_llm_provider = self.config.get("tools.summarization.llm.provider", self.config.get("llm.default_provider"))
            tool_llm_model = self.config.get("tools.summarization.llm.model", self.config.get(f"llm.{tool_llm_provider}.default_model_for_summarization", self.config.get(f"llm.{tool_llm_provider}.default_model")))
            self.llm_handler = create_llm_handler(tool_llm_provider, tool_llm_model)
        
        logger.info(f"SummarizationTool initialized. Using LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default'}")
    
    async def execute(self, text_to_summarize: str, 
                      topic_context: Optional[str] = None, 
                      max_summary_words: int = 150,
                      output_format: str = "paragraph", # "paragraph" or "bullet_points"
                      temperature: Optional[float] = None,
                      custom_instructions: Optional[str] = None
                      ) -> str: # Returns the summary string or an error message
        """
        Summarizes the provided text using an LLM.
        
        Args:
            text_to_summarize: The text content to be summarized.
            topic_context: Optional context about the topic of the text.
            max_summary_words: Approximate maximum number of words for the summary.
            output_format: Desired format ("paragraph" or "bullet_points").
            temperature: Optional temperature for LLM generation (uses tool default if None).
            custom_instructions: Optional additional instructions for the LLM.
            
        Returns:
            The generated summary as a string, or an error message if summarization fails.
        """
        tool_name = "SummarizationTool"
        logger.info(f"{tool_name}: Summarizing text. Topic: '{topic_context or 'N/A'}', MaxWords: {max_summary_words}, Format: {output_format}")
        
        if not text_to_summarize or not text_to_summarize.strip():
            logger.warning(f"{tool_name}: No text provided for summarization.")
            return "Error: No text provided for summarization."
        
        # Truncate very long input text to avoid exceeding LLM context limits (crude truncation)
        # A better approach would be iterative summarization or map-reduce for very long texts.
        max_input_chars = self.config.get("tools.summarization.max_input_chars", 30000)
        if len(text_to_summarize) > max_input_chars: # pragma: no cover
            logger.warning(f"{tool_name}: Input text length ({len(text_to_summarize)} chars) exceeds max ({max_input_chars}). Truncating.")
            text_to_summarize = text_to_summarize[:max_input_chars//2] + \
                                f"\n... (TEXT TRUNCATED DUE TO LENGTH) ...\n" + \
                                text_to_summarize[-(max_input_chars//2):]
        
        prompt = self._build_summarization_prompt(
            text_to_summarize, topic_context, max_summary_words, output_format, custom_instructions
        )
        
        eff_temperature = temperature if temperature is not None \
            else self.config.get("tools.summarization.llm.temperature", 0.2) # Low temp for factual summary
        max_tokens_for_summary = max_summary_words * 3 # Rough estimate: words to tokens (generous)
                         
        try:
            logger.debug(f"{tool_name}: Sending prompt to LLM (first 200 chars): {prompt[:200]}...")
            summary_result = await self.llm_handler.generate_text(
                prompt=prompt,
                max_tokens=max_tokens_for_summary,
                temperature=eff_temperature
            )
            
            if summary_result and not summary_result.startswith("Error:"):
                logger.info(f"{tool_name}: Summary generated successfully for topic '{topic_context or 'N/A'}'. Length: {len(summary_result.split())} words (approx).")
                return summary_result.strip()
            else: # pragma: no cover
                logger.error(f"{tool_name}: LLM generation for summary failed or returned error: {summary_result}")
                return f"Error: LLM failed to generate summary - {summary_result}"

        except Exception as e: # pragma: no cover
            logger.error(f"{tool_name}: Exception during summarization LLM call: {e}", exc_info=True)
            return f"Error: Exception during summarization - {type(e).__name__}: {e}"

    def _build_summarization_prompt(self, text: str, topic: Optional[str], 
                                   max_words: int, format_type: str,
                                   custom_instr: Optional[str]) -> str: # pragma: no cover
        """Constructs the prompt for the summarization LLM."""
        
        prompt_lines = [
            "You are an expert summarization AI. Please summarize the following text accurately and concisely."
        ]
        if topic:
            prompt_lines.append(f"The text is about: {topic}.")
        
        prompt_lines.append(f"The summary should be approximately {max_words} words or less.")
        
        if format_type.lower() == "bullet_points":
            prompt_lines.append("Present the summary as a series of key bullet points, each starting with '-' or '*'.")
        else: # Default to paragraph
            prompt_lines.append("Present the summary as a coherent paragraph or a few short paragraphs.")
            
        prompt_lines.extend([
            "Focus on extracting the most critical information and main ideas.",
            "Maintain a neutral and objective tone.",
            "Ensure factual accuracy with respect to the original text."
        ])

        if custom_instr: # pragma: no cover
            prompt_lines.append(f"\nAdditional specific instructions for this summary: {custom_instr}")
            
        prompt_lines.append("\nText to Summarize:\n---BEGIN TEXT---")
        prompt_lines.append(text)
        prompt_lines.append("---END TEXT---\n\nConcise Summary:")
        
        return "\n".join(prompt_lines)

    async def shutdown(self): # pragma: no cover
        """Perform any cleanup for the SummarizationTool (e.g., if it managed persistent connections)."""
        logger.info(f"SummarizationTool ({self.llm_handler.provider_name}/{self.llm_handler.model_name}) shutting down.")
        # If LLMHandler had its own shutdown, call it here if this tool "owns" the handler.
        # For now, assume LLMHandlers are managed globally by ModelRegistry or per-agent.

# Example usage (conceptual, typically called by an agent)
async def _summarization_tool_example(): # pragma: no cover
    # Config() will load .env if present
    config = Config()
    
    # Get a default LLM handler (or configure a specific one for summarization)
    # default_llm = create_llm_handler() 
    
    summarizer = SummarizationTool(config=config) # Uses its own configured LLM
    
    sample_text = (
        "The Augmentic Project's MindX system is an innovative artificial intelligence platform "
        "designed for autonomous self-improvement. It leverages a multi-agent architecture where specialized "
        "agents collaborate to analyze, modify, and evaluate the system's own codebase. Key components include "
        "a CoordinatorAgent for high-level orchestration, a SelfImprovementAgent for tactical code modifications, "
        "and various monitoring agents to provide data-driven insights. The goal is to create a system that "
        "can evolve its capabilities over time, learning from its operations and enhancing its performance "
        "and robustness with minimal human intervention. This approach draws inspiration from concepts in "
        "Darwinian evolution and theoretical self-referential systems, adapted for practical software engineering challenges."
        "Further research focuses on advanced planning, safer evaluation, and expanding the scope of "
        "self-modifiable system artifacts beyond just Python code."
    )
    
    print("--- Paragraph Summary (Default) ---")
    summary_p = await summarizer.execute(text_to_summarize=sample_text, topic_context="MindX System Overview", max_summary_words=50)
    print(summary_p)
    
    print("\n--- Bullet Point Summary ---")
    summary_b = await summarizer.execute(text_to_summarize=sample_text, topic_context="MindX Key Features", max_summary_words=70, output_format="bullet_points")
    print(summary_b)

    await summarizer.shutdown()

# if __name__ == "__main__": # pragma: no cover
#     # This setup allows running the example if this file is executed directly
#     # It ensures .env is loaded for Config and logging is setup.
#     project_r = Path(__file__).resolve().parent.parent.parent
#     env_p = project_r / ".env"
#     if env_p.exists(): from dotenv import load_dotenv; load_dotenv(dotenv_path=env_p, override=True)
#     else: print(f"SummarizationTool Example: .env not found at {env_p}", file=sys.stderr)
#     logging.basicConfig(level=logging.INFO) # Basic logging for standalone run
    
#     asyncio.run(_summarization_tool_example())
