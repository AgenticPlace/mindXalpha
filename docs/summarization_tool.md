# Summarization Tool (`summarization_tool.py`)

## Introduction

The `SummarizationTool` is a component within the MindX toolkit (Augmentic Project) designed to provide text summarization capabilities to other agents. It leverages a configured Large Language Model (LLM) to condense longer pieces of text into shorter summaries, taking into account optional topic context, desired length, and output format.

## Explanation

### Core Features

1.  **Initialization (`__init__`):**
    *   Accepts an optional `Config` instance and an optional `LLMHandler` instance.
    *   If no `LLMHandler` is provided, it creates its own by calling `create_llm_handler`. The LLM provider and model for this tool can be specifically configured in the global `Config` under `tools.summarization.llm.*`. If not specified, it falls back to the system's default LLM settings. This allows using a potentially different (e.g., more cost-effective or specialized for summarization) LLM for this tool than an agent's primary reasoning LLM.

2.  **Main Execution Method (`async execute`):**
    *   This is the primary interface for the tool. It takes the following arguments:
        -   `text_to_summarize`: The main string content that needs summarization.
        -   `topic_context` (Optional): A string providing context about the topic of the text, which can help the LLM generate a more relevant summary.
        -   `max_summary_words` (Default: 150): An approximate target for the maximum number of words in the output summary.
        -   `output_format` (Default: "paragraph"): Specifies the desired output format. Currently supports `"paragraph"` or `"bullet_points"`.
        -   `temperature` (Optional): Allows overriding the default temperature for the LLM generation specifically for this summarization task.
        -   `custom_instructions` (Optional): Allows passing additional, specific instructions to the LLM for this summarization task.
    *   **Input Handling:** Checks for empty input text. It also implements a crude truncation for extremely long input texts (based on `tools.summarization.max_input_chars` config) to prevent exceeding LLM context limits, with a warning. (Note: True summarization of very long documents often requires more sophisticated techniques like iterative summarization or map-reduce style processing).
    *   **Prompt Construction (`_build_summarization_prompt`):** Creates a detailed prompt for the LLM, incorporating the text, topic, length constraints, desired format, and any custom instructions.
    *   **LLM Interaction:** Calls the `generate_text` method of its `LLMHandler` instance.
    *   **Output:** Returns the generated summary string or an error message string if the process fails.

3.  **Prompt Building (`_build_summarization_prompt`):**
    *   This private method assembles the prompt, instructing the LLM on its role as a summarizer and providing all necessary constraints and the text itself.

4.  **Asynchronous Operation:** The `execute` method is `async` as it involves an `await`ed call to the `LLMHandler`.

### Technical Details

-   **LLM Dependency:** Relies on an `LLMHandler` (from `mindx.llm.llm_factory`) for actual LLM communication.
-   **Configuration:** Uses `mindx.utils.config.Config` for settings like:
    -   `tools.summarization.llm.provider` & `tools.summarization.llm.model`: To specify a particular LLM for summarization tasks.
    -   `tools.summarization.llm.temperature`: Default temperature for summarization.
    -   `tools.summarization.max_input_chars`: Limit for input text length before truncation.
-   **Error Handling:** Catches exceptions during LLM calls and returns error messages as strings.

## Usage

The `SummarizationTool` would be instantiated and used by an agent that needs to condense text.

```python
# Conceptual usage within an agent (e.g., BDIAgent or a research assistant agent)
# from mindx.tools.summarization_tool import SummarizationTool
# from mindx.llm.llm_factory import create_llm_handler # If providing a specific handler
# from mindx.utils.config import Config

# class MyResearchAgent:
#     def __init__(self, config: Config, ...):
#         self.config = config
#         # Option 1: Tool creates its own handler based on config
#         self.summarizer = SummarizationTool(config=self.config) 
#         # Option 2: Provide a specific handler
#         # specific_llm_for_summaries = create_llm_handler("ollama", "mistral")
#         # self.summarizer_custom = SummarizationTool(config=self.config, llm_handler=specific_llm_for_summaries)
#         # ...

#     async def process_long_document(self, document_text: str, document_topic: str):
#         logger.info(f"Agent: Summarizing document on '{document_topic}'")
        
#         summary = await self.summarizer.execute(
#             text_to_summarize=document_text,
#             topic_context=document_topic,
#             max_summary_words=200,
#             output_format="bullet_points"
#         )
        
#         if summary.startswith("Error:"):
#             logger.error(f"Agent: Failed to summarize document: {summary}")
#         else:
#             logger.info(f"Agent: Generated Summary for '{document_topic}':\n{summary}")
#             # Agent can now use this summary for further reasoning or belief updates.
#             # await self.belief_system.add_belief(f"summary.{document_topic}", summary, ...)
