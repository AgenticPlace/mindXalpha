# Web Search Tool (`mindX/tools/web_search_tool.py`)

## Introduction

The `WebSearchTool` is a utility component for MindX agents (Augmentic Project) that provides the capability to perform web searches using the Google Custom Search JSON API. This enables agents to gather external information from the internet as part of their reasoning or task execution processes. If API keys are not configured, it falls back to generating plausible mock search results.

## Explanation

### Core Features

1.  **Initialization (`__init__`):**
    *   Accepts an optional `Config` instance.
    *   Retrieves `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` from the `Config` object (which can load them from environment variables like `MINDX_TOOLS__WEB_SEARCH__GOOGLE_API_KEY` or directly named environment variables as fallbacks). Direct overrides via constructor arguments are also possible for testing or specific instantiation.
    *   Initializes an `httpx.AsyncClient` instance for making asynchronous HTTP requests. This client is reused for multiple searches by the same tool instance and configured with a timeout.

2.  **Main Execution Method (`async execute`):**
    *   This is the primary interface for the tool. It takes:
        -   `query`: The search query string.
        -   `num_results` (Default: 5): The desired number of search results to return. The Google API limits this to a maximum of 10 per request.
    *   **API Key Check:** If the API key or search engine ID is not configured, it logs a warning and calls `_generate_mock_results()`.
    *   **API Request:** If configured, it constructs the URL and parameters for the Google Custom Search API and makes an asynchronous GET request using `self.http_client`.
    *   **Error Handling:**
        -   Uses `response.raise_for_status()` to check for HTTP errors (4xx, 5xx).
        -   Catches `httpx.TimeoutException`, `httpx.HTTPStatusError`, and other general exceptions.
        -   In case of any error, it logs the issue and typically returns a formatted error message string or could fall back to mock results if desired (current implementation returns error strings).
    *   **Result Parsing & Formatting:** If the API call is successful, it parses the JSON response and uses `_format_google_search_results()` to create a human-readable string summary of the results.

3.  **Result Formatting (`_format_google_search_results`):**
    *   Takes the raw JSON dictionary from the Google API.
    *   Extracts the original query, search time, total results estimate, and for each item: `title`, `link`, and `snippet`.
    *   Formats this information into a structured, multi-line string.

4.  **Mock Results (`_generate_mock_results`):**
    *   Provides a fallback when API keys are missing.
    *   Generates a predefined number of plausible-looking (but entirely fictional) search result entries based on the query. This allows agents using the tool to function even without live web access during development or testing.

5.  **Resource Management (`async shutdown`):**
    *   Provides an `async shutdown()` method to properly close the `httpx.AsyncClient`. This should be called when the tool instance is no longer needed if it's long-lived (e.g., when the owning agent shuts down).

### Technical Details

-   **Dependencies:** Uses the `httpx` library (`pip install httpx`) for asynchronous HTTP requests.
-   **API Used:** Google Custom Search JSON API. Requires setting up a Custom Search Engine (CSE) in the Google Cloud Console or Programmable Search Engine control panel to get a "Search engine ID" (cx) and an API key.
-   **Configuration:** API keys and Search Engine ID are expected to be configured via the `Config` object (e.g., from `.env` file: `MINDX_TOOLS__WEB_SEARCH__GOOGLE_API_KEY`, `MINDX_TOOLS__WEB_SEARCH__GOOGLE_SEARCH_ENGINE_ID`, or older `GOOGLE_SEARCH_API_KEY`).
-   **Asynchronous:** The `execute` method is `async` due to the network request.

## Usage

The `WebSearchTool` would be instantiated and used by an agent that needs to perform web lookups.

```python
# Conceptual usage within an agent (e.g., BDIAgent)

# from mindx.tools.web_search_tool import WebSearchTool
# from mindx.utils.config import Config # Assuming global config

# class MyResearchBDIAgent:
#     def __init__(self, config: Config, ...):
#         self.config = config
#         self.web_search_tool = WebSearchTool(config=self.config)
#         # ...

#     async def _execute_search_action(self, query: str) -> str:
#         logger.info(f"Agent: Performing web search for '{query}'...")
#         search_results_str = await self.web_search_tool.execute(query=query, num_results=3)
        
#         if search_results_str.startswith("Error:"):
#             logger.error(f"Agent: Web search failed: {search_results_str}")
#             # Update belief about search failure
#             # await self.belief_system.add_belief(f"search.last_error.{query}", search_results_str, ...)
#         else:
#             logger.info(f"Agent: Web search successful for '{query}'.")
#             # Update belief with search results
#             # await self.belief_system.add_belief(f"knowledge.search_results.{query}", search_results_str, ...)
        
#         return search_results_str

#     async def on_shutdown(self):
#         await self.web_search_tool.shutdown()

# Example call:
# research_agent = MyResearchBDIAgent(Config())
# report_section = await research_agent._execute_search_action("current AI safety research")
# await research_agent.on_shutdown() 
