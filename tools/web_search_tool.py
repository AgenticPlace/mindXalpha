# mindx/tools/web_search_tool.py
"""
WebSearchTool for MindX agents.
Utilizes Google Custom Search API to perform web searches.
"""
import os
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Coroutine

# httpx is an async-capable HTTP client
import httpx # Requires: pip install httpx

# from .base import BaseTool # Conceptual: if BaseTool exists
from mindx.utils.config import Config
from mindx.utils.logging_config import get_logger
# from mindx.llm.llm_factory import LLMHandler # Only if LLM is used for mock results

logger = get_logger(__name__)

class WebSearchTool: # Replace with `class WebSearchTool(BaseTool):` if BaseTool is defined
    """
    Tool for searching the web using Google Custom Search JSON API.
    Requires GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID to be configured.
    """
    
    def __init__(self, 
                 config: Optional[Config] = None,
                 api_key_override: Optional[str] = None, # Allow direct override for testing/specifics
                 search_engine_id_override: Optional[str] = None,
                 bdi_agent_ref: Optional[Any] = None): # For BaseTool compatibility
        """
        Initialize the web search tool.
        
        Args:
            config: Optional Config instance.
            api_key_override: Directly provide API key, bypassing config.
            search_engine_id_override: Directly provide Search Engine ID, bypassing config.
            bdi_agent_ref: Optional reference to owning BDI agent.
        """
        # super().__init__(config, bdi_agent_ref=bdi_agent_ref) # If inheriting BaseTool
        self.config = config or Config()
        self.agent_id_for_logging = getattr(bdi_agent_ref, 'agent_id', 'WebSearchTool') # Get agent_id if tool is owned

        self.api_key: Optional[str] = api_key_override or \
                                     self.config.get("tools.web_search.google_api_key", 
                                                     os.environ.get("GOOGLE_SEARCH_API_KEY"))
        self.search_engine_id: Optional[str] = search_engine_id_override or \
                                               self.config.get("tools.web_search.google_search_engine_id", 
                                                               os.environ.get("GOOGLE_SEARCH_ENGINE_ID"))
        
        # Initialize httpx.AsyncClient once
        self.http_client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=self.config.get("tools.web_search.timeout_seconds", 20.0), # Configurable timeout
            follow_redirects=True
        )
        
        if not self.api_key or not self.search_engine_id: # pragma: no cover
            logger.warning(
                f"{self.agent_id_for_logging}: Google Search API key or Search Engine ID not configured. "
                "Web search will use mock results or fail if mocks are disabled."
            )
        logger.info(f"{self.agent_id_for_logging}: WebSearchTool initialized. API Key Configured: {bool(self.api_key)}, SE ID Configured: {bool(self.search_engine_id)}")

    async def execute(self, query: str, num_results: int = 5) -> str: # Returns formatted string or error string
        """
        Executes a web search using Google Custom Search API.
        
        Args:
            query: The search query string.
            num_results: Desired number of search results (max 10 due to API limits).
            
        Returns:
            A formatted string containing search results, or an error message, or mock results.
        """
        tool_name = "WebSearchTool"
        logger.info(f"{self.agent_id_for_logging}: Executing web search. Query: '{query}', NumResults: {num_results}")
        
        if not query or not isinstance(query, str): # pragma: no cover
            logger.warning(f"{tool_name}: Invalid query provided.")
            return "Error: Invalid or empty query provided for web search."

        if not self.api_key or not self.search_engine_id: # pragma: no cover
            logger.warning(f"{tool_name}: API key or Search Engine ID missing. Generating mock results.")
            return self._generate_mock_results(query, num_results)
        
        # Google Custom Search API allows max 10 results per request
        actual_num_results = min(max(1, num_results), 10) 
        
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": actual_num_results
        }
        
        try:
            response = await self.http_client.get(search_url, params=params)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
            
            results_json = response.json()
            return self._format_google_search_results(results_json, query)
        
        except httpx.TimeoutException as e_timeout: # pragma: no cover
            logger.error(f"{tool_name}: Web search request timed out for query '{query}': {e_timeout}")
            return f"Error: Web search request timed out. ({e_timeout})"
        except httpx.HTTPStatusError as e_http: # pragma: no cover
            error_detail = "Unknown API error."
            try: error_detail = e_http.response.json().get("error", {}).get("message", str(e_http))
            except: pass # Keep original if JSON parsing fails
            logger.error(f"{tool_name}: HTTP error during web search for '{query}': {e_http.response.status_code} - {error_detail}", exc_info=True)
            return f"Error: Web search API request failed with status {e_http.response.status_code}. Detail: {error_detail}"
        except Exception as e_general: # pragma: no cover
            logger.error(f"{tool_name}: Unexpected error during web search for '{query}': {e_general}", exc_info=True)
            return f"Error: Unexpected error during web search - {type(e_general).__name__}: {e_general}"

    def _format_google_search_results(self, results_json: Dict[str, Any], query: str) -> str: # pragma: no cover
        """Formats results from Google Custom Search API into a readable string."""
        original_query = query # Use the passed query for consistency
        search_info = results_json.get("searchInformation", {})
        formatted_time = search_info.get("formattedSearchTime", "N/A")
        total_results = search_info.get("formattedTotalResults", "N/A")

        output_parts = [f"Search Results for: \"{original_query}\" (Time: {formatted_time}s, Approx. Total: {total_results})\n"]
        
        items = results_json.get("items")
        if not items:
            output_parts.append("\nNo results found for this query.")
            return "".join(output_parts)
        
        for i, item in enumerate(items, 1):
            title = item.get("title", "No Title")
            link = item.get("link", "No Link")
            snippet = item.get("snippet", "No Snippet").replace("\n", " ").strip() # Clean up snippet
            
            output_parts.append(f"\nResult {i}:")
            output_parts.append(f"  Title: {title}")
            output_parts.append(f"  Link: {link}")
            output_parts.append(f"  Snippet: {snippet}")
        
        return "\n".join(output_parts)
    
    def _generate_mock_results(self, query: str, num_results: int) -> str: # pragma: no cover
        """Generates plausible mock search results if API is unavailable."""
        logger.info(f"{self.agent_id_for_logging}: Generating mock search results for query '{query}'.")
        # This could be enhanced to use a simple LLM call if an LLMHandler is passed to __init__
        # For now, static mock:
        output_parts = [f"Mock Search Results for: \"{query}\" (API Key/ID Not Configured)\n"]
        for i in range(1, num_results + 1):
            output_parts.append(f"\nResult {i}:")
            output_parts.append(f"  Title: Mock Result {i} - {query[:50]}")
            output_parts.append(f"  Link: https://example.com/mocksearch?q={query.replace(' ','+')}&result={i}")
            output_parts.append(f"  Snippet: This is mock search result number {i} for the query '{query}'. It demonstrates the expected format of real search results, which would contain relevant information from the web.")
        return "\n".join(output_parts)

    async def shutdown(self): # pragma: no cover
        """Closes the HTTP client."""
        if self.http_client and not self.http_client.is_closed:
            await self.http_client.aclose()
            logger.info(f"{self.agent_id_for_logging}: WebSearchTool's HTTP client closed.")

# Example usage (conceptual, typically called by an agent)
async def _web_search_tool_example(): # pragma: no cover
    config = Config() # Ensure .env is loaded for API keys for this example
    
    # To test with real API, ensure GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID are in .env
    # Otherwise, it will use mock results.
    search_tool = WebSearchTool(config=config)
    
    query1 = "latest advancements in AI self-improvement"
    print(f"\n--- Searching for: '{query1}' ---")
    results1 = await search_tool.execute(query=query1, num_results=3)
    print(results1)

    query2 = "python pathlib tutorial"
    print(f"\n--- Searching for: '{query2}' ---")
    results2 = await search_tool.execute(query=query2, num_results=2)
    print(results2)

    await search_tool.shutdown() # Important to close the client

# if __name__ == "__main__": # pragma: no cover
#     # This setup allows running the example if this file is executed directly
#     project_r = Path(__file__).resolve().parent.parent.parent
#     env_p = project_r / ".env"
#     if env_p.exists(): from dotenv import load_dotenv; load_dotenv(dotenv_path=env_p, override=True)
#     else: print(f"WebSearchTool Example: .env not found at {env_p}. Mock results likely.", file=sys.stderr)
#     logging.basicConfig(level=logging.DEBUG) # More verbose for tool example
    
#     asyncio.run(_web_search_tool_example())
