# Note Taking Tool (`note_taking_tool.py`)

## Introduction

The `NoteTakingTool` is a utility component for MindX agents (Augmentic Project). It provides a simple mechanism for agents to create, read, update, delete, and list textual notes. Notes are stored as plain text files (`.txt`) in a configurable directory structure, organized by "topic" which typically maps to the filename.

This tool is designed to be used by agents like the `BDIAgent` or `StrategicEvolutionAgent` to persist information, scratchpad thoughts, store intermediate results, or maintain logs related to their tasks.

## Explanation

### Core Features

1.  **Initialization (`__init__`):**
    *   Accepts an optional `notes_dir` (absolute `Path`) argument. If not provided, it defaults to a path constructed from `PROJECT_ROOT / config.get("tools.note_taking.default_notes_dir_relative_to_project", "data/agent_notes/general_notes")`.
    *   Ensures the notes directory exists upon initialization.
    *   Can conceptually accept a `config` instance and `bdi_agent_ref` if it were to inherit from a common `BaseTool` class (as per the `BaseTool` stub provided earlier).

2.  **Main Execution Method (`async execute`):**
    *   This is the primary interface for the tool. It takes:
        -   `action`: A string specifying the operation: `"add"`, `"update"`, `"read"`, `"delete"`, `"list"`.
        -   `topic`: A string representing the subject or identifier of the note. This is used to generate a sanitized filename.
        -   `content`: An optional string containing the note's content (required for "add" and "update").
        -   `target_filename`: An optional string. If provided, this is used as the filename (potentially including relative subdirectories within the main `notes_dir_abs`). This allows for more structured note organization if needed. If `None`, the filename is derived from `topic`.
    *   File operations (`write_text`, `read_text`, `unlink`) are executed asynchronously using `loop.run_in_executor(None, ...)` to prevent blocking the agent's event loop.

3.  **Filename Sanitization (`_sanitize_filename`):**
    *   Converts a `topic` string into a safe filename by replacing invalid characters with underscores, reducing multiple underscores, stripping leading/trailing problematic characters, and limiting length.
    *   Provides a fallback filename (UUID-based) if sanitization results in an empty string.

4.  **Path Construction (`_get_note_path`):**
    *   Uses the sanitized topic to construct the full `Path` to the note file within `self.notes_dir_abs`.

5.  **Action Handlers (Private Methods):**
    *   The `execute` method dispatches to private helper methods for each action (though in this refactored version, the logic is directly within `execute` after path resolution).
    *   **`add`**: Creates a new note file. Fails if a file with the same resolved path already exists (to prevent accidental overwrite; `update` should be used). Requires `content`.
    *   **`update`**: Overwrites an existing note file with new `content`. Fails if the file does not exist. Requires `content`.
    *   **`read`**: Reads and returns the content of an existing note file. Fails if the file does not exist.
    *   **`delete`**: Deletes an existing note file. Fails if the file does not exist.
    *   **`list`**: Lists the names (topics/filenames without `.txt`) of all notes found in the *immediate* `self.notes_dir_abs`. It does not currently list notes in subdirectories recursively if `target_filename` was used to create them.

### Technical Details

-   **Path Handling:** Uses `pathlib.Path` for robust path operations. The main notes directory is resolved to an absolute path. When `target_filename` is used, it's treated as relative to `notes_dir_abs`, and a security check ensures it doesn't try to write outside this base directory.
-   **Asynchronous File I/O:** Employs `asyncio.get_running_loop().run_in_executor(None, ...)` for file operations, making the tool non-blocking for asynchronous agents.
-   **Error Handling:** Includes `try-except` blocks for file operations and returns informative error messages as strings.
-   **Configuration:** The base notes directory can be configured via `Config`.

## Usage

The `NoteTakingTool` would be instantiated and used by an agent:

```python
# Conceptual usage within an agent (e.g., BDIAgent or StrategicEvolutionAgent)
# from mindx.tools.note_taking_tool import NoteTakingTool
# from mindx.utils.config import PROJECT_ROOT # If agent needs to construct a specific notes_dir

# class MyAgent:
#     def __init__(self, agent_id: str, config: Config, ...):
#         self.agent_id = agent_id
#         # Example: agent-specific notes directory
#         agent_notes_dir = PROJECT_ROOT / "data" / "agent_notes" / self.agent_id 
#         self.note_tool = NoteTakingTool(notes_dir=agent_notes_dir, config=config) 
#         # ...

#     async def perform_research_and_document(self, research_topic: str, query: str):
#         # ... (agent performs web search or other info gathering) ...
#         search_results = f"Found information about {query}..." 
        
#         # Add initial notes
#         add_status = await self.note_tool.execute(
#             action="add", 
#             topic=research_topic, 
#             content=f"Initial research findings for {research_topic}:\n{search_results}"
#         )
#         print(add_status)

#         # Later, update the note
#         updated_content = await self.note_tool.execute(action="read", topic=research_topic)
#         updated_content += "\n\nFurther analysis: Discovered X and Y."
#         update_status = await self.note_tool.execute(action="update", topic=research_topic, content=updated_content)
#         print(update_status)

#         # List notes
#         all_notes = await self.note_tool.execute(action="list", topic="") # Topic is ignored for list
#         print(all_notes)

#         # Using target_filename for structured notes
#         report_structure_status = await self.note_tool.execute(
#             action="add",
#             topic="ignored_when_target_filename_is_set", # Topic can be for logging but filename takes precedence
#             target_filename=f"project_alpha/reports/{research_topic}_section1.txt",
#             content="This is section 1 of the report."
#         )
#         print(report_structure_status)
