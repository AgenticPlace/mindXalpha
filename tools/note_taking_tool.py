# mindx/tools/note_taking_tool.py
"""
NoteTakingTool for MindX agents.
Allows agents to create, read, update, and delete textual notes organized by topic.
"""
import os
import logging
import json # Not used in current version, but could be for structured notes
import asyncio # For async file operations if needed, current is sync
from pathlib import Path
from typing import Dict, Any, List, Optional

# from .base import BaseTool # Conceptual: if BaseTool exists
from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

class NoteTakingTool: # Replace with `class NoteTakingTool(BaseTool):` if BaseTool is defined
    """Tool for taking and managing textual notes, stored as .txt files."""
    
    def __init__(self, 
                 notes_dir: Optional[Path] = None, 
                 config: Optional[Config] = None,
                 bdi_agent_ref: Optional[Any] = None): # For BaseTool compatibility
        """
        Initialize the note taking tool.
        
        Args:
            notes_dir: Absolute Path to the directory to store notes. 
                       If None, defaults to PROJECT_ROOT / "data" / "agent_notes" / "general_notes".
            config: Optional Config instance.
            bdi_agent_ref: Optional reference to the owning BDI agent (for BaseTool).
        """
        # super().__init__(config, bdi_agent_ref=bdi_agent_ref) # If inheriting BaseTool
        self.config = config or Config()
        
        if notes_dir:
            self.notes_dir_abs = notes_dir
        else:
            default_rel_path = self.config.get("tools.note_taking.default_notes_dir_relative_to_project", "data/agent_notes/general_notes")
            self.notes_dir_abs = PROJECT_ROOT / default_rel_path
        
        try:
            self.notes_dir_abs.mkdir(parents=True, exist_ok=True)
            logger.info(f"NoteTakingTool initialized. Notes directory: {self.notes_dir_abs}")
        except Exception as e: # pragma: no cover
            logger.error(f"NoteTakingTool: Failed to create notes directory {self.notes_dir_abs}: {e}", exc_info=True)
            # Potentially raise an error or operate in a degraded mode if dir is essential
            raise # Re-raise for critical failure during init


    def _sanitize_filename(self, topic: str) -> str: # pragma: no cover
        """Sanitizes a topic string into a safe filename."""
        if not topic or not isinstance(topic, str): # pragma: no cover
            logger.error("Invalid topic provided for sanitization.")
            raise ValueError("Topic must be a non-empty string.")
            
        # Replace common problematic characters with underscores
        # Allow alphanumeric, underscore, hyphen, dot.
        sanitized = re.sub(r'[^\w\-\.]', '_', topic)
        # Reduce multiple underscores to a single one
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores/dots
        sanitized = sanitized.strip('_.')
        
        if not sanitized: # If all chars were invalid
            sanitized = f"note_{str(uuid.uuid4())[:8]}" # Fallback to a UUID based name
            logger.warning(f"Topic '{topic}' resulted in empty sanitized filename, using '{sanitized}'.")

        # Limit length (OS limits are usually around 255 bytes/chars)
        max_len = 100 
        if len(sanitized) > max_len:
            sanitized = sanitized[:max_len]
            logger.debug(f"Sanitized filename for topic '{topic}' truncated to '{sanitized}'.")
        return sanitized

    def _get_note_path(self, topic: str) -> Path: # pragma: no cover
        """Constructs the full path for a note file based on the topic."""
        filename_base = self._sanitize_filename(topic)
        return self.notes_dir_abs / f"{filename_base}.txt"

    async def execute(self, action: str, topic: str, content: Optional[str] = None, 
                      target_filename: Optional[str] = None) -> str: # pragma: no cover
        """
        Executes a note-taking action. All file operations are run in a thread pool executor.
        """
        logger.info(f"NoteTakingTool: Action='{action}', Topic='{topic}', TargetFilename='{target_filename}' Content Snippet='{(content or '')[:50]}...'")
        
        file_path: Path
        if target_filename: # If a specific filename (potentially with subdirs relative to notes_dir) is given
            # Ensure target_filename is relative and safe
            clean_target_filename = Path(*(p for p in Path(target_filename).parts if p not in ('.','..')))
            file_path = (self.notes_dir_abs / clean_target_filename).resolve()
            # Security check: ensure resolved path is still within notes_dir_abs
            if self.notes_dir_abs not in file_path.parents: # pragma: no cover
                logger.error(f"NoteTakingTool: Invalid target_filename '{target_filename}' attempts to access outside notes directory.")
                return f"Error: Invalid target_filename '{target_filename}'."
        else:
            file_path = self._get_note_path(topic) # Use topic to generate filename

        loop = asyncio.get_running_loop()

        try:
            if action.lower() == "add":
                if file_path.exists(): # pragma: no cover
                    msg = f"Note for topic/file '{topic}/{file_path.name}' already exists. Use 'update' or provide different target_filename."
                    logger.warning(f"NoteTakingTool: {msg}")
                    return msg
                if content is None: return "Error: Content is required for 'add' action."
                await loop.run_in_executor(None, self._write_file_sync, file_path, content)
                return f"Added note for topic '{topic}' (file: {file_path.name})."
            
            elif action.lower() == "update":
                if not file_path.exists(): # pragma: no cover
                    msg = f"Note for topic/file '{topic}/{file_path.name}' does not exist. Use 'add' or provide different target_filename."
                    logger.warning(f"NoteTakingTool: {msg}")
                    # Option: treat update as add if not exists? For now, require existence.
                    return msg 
                if content is None: return "Error: Content is required for 'update' action."
                await loop.run_in_executor(None, self._write_file_sync, file_path, content)
                return f"Updated note for topic '{topic}' (file: {file_path.name})."

            elif action.lower() == "read":
                if not file_path.exists(): return f"Note for topic/file '{topic}/{file_path.name}' not found."
                read_content = await loop.run_in_executor(None, file_path.read_text, "utf-8")
                logger.info(f"NoteTakingTool: Read note for topic '{topic}' (file: {file_path.name}). Length: {len(read_content)}")
                return read_content

            elif action.lower() == "delete":
                if not file_path.exists(): return f"Note for topic/file '{topic}/{file_path.name}' not found for deletion."
                await loop.run_in_executor(None, file_path.unlink)
                logger.info(f"NoteTakingTool: Deleted note for topic '{topic}' (file: {file_path.name}).")
                return f"Deleted note for topic '{topic}' (file: {file_path.name})."

            elif action.lower() == "list":
                # List files in the immediate notes_dir_abs, not recursively for simplicity
                notes_files = [f.name for f in self.notes_dir_abs.iterdir() if f.is_file() and f.suffix == ".txt"]
                if not notes_files: return "No notes found in the main notes directory."
                logger.info(f"NoteTakingTool: Listed {len(notes_files)} notes from {self.notes_dir_abs}.")
                return "Available notes (topics/filenames):\n" + "\n".join(sorted(f[:-4] for f in notes_files)) # Show stem
            
            else: # pragma: no cover
                logger.warning(f"NoteTakingTool: Unknown action '{action}'.")
                return f"Unknown action: {action}. Supported: add, update, read, delete, list."
        except Exception as e: # pragma: no cover
            logger.error(f"NoteTakingTool: Error during action '{action}' for topic '{topic}' (file: {file_path.name}): {e}", exc_info=True)
            return f"Error performing '{action}' on note '{topic}': {type(e).__name__} - {e}"

    def _write_file_sync(self, file_path: Path, content: str): # pragma: no cover
        """Synchronous helper for writing file content, to be run in executor."""
        file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
        file_path.write_text(content, encoding="utf-8")
