# mindx/docs/documentation_agent.py
"""
Documentation Agent for MindX. (Functional Stub)

This module provides a stub for a documentation agent. In a full implementation,
it would use a tool like Sphinx to generate, manage, and serve documentation
for the MindX codebase and related artifacts. This version mocks these processes.
"""

import os
import logging
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Coroutine

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

class DocumentationAgent: # pragma: no cover
    """
    Functional Stub for a Documentation Agent for MindX.
    Manages documentation generation (conceptually using Sphinx).
    """
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DocumentationAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 config_override: Optional[Config] = None, 
                 test_mode: bool = False):
        
        if hasattr(self, '_initialized') and self._initialized and not test_mode:
            return

        self.config = config_override or Config()
        self.agent_id = self.config.get("docs.agent.agent_id", "documentation_agent_mindx")
        
        # Base directories from config, relative to PROJECT_ROOT
        self.source_dir_rel = Path(self.config.get("docs.agent.source_dir_relative_to_project", "mindx")) # e.g., "mindx" or "src"
        self.output_dir_rel = Path(self.config.get("docs.agent.output_dir_relative_to_project", "data/docs_build/html"))
        self.config_dir_rel = Path(self.config.get("docs.agent.sphinx_config_dir_relative_to_project", "docs_config/source")) # For conf.py, index.rst

        self.source_dir_abs = PROJECT_ROOT / self.source_dir_rel
        self.output_dir_abs = PROJECT_ROOT / self.output_dir_rel
        self.config_dir_abs = PROJECT_ROOT / self.config_dir_rel
        
        self.status: str = "STUB_INITIALIZED"
        self.last_build_time: Optional[float] = None
        self.last_build_status: Optional[str] = None # "SUCCESS", "FAILURE"
        self.build_history: List[Dict[str, Any]] = [] # Store last few build attempts

        # Ensure output directory exists for mock operations
        try:
            self.output_dir_abs.mkdir(parents=True, exist_ok=True)
            self.config_dir_abs.mkdir(parents=True, exist_ok=True) # For mock conf.py
        except Exception as e:
            logger.error(f"{self.agent_id}: Error creating documentation directories: {e}")


        # Mock Sphinx config for conceptual use
        self.sphinx_config_settings: Dict[str, Any] = self.config.get("docs.agent.sphinx_settings", {
            "project": "MindX AutoDocs", "author": "Augmentic Project", "version": "0.1",
            "extensions": ["sphinx.ext.autodoc", "myst_parser"], "html_theme": "alabaster"
        })

        logger.info(
            f"{self.agent_id} (Stub) initialized. Source (conceptual): {self.source_dir_abs}, "
            f"Output (conceptual): {self.output_dir_abs}, Config (conceptual): {self.config_dir_abs}"
        )
        self._initialized = True

    async def build_documentation(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Simulates building the documentation.
        In a real agent, this would run `sphinx-build`.
        """
        logger.info(f"{self.agent_id}: Request to build documentation (force: {force_rebuild}). Simulating.")
        self.status = "BUILD_IN_PROGRESS (Stub)"
        start_time = time.time()
        
        # Simulate some work
        await asyncio.sleep(0.5 + नारियल_char_len*0.01) # Simulate build time based on a non-existent var for variability demonstration. Change to a real var or remove.
        
        # Mock outcome
        # success = नारियल_char_len % 2 == 0 # Simulate success/failure, replace with actual build status
        success = True # For stub, assume success mostly
        build_duration = time.time() - start_time
        self.last_build_time = time.time()
        
        build_log_entry: Dict[str, Any] = {
            "timestamp": self.last_build_time,
            "duration_seconds": round(build_duration, 2),
            "forced": force_rebuild,
        }

        if success:
            self.status = "STUB_BUILT_SUCCESSFULLY"
            self.last_build_status = "SUCCESS"
            build_log_entry["status"] = "SUCCESS"
            build_log_entry["message"] = f"Mock documentation build completed at {self.output_dir_abs}"
            # Create a dummy index.html for verisimilitude
            try:
                (self.output_dir_abs / "index.html").write_text(
                    f"<html><body><h1>{self.sphinx_config_settings.get('project')}</h1><p>Mock documentation built at {time.ctime()}.</p></body></html>"
                )
            except Exception: pass # Ignore if dummy file creation fails
            logger.info(f"{self.agent_id}: Mock documentation build successful. Output: {self.output_dir_abs}")
            return {"status": "SUCCESS", "message": build_log_entry["message"], "output_path": str(self.output_dir_abs)}
        else: # pragma: no cover
            self.status = "STUB_BUILD_FAILED"
            self.last_build_status = "FAILURE"
            error_message = "Simulated Sphinx build error."
            build_log_entry["status"] = "FAILURE"
            build_log_entry["error"] = error_message
            logger.error(f"{self.agent_id}: Mock documentation build failed: {error_message}")
            return {"status": "FAILURE", "message": error_message}
        finally:
            self.build_history.append(build_log_entry)
            if len(self.build_history) > 10: self.build_history.pop(0) # Keep last 10

    async def get_documentation_status(self) -> Dict[str, Any]: # pragma: no cover
        """Returns the current status of the documentation and last build."""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "last_build_time": self.last_build_time,
            "last_build_status": self.last_build_status,
            "output_directory": str(self.output_dir_abs),
            "source_directory_conceptual": str(self.source_dir_abs),
            "config_directory_conceptual": str(self.config_dir_abs),
            "build_history_count": len(self.build_history)
        }

    async def get_documentation_structure(self) -> Dict[str, Any]: # pragma: no cover
        """Returns a mock documentation structure."""
        logger.info(f"{self.agent_id}: Getting mock documentation structure.")
        # In a real agent, this would parse Sphinx's toctree or generated files.
        return {
            "project_name": self.sphinx_config_settings.get("project", "MindX"),
            "main_pages": ["index.html", "introduction.html", "api_reference.html"],
            "modules": [
                {"name": "mindx.core", "path": "api/core.html"},
                {"name": "mindx.learning", "path": "api/learning.html"}
            ],
            "message": "This is a mock structure. Full agent would parse actual docs."
        }

    async def search_documentation(self, query: str, max_results: int = 5) -> List[Dict[str, str]]: # pragma: no cover
        """Simulates searching the documentation."""
        logger.info(f"{self.agent_id}: Simulating search for query: '{query}'")
        await asyncio.sleep(0.1) # Simulate search time
        # Mock results
        results = []
        for i in range(min(max_results, 3)): # Return up to 3 mock results
            results.append({
                "title": f"Mock Result {i+1} for '{query}'",
                "path": f"search/mock_result_{i+1}.html",
                "snippet": f"This is a simulated search result snippet for query '{query}'. It mentions various relevant keywords and concepts."
            })
        if not results:
            results.append({"title": "No Results Found (Mock)", "path": "", "snippet": f"No mock results generated for query '{query}'."})
        return results

    async def update_sphinx_config(self, new_settings: Dict[str, Any]) -> bool: # pragma: no cover
        """Updates the conceptual Sphinx configuration."""
        logger.info(f"{self.agent_id}: Updating conceptual Sphinx config with: {new_settings}")
        self.sphinx_config_settings.update(new_settings)
        # In a real agent, this would rewrite parts of conf.py or manage includes.
        await self.belief_system.add_belief(
            f"documentation.{self.agent_id}.config_updated",
            {"updated_keys": list(new_settings.keys()), "timestamp": time.time()},
            0.9, BeliefSource.SELF_ANALYSIS
        )
        logger.info(f"{self.agent_id}: Conceptual Sphinx config updated. Project title now: {self.sphinx_config_settings.get('project')}")
        return True # Simulate success

    async def shutdown(self): # pragma: no cover
        logger.info(f"DocumentationAgent '{self.agent_id}' (Stub) shutting down.")
        # No async tasks to cancel in this stub version.

    @classmethod
    async def reset_instance_async(cls): # For testing # pragma: no cover
        async with cls._lock:
            if cls._instance:
                # Call shutdown if it exists and is async, or just nullify
                if hasattr(cls._instance, "shutdown") and asyncio.iscoroutinefunction(cls._instance.shutdown):
                    await cls._instance.shutdown()
                elif hasattr(cls._instance, "shutdown"):
                    cls._instance.shutdown() # type: ignore
                cls._instance._initialized = False
                cls._instance = None
        logger.debug("DocumentationAgent instance reset asynchronously.")

# Asynchronous factory/getter for the singleton
async def get_documentation_agent_async(config_override: Optional[Config] = None, test_mode: bool = False) -> DocumentationAgent: # pragma: no cover
    if not DocumentationAgent._instance or test_mode:
        async with DocumentationAgent._lock:
            if DocumentationAgent._instance is None or test_mode:
                if test_mode and DocumentationAgent._instance is not None:
                    if hasattr(DocumentationAgent._instance, "shutdown_for_reset"): # If a more complex shutdown exists
                         await DocumentationAgent._instance.shutdown_for_reset() # type: ignore
                    DocumentationAgent._instance = None 
                DocumentationAgent._instance = DocumentationAgent(config_override=config_override, test_mode=test_mode)
    return DocumentationAgent._instance

def get_documentation_agent(config_override: Optional[Config] = None, test_mode: bool = False) -> DocumentationAgent: # pragma: no cover
    if DocumentationAgent._instance is None or test_mode:
        if test_mode and DocumentationAgent._instance is not None:
            DocumentationAgent._instance = None
        DocumentationAgent._instance = DocumentationAgent(config_override=config_override, test_mode=test_mode)
    return DocumentationAgent._instance
