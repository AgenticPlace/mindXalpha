#!/usr/bin/env python3
# mindx/tools/base_gen_agent.py
"""
BaseGenAgent: Generates Markdown documentation from a codebase directory.
This agent is designed to be configurable and callable by other MindX components.
"""
import argparse
import fnmatch
import pathlib
import sys
import os
import json
from typing import List, Optional, Dict, Tuple, Any

import pathspec # Requires: pip install pathspec

from mindx.utils.config import Config, PROJECT_ROOT # For default config location
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseGenAgent:
    """
    Agent for generating a Markdown document from a codebase.
    It includes a directory tree and file contents, respecting .gitignore
    and custom include/exclude patterns loaded from an external configuration file.
    """

    DEFAULT_CONFIG_DATA = { # Fallback if no external config file is found
        "HARD_CODED_EXCLUDES": [
            "*.md", "*.txt", "*.pdf", "*.doc", "*.docx", "*.odt", "*.ttf", "*.otf", "*.woff", "*.woff2",
            "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.svg", "*.ico",
            "*.mp3", "*.wav", "*.ogg", "*.flac", "*.aac", "*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm",
            "*.zip", "*.tar.gz", "*.tar.bz2", "*.rar", "*.7z", "*.gz", "*.bz2", "*.o", "*.so", "*.a",
            "*.dylib", "*.dll", "*.exe", "*.com", "*.msi", "*.deb", "*.rpm", "*.class", "*.jar",
            "*.war", "*.ear", "*.pyc", "*.pyo", "__pycache__/", "package-lock.json", "yarn.lock",
            "poetry.lock", "Pipfile.lock", "composer.lock", "node_modules/", "vendor/", "vendors/",
            "venv/", ".venv/", "env/", ".env_dev", ".env_prod", "target/", "dist/", "build/", "out/",
            "bin/", "obj/", ".git/", ".hg/", ".svn/", ".bzr/", ".DS_Store", "Thumbs.db", "*.log",
            "*.tmp", "*.temp", "*.swp", "*.swo", ".idea/", ".vscode/", "*.sublime-project",
            "*.sublime-workspace", "coverage.xml", ".coverage", "nosetests.xml", "*.bak", "*.old",
            ".gitattributes", ".gitmodules", "LICENSE", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md",
            "SECURITY.md", "poetry.toml"
        ],
        "LANGUAGE_MAPPING": { ".py": "python", ".js": "javascript", ".json": "json", ".md": "markdown", ".html":"html", ".css":"css", ".sh":"bash", ".yaml":"yaml", ".yml":"yaml", ".xml":"xml", ".java":"java", ".c":"c", ".cpp":"cpp", ".h":"c", ".hpp":"cpp", ".rs":"rust", ".go":"go", ".ts":"typescript", ".tsx":"typescript", ".rb":"ruby", ".php":"php", ".swift":"swift", ".kt":"kotlin", ".scala":"scala", ".pl":"perl", ".lua":"lua", ".r":"r", ".sql":"sql", ".dockerfile":"dockerfile", "dockerfile":"dockerfile", ".gitignore":"gitignore"},
        "base_gen_agent": { "max_file_size_kb_for_inclusion": 1024, "default_output_filename": "codebase_snapshot.md"}
    }

    def __init__(self, config_file_path_str: Optional[str] = None, agent_id: str = "base_gen_agent_v2"):
        """
        Initialize the BaseGenAgent.

        Args:
            config_file_path_str: Optional path to a custom basegen_config.json file.
                                  If None, tries 'data/config/basegen_config.json' relative to PROJECT_ROOT.
            agent_id: Identifier for this agent instance.
        """
        self.agent_id = agent_id
        self.config_data = self._load_agent_config(config_file_path_str)
        self.log_prefix = f"{self.agent_id}:"
        logger.info(f"{self.log_prefix} Initialized. Using {len(self.config_data.get('HARD_CODED_EXCLUDES',[]))} hardcoded excludes. Max file size for inclusion: {self._get_agent_specific_config('max_file_size_kb_for_inclusion')}KB.")

    def _get_agent_specific_config(self, key: str, default_value: Any = None) -> Any:
        """Helper to get config under 'base_gen_agent' key."""
        return self.config_data.get("base_gen_agent", {}).get(key, default_value)

    def _load_agent_config(self, config_file_path_override_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration settings.
        Priority: config_file_path_override_str > PROJECT_ROOT/data/config/basegen_config.json > class DEFAULT_CONFIG_DATA.
        """
        config_to_load: Optional[Path] = None
        
        if config_file_path_override_str:
            config_to_load = Path(config_file_path_override_str)
            if not config_to_load.exists(): # pragma: no cover
                logger.warning(f"{self.log_prefix} Specified config override '{config_to_load}' not found. Trying default.")
                config_to_load = None # Fallback to default location

        if not config_to_load:
            default_path = PROJECT_ROOT / "data" / "config" / "basegen_config.json"
            if default_path.exists():
                config_to_load = default_path
            else: # pragma: no cover
                logger.info(f"{self.log_prefix} Default config '{default_path}' not found. Using internal defaults.")
                return self.DEFAULT_CONFIG_DATA.copy() # Use a copy

        if config_to_load and config_to_load.exists():
            try:
                with config_to_load.open("r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                
                # Merge with internal defaults: loaded config values override or extend.
                merged_config = self.DEFAULT_CONFIG_DATA.copy() # Start with internal defaults
                
                # Deep merge for nested dicts like LANGUAGE_MAPPING and base_gen_agent settings
                # For lists like HARD_CODED_EXCLUDES, append and deduplicate.
                for key, value in loaded_config.items():
                    if key in merged_config:
                        if isinstance(merged_config[key], list) and isinstance(value, list):
                            merged_config[key].extend(value)
                            merged_config[key] = sorted(list(set(merged_config[key])))
                        elif isinstance(merged_config[key], dict) and isinstance(value, dict):
                            merged_config[key].update(value) # Simple update for 1-level dicts
                        else: # Direct override for other types or if types don't match for merge
                            merged_config[key] = value
                    else: # New key from loaded config
                        merged_config[key] = value
                
                logger.info(f"{self.log_prefix} Loaded and merged agent config from '{config_to_load}'")
                return merged_config
            except Exception as e: # pragma: no cover
                logger.error(f"{self.log_prefix} Error reading or parsing config file '{config_to_load}': {e}. Using internal defaults.")
                return self.DEFAULT_CONFIG_DATA.copy()
        
        # Should be unreachable if logic above is correct, but as a final fallback:
        return self.DEFAULT_CONFIG_DATA.copy() # pragma: no cover


    def _guess_language(self, ext: str) -> str:
        ext = ext.lower()
        mapping = self.config_data.get("LANGUAGE_MAPPING", {})
        return mapping.get(ext, '')

    def _load_gitignore_specs(self, root: pathlib.Path) -> Optional[pathspec.PathSpec]:
        # (Unchanged from previous BaseGenAgent - this logic is sound)
        patterns = [];
        try:
            for gitignore_path in root.rglob(".gitignore"):
                try: lines = gitignore_path.read_text(encoding="utf-8").splitlines()
                except Exception as e: logger.warning(f"{self.log_prefix} Could not read {gitignore_path}: {e}"); continue
                try: rel_dir_of_gitignore = gitignore_path.parent.relative_to(root)
                except ValueError: rel_dir_of_gitignore = pathlib.Path("")
                for line in lines:
                    line = line.strip();
                    if not line or line.startswith("#"): continue
                    if rel_dir_of_gitignore != pathlib.Path(""): pattern = str(rel_dir_of_gitignore / line.lstrip('/')) if line.startswith('/') else str(rel_dir_of_gitignore / line)
                    else: pattern = line
                    patterns.append(pattern)
            if root.is_dir(): patterns.append(".git/") # Ensure .git at root is ignored
            logger.debug(f"{self.log_prefix} Loaded {len(patterns)} patterns from .gitignore files for root {root}.")
            return pathspec.PathSpec.from_lines("gitwildmatch", patterns) if patterns else None
        except Exception as e: logger.error(f"{self.log_prefix} Error loading .gitignore for {root}: {e}"); return None

    def _should_include_file(
        self, file_path: pathlib.Path, root_path: pathlib.Path, 
        include_patterns: Optional[List[str]], user_exclude_patterns: Optional[List[str]], # Renamed for clarity
        gitignore_spec: Optional[pathspec.PathSpec] ) -> bool:
        # (Unchanged from previous BaseGenAgent - this logic is sound)
        try: relative_file_path = file_path.relative_to(root_path)
        except ValueError: logger.warning(f"{self.log_prefix} File {file_path} not under root {root_path}. Excluding."); return False
        path_for_match_str = relative_file_path.as_posix()
        if gitignore_spec and gitignore_spec.match_file(path_for_match_str): logger.debug(f"{self.log_prefix} Exclude '{path_for_match_str}' by .gitignore."); return False
        
        # Combine user excludes with hardcoded ones from config
        hardcoded_excludes = self.config_data.get("HARD_CODED_EXCLUDES", [])
        all_exclude_patterns = list(set((user_exclude_patterns or []) + hardcoded_excludes))

        if all_exclude_patterns and any(fnmatch.fnmatch(path_for_match_str, pattern) for pattern in all_exclude_patterns):
            logger.debug(f"{self.log_prefix} Exclude '{path_for_match_str}' by exclude pattern.")
            return False
        if include_patterns and not any(fnmatch.fnmatch(path_for_match_str, pattern) for pattern in include_patterns):
            logger.debug(f"{self.log_prefix} Exclude '{path_for_match_str}' (no include match).")
            return False
        return True

    def _build_tree_dict(self, root_scan_path: pathlib.Path, relative_file_paths: List[pathlib.Path]) -> Dict[str, Any]:
        # (Unchanged from previous BaseGenAgent)
        tree: Dict[str, Any] = {}
        for rel_path in relative_file_paths:
            parts = rel_path.parts; current_level = tree
            for part in parts[:-1]: current_level = current_level.setdefault(part, {})
            current_level[parts[-1]] = {} # Mark file
        return tree

    def _format_tree_lines(self, tree_dict: Dict[str, Any], indent_str: str = "") -> List[str]:
        # (Unchanged from previous BaseGenAgent - improved tree formatting)
        lines = []; sorted_keys = sorted(tree_dict.keys(), key=lambda k: (not bool(tree_dict[k]), k))
        for i, key in enumerate(sorted_keys):
            is_last = (i == len(sorted_keys) - 1); connector = "└── " if is_last else "├── "
            if tree_dict[key]: lines.append(f"{indent_str}{connector}{key}/"); new_indent = indent_str + ("    " if is_last else "│   "); lines.extend(self._format_tree_lines(tree_dict[key], new_indent))
            else: lines.append(f"{indent_str}{connector}{key}")
        return lines

    def generate_documentation(
        self,
        root_path_str: str,
        output_file_str: Optional[str] = None, # Made optional, defaults from config
        include_patterns: Optional[List[str]] = None,
        user_exclude_patterns: Optional[List[str]] = None,
        use_gitignore: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates a Markdown document for the codebase.
        Returns a status dictionary: {"status": "SUCCESS"|"ERROR", "message": str, "output_file": str|None, "files_included": int}.
        """
        root_path = pathlib.Path(root_path_str).resolve()
        
        if output_file_str:
            output_file = pathlib.Path(output_file_str).resolve()
        else:
            default_fname = self._get_agent_specific_config("default_output_filename", "codebase_snapshot.md")
            output_file = PROJECT_ROOT / "data" / "generated_docs" / default_fname # Store in project data by default
        
        logger.info(f"{self.log_prefix} Starting documentation generation for '{root_path}' -> '{output_file}'")

        if not root_path.is_dir():
            err_msg = f"Input path '{root_path}' is not a valid directory."; logger.error(f"{self.log_prefix} {err_msg}")
            return {"status": "ERROR", "message": err_msg, "output_file": None, "files_included": 0}

        gitignore_spec = self._load_gitignore_specs(root_path) if use_gitignore else None
        
        included_files_relative_to_root: List[pathlib.Path] = []
        try:
            for file_abs_path in sorted(root_path.rglob("*")): # Iterate over absolute paths
                if file_abs_path.is_file() and self._should_include_file(
                    file_abs_path, root_path, include_patterns, user_exclude_patterns, gitignore_spec ):
                    included_files_relative_to_root.append(file_abs_path.relative_to(root_path))
        except Exception as e: # pragma: no cover
            err_msg = f"Error scanning directory '{root_path}': {e}"; logger.error(f"{self.log_prefix} {err_msg}", exc_info=True)
            return {"status": "ERROR", "message": err_msg, "output_file": None, "files_included": 0}

        if not included_files_relative_to_root:
            logger.warning(f"{self.log_prefix} No files matched criteria in '{root_path}'. Output will be minimal.")

        tree_str = "[No files included to generate tree]"
        if included_files_relative_to_root:
            try:
                tree_dict = self._build_tree_dict(root_path, included_files_relative_to_root)
                tree_lines = self._format_tree_lines(tree_dict); tree_str = "\n".join(tree_lines)
            except Exception as e: # pragma: no cover
                err_msg = f"Error building directory tree: {e}"; logger.error(f"{self.log_prefix} {err_msg}", exc_info=True)
                return {"status": "ERROR", "message": err_msg, "output_file": None, "files_included": 0}

        md_lines = [f"# Codebase Documentation: {root_path.name}", f"Generated by: {self.agent_id} at {datetime.now().isoformat()}", 
                    "", "## Directory Tree Overview", "", "```text", tree_str, "```", "", "## File Contents", ""]
        
        max_file_size = self._get_agent_specific_config("max_file_size_kb_for_inclusion", 1024) * 1024
        for rel_path in included_files_relative_to_root:
            file_abs_path = root_path / rel_path
            md_lines.append(f"### `{rel_path.as_posix()}`")
            md_lines.append("")
            language_tag = self._guess_language(file_abs_path.suffix)
            md_lines.append(f"```{language_tag}")
            try:
                if file_abs_path.stat().st_size > max_file_size: # pragma: no cover
                    content = f"Error: File '{rel_path.as_posix()}' too large (>{max_file_size // 1024}KB). Omitted."
                    logger.warning(f"{self.log_prefix} {content}")
                else: content = file_abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception as e_read: content = f"Error reading file '{rel_path.as_posix()}': {e_read}"; logger.warning(f"{self.log_prefix} {content}") # pragma: no cover
            md_lines.append(content); md_lines.append("```"); md_lines.append("")

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as f: f.write("\n".join(md_lines))
            msg = f"Markdown generated: {output_file}"; logger.info(f"{self.log_prefix} {msg}")
            return {"status": "SUCCESS", "message": msg, "output_file": str(output_file), "files_included": len(included_files_relative_to_root)}
        except Exception as e: # pragma: no cover
            err_msg = f"Error writing output to '{output_file}': {e}"; logger.error(f"{self.log_prefix} {err_msg}", exc_info=True)
            return {"status": "ERROR", "message": err_msg, "output_file": str(output_file), "files_included": len(included_files_relative_to_root)}


def main_cli(): # pragma: no cover
    """CLI entry point for BaseGenAgent."""
    parser = argparse.ArgumentParser( description="Generate Markdown documentation for a codebase.", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument("input_dir", help="Path to the codebase root directory.")
    parser.add_argument("-o", "--output", default=None, help="Output Markdown file name. Defaults to data/generated_docs/<input_dir_name>_snapshot.md.")
    parser.add_argument("--include", nargs="+", help="Glob pattern(s) for files to include (relative to root).")
    parser.add_argument("--exclude", nargs="+", help="Glob pattern(s) for files to exclude (relative to root).")
    parser.add_argument("--no-gitignore", action="store_true", help="Disable applying .gitignore file exclusions.")
    parser.add_argument("--config-file", default=None, help="Path to a custom basegen_config.json file.")
    
    args = parser.parse_args()

    # Ensure MindX top-level logging is configured if run as standalone script
    if not logging.getLogger("mindx").hasHandlers() and not logging.getLogger().hasHandlers():
        logging.basicConfig(level=os.environ.get("MINDX_LOG_LEVEL", "INFO").upper(), 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    try:
        agent = BaseGenAgent(config_file_path_str=args.config_file)
        
        output_filename = args.output
        if not output_filename: # Construct default if not provided
            input_dir_name = Path(args.input_dir).resolve().name # Get name of the input directory
            default_fname_from_config = agent._get_agent_specific_config("default_output_filename", "codebase_snapshot.md")
            # Ensure a more unique default name based on input dir if possible
            if default_fname_from_config == "codebase_snapshot.md": # only override if it's the generic default
                 output_filename = PROJECT_ROOT / "data" / "generated_docs" / f"{input_dir_name}_{default_fname_from_config}"
            else:
                 output_filename = PROJECT_ROOT / "data" / "generated_docs" / default_fname_from_config


        result = agent.generate_documentation(
            root_path_str=args.input_dir,
            output_file_str=str(output_filename), # Ensure it's a string
            include_patterns=args.include,
            user_exclude_patterns=args.exclude, # Pass as user_exclude_patterns
            use_gitignore=not args.no_gitignore,
        )
        
        print(json.dumps(result, indent=2)) # Output result as JSON for CLI scripting
        sys.exit(0 if result["status"] == "SUCCESS" else 1)

    except Exception as e:
        print(json.dumps({"status": "FATAL_ERROR", "message": str(e), "error_type": type(e).__name__}), file=sys.stderr)
        # logger.critical(f"A fatal error occurred in BaseGenAgent CLI: {e}", exc_info=True) # Logger might not be fully setup if Config fails
        sys.exit(2)

if __name__ == "__main__": # pragma: no cover
    main_cli()
