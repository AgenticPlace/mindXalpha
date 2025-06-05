# mindx/tools/base_gen_agent.py
"""
BaseGenAgent (Codebase Documentation Generator): Generates Markdown 
documentation from a codebase directory. This version explicitly defaults
to loading its configuration from 'mindx/data/config/basegen_config.json'.
"""
import argparse
import fnmatch
import pathlib
import sys
import os
import json
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
import copy

import pathspec 

from mindx.utils.config import Config, PROJECT_ROOT # For PROJECT_ROOT
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseGenAgent:
    """
    Agent/Tool for generating a single Markdown document summarizing a codebase.
    It includes a directory tree and specified file contents, respecting .gitignore
    and custom include/exclude patterns. Its configuration is primarily loaded from
    PROJECT_ROOT/mindx/data/config/basegen_config.json.
    """

    # This is the *primary default path* for this agent's specific config.
    # It's relative to PROJECT_ROOT.
    DEFAULT_AGENT_CONFIG_FILE_PATH_RELATIVE = Path("mindx") / "data" / "config" / "basegen_config.json"

    # Internal fallback defaults if the JSON config file is missing or completely invalid.
    INTERNAL_FALLBACK_CONFIG_DATA: Dict[str, Any] = {
        "HARD_CODED_EXCLUDES": [
            "*.md", "*.txt", "*.pdf", "*.doc", "*.docx", "*.odt", "*.rtf",
            "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff", "*.svg", "*.ico", "*.webp",
            "*.mp3", "*.wav", "*.ogg", "*.flac", "*.aac", "*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.flv",
            "*.zip", "*.tar", "*.tar.gz", "*.tar.bz2", "*.rar", "*.7z", "*.gz", "*.bz2", "*.xz",
            "*.o", "*.so", "*.a", "*.dylib", "*.dll", "*.exe", "*.com", "*.msi", "*.deb", "*.rpm", "*.app",
            "*.class", "*.jar", "*.war", "*.ear", "*.pyc", "*.pyo", "*.pyd",
            "__pycache__/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/", "build/", "dist/", "target/",
            "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock", "composer.lock",
            "node_modules/", "bower_components/", "vendor/", "vendors/", 
            "venv/", ".venv/", "env/", ".env", ".env.*",
            "output/", "out/", "bin/", "obj/", ".git/", ".hg/", ".svn/", ".bzr/",
            ".DS_Store", "Thumbs.db", "*.log", "*.log.*", "*.tmp", "*.temp", "*.swp", "*.swo",
            ".idea/", ".vscode/", "*.sublime-project", "*.sublime-workspace",
            "coverage.xml", ".coverage", "nosetests.xml", "pytestdebug.log",
            "*.bak", "*.old", "*.orig", ".gitattributes", ".gitmodules",
            "LICENSE", "LICENSE.*", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md",
            "poetry.toml", "setup.cfg", "MANIFEST.in", "mkdocs.yml"
        ],
        "LANGUAGE_MAPPING": {
            ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
            ".java": "java", ".kt": "kotlin", ".swift": "swift", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
            ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php", ".sh": "bash",
            ".html": "html", ".css": "css", ".scss": "scss", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".xml": "xml", ".sql": "sql", ".dockerfile": "dockerfile", "dockerfile": "dockerfile",
            ".md": "markdown", ".rst": "rst", ".gitignore":"gitignore",
        },
        "base_gen_agent_settings": { 
            "max_file_size_kb_for_inclusion": 1024,
            "default_output_filename_stem": "codebase_snapshot",
            "output_subdir_relative_to_project": "data/generated_docs/codebase_snapshots"
        }
    }

    def __init__(self, config_file_path_override_str: Optional[str] = None, agent_id: str = "code_doc_generator_v2.1"):
        self.agent_id = agent_id
        self.log_prefix = f"{self.agent_id}:"
        
        # Determine the config file path: CLI override > Default path > None
        if config_file_path_override_str:
            self.agent_config_file_path = Path(config_file_path_override_str).resolve()
            logger.info(f"{self.log_prefix} Using specified config file: '{self.agent_config_file_path}'")
        else:
            self.agent_config_file_path = PROJECT_ROOT / self.DEFAULT_AGENT_CONFIG_FILE_PATH_RELATIVE
            logger.info(f"{self.log_prefix} Using default config file path: '{self.agent_config_file_path}'")
        
        self.config_data = self._load_and_merge_config_from_file()

        max_file_kb = self.get_agent_setting('max_file_size_kb_for_inclusion', 1024)
        logger.info(f"{self.log_prefix} Initialized. Max file size for content inclusion: {max_file_kb}KB.")

    def get_agent_setting(self, key: str, default_value: Any = None) -> Any:
        """Helper to get settings from the 'base_gen_agent_settings' block of config_data."""
        return self.config_data.get("base_gen_agent_settings", {}).get(key, default_value)

    def _load_and_merge_config_from_file(self) -> Dict[str, Any]:
        """
        Loads agent-specific configuration from its dedicated JSON file,
        merging it with internal fallback defaults.
        """
        current_config = copy.deepcopy(self.INTERNAL_FALLBACK_CONFIG_DATA) # Start with hardcoded class defaults

        if self.agent_config_file_path.exists() and self.agent_config_file_path.is_file():
            try:
                with self.agent_config_file_path.open("r", encoding="utf-8") as f:
                    loaded_config_from_file = json.load(f)
                
                self._deep_update_dict(current_config, loaded_config_from_file)
                logger.info(f"{self.log_prefix} Loaded and merged agent config from '{self.agent_config_file_path}'")
            except Exception as e: # pragma: no cover
                logger.error(f"{self.log_prefix} Error reading/parsing config file '{self.agent_config_file_path}': {e}. Using internal fallback defaults only.")
                current_config = copy.deepcopy(self.INTERNAL_FALLBACK_CONFIG_DATA) # Revert to pure defaults on error
        else: 
             logger.warning(f"{self.log_prefix} Agent config file '{self.agent_config_file_path}' not found. Using internal fallback defaults. Consider creating the file.")
        return current_config

    def _save_agent_config_to_file(self) -> bool: # pragma: no cover
        """Saves the current self.config_data to this agent's config file."""
        try:
            self.agent_config_file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.agent_config_file_path.open("w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=2)
            logger.info(f"{self.log_prefix} Agent configuration saved to '{self.agent_config_file_path}'")
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Error saving agent configuration to '{self.agent_config_file_path}': {e}", exc_info=True)
            return False

    def _deep_update_dict(self, target: Dict, source: Dict): # pragma: no cover
        """Helper for recursively updating nested dictionaries. Source values override target values."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update_dict(target[key], value)
            elif isinstance(value, list) and key in target and isinstance(target[key], list) and key == "HARD_CODED_EXCLUDES":
                # Special merge for HARD_CODED_EXCLUDES: append unique items from source to target
                existing_set = set(target[key])
                for item in value:
                    if item not in existing_set: target[key].append(item)
                target[key] = sorted(list(set(target[key]))) # Ensure sorted unique list
            else: # Direct override for other keys or if types mismatch for deep merge
                target[key] = value
    
    def update_config_setting(self, key_path: str, new_value: Any, save_after: bool = True) -> bool: # pragma: no cover
        """
        Updates a specific configuration setting in memory and optionally saves to file.
        key_path can be dot-separated for nested keys.
        Special list operations for "HARD_CODED_EXCLUDES":
        - To append unique: `new_value = ["item_to_add", {"_LIST_OP_": "APPEND_UNIQUE"}]` (op dict is first)
        - To remove: `new_value = ["item_to_remove", {"_LIST_OP_": "REMOVE"}]`
        - To replace: Pass the new list directly.
        """
        logger.info(f"{self.log_prefix} Attempting to update config: '{key_path}' with value (type: {type(new_value).__name__})")
        try:
            keys = key_path.split('.')
            d = self.config_data
            for key_part in keys[:-1]: # Navigate to the parent dictionary
                d = d.setdefault(key_part, {})
                if not isinstance(d, dict):
                    logger.error(f"{self.log_prefix} Config update failed: Path part '{key_part}' in '{key_path}' is not a dictionary."); return False
            
            final_key = keys[-1]
            
            # Handle special list operations for HARD_CODED_EXCLUDES
            if final_key == "HARD_CODED_EXCLUDES" and isinstance(new_value, list) and \
               new_value and isinstance(new_value[-1], dict) and "_LIST_OP_" in new_value[-1]:
                op_dict = new_value.pop() # Operation is the last item
                operation = op_dict["_LIST_OP_"]
                items_to_operate = new_value
                
                current_list = d.get(final_key, [])
                if not isinstance(current_list, list): current_list = [] # Ensure it's a list

                if operation == "APPEND_UNIQUE":
                    for item in items_to_operate:
                        if item not in current_list: current_list.append(item)
                    d[final_key] = sorted(list(set(current_list)))
                elif operation == "REMOVE":
                    d[final_key] = [item for item in current_list if item not in items_to_operate]
                else: # Default to replace if op unknown, or treat as error
                    logger.warning(f"{self.log_prefix} Unknown list operation '{operation}' for '{key_path}'. Replacing list.")
                    d[final_key] = items_to_operate # items_to_operate here is new_value without op_dict
            elif isinstance(new_value, dict) and final_key in d and isinstance(d[final_key], dict) and new_value.get("_MERGE_DEEP_", False):
                # Deep merge if _MERGE_DEEP_ is true (and remove it from new_value)
                new_value_copy = new_value.copy(); new_value_copy.pop("_MERGE_DEEP_")
                self._deep_update_dict(d[final_key], new_value_copy)
            else: # Direct set/override for other cases
                d[final_key] = new_value
            
            logger.info(f"{self.log_prefix} Config setting '{key_path}' updated in memory.")
            if save_after:
                return self._save_agent_config_to_file()
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to update config setting '{key_path}': {e}", exc_info=True)
            return False

    def _guess_language(self, file_extension: str) -> str: # pragma: no cover
        ext = file_extension.lower()
        if not ext.startswith('.'): ext = '.' + ext
        mapping = self.config_data.get("LANGUAGE_MAPPING", self.INTERNAL_FALLBACK_CONFIG_DATA["LANGUAGE_MAPPING"])
        return mapping.get(ext, '')

    def _load_gitignore_specs(self, root_path: pathlib.Path) -> Optional[pathspec.PathSpec]: # pragma: no cover
        # (Unchanged from previous functional version)
        all_patterns: List[str] = [];
        try:
            for gitignore_file_path in root_path.rglob(".gitignore"):
                try: lines = gitignore_file_path.read_text(encoding="utf-8").splitlines()
                except Exception as e_read: logger.warning(f"{self.log_prefix} Could not read {gitignore_file_path}: {e_read}"); continue
                try: gitignore_dir_relative_to_root = gitignore_file_path.parent.relative_to(root_path)
                except ValueError: gitignore_dir_relative_to_root = pathlib.Path("") 
                for line in lines:
                    line = line.strip();
                    if not line or line.startswith("#"): continue
                    # Construct pattern relative to root_path for pathspec
                    current_pattern_parts = []
                    if gitignore_dir_relative_to_root != pathlib.Path("."):
                        current_pattern_parts.extend(gitignore_dir_relative_to_root.parts)
                    
                    # Handle patterns starting with '/' (relative to .gitignore location) vs not
                    if line.startswith('/'):
                        current_pattern_parts.append(line.lstrip('/'))
                    else: # Relative to current level or any level below
                        current_pattern_parts.append(line) # pathspec handles this "anywhere below"
                    
                    all_patterns.append(Path(*current_pattern_parts).as_posix())

            if (root_path / ".git").is_dir(): all_patterns.append("/.git/") # Ensure .git at root is ignored
            if all_patterns: logger.debug(f"{self.log_prefix} Loaded {len(all_patterns)} patterns from .gitignore for {root_path}."); return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, all_patterns)
            else: logger.debug(f"{self.log_prefix} No .gitignore patterns found for {root_path}."); return None
        except Exception as e: logger.error(f"{self.log_prefix} Error loading .gitignore for {root_path}: {e}", exc_info=True); return None


    def _should_include_file( self, file_abs_path: pathlib.Path, root_path: pathlib.Path, 
        include_glob_patterns: Optional[List[str]], user_exclude_glob_patterns: Optional[List[str]], 
        gitignore_spec: Optional[pathspec.PathSpec], use_gitignore_rules: bool = True ) -> bool: # pragma: no cover
        # (Unchanged from previous functional version, uses self.config_data for HARD_CODED_EXCLUDES)
        try: relative_file_path_for_match = file_abs_path.relative_to(root_path).as_posix()
        except ValueError: logger.warning(f"{self.log_prefix} File {file_abs_path} not under root {root_path}. Excluding."); return False
        if use_gitignore_rules and gitignore_spec and gitignore_spec.match_file(relative_file_path_for_match): logger.debug(f"{self.log_prefix} Exclude '{relative_file_path_for_match}' by .gitignore."); return False
        
        hardcoded_excludes = self.config_data.get("HARD_CODED_EXCLUDES", self.INTERNAL_FALLBACK_CONFIG_DATA["HARD_CODED_EXCLUDES"])
        all_exclude_glob_patterns = list(set((user_exclude_glob_patterns or []) + hardcoded_excludes))
        
        if any(fnmatch.fnmatch(relative_file_path_for_match, pattern) for pattern in all_exclude_glob_patterns): logger.debug(f"{self.log_prefix} Exclude '{relative_file_path_for_match}' by custom exclude pattern."); return False
        if include_glob_patterns and not any(fnmatch.fnmatch(relative_file_path_for_match, pattern) for pattern in include_glob_patterns): logger.debug(f"{self.log_prefix} Exclude '{relative_file_path_for_match}' (no include match)."); return False
        return True

    def _build_tree_dict(self, root_scan_path: pathlib.Path, relative_file_paths_to_include: List[pathlib.Path]) -> Dict[str, Any]: # pragma: no cover
        # (Unchanged from previous)
        tree: Dict[str, Any] = {};
        for rel_path in sorted(relative_file_paths_to_include): parts = rel_path.parts; current_level = tree;
        for part in parts[:-1]: current_level = current_level.setdefault(part, {}); current_level[parts[-1]] = {} # Mark as file
        return tree

    def _format_tree_lines(self, tree_dict: Dict[str, Any], indent_str: str = "", is_root: bool = True) -> List[str]: # pragma: no cover
        # (Unchanged from previous)
        lines = []; sorted_keys = sorted(tree_dict.keys(), key=lambda k: (not bool(tree_dict[k]), k.lower()))
        for i, key in enumerate(sorted_keys):
            is_last_item_in_level = (i == len(sorted_keys) - 1); 
            connector = "" if is_root else ("└── " if is_last_item_in_level else "├── ")
            new_indent_for_children = indent_str + ("    " if (is_root or is_last_item_in_level) else "│   ")
            is_directory = bool(tree_dict[key]); 
            lines.append(f"{indent_str}{connector}{key}{'/' if is_directory else ''}")
            if is_directory: lines.extend(self._format_tree_lines(tree_dict[key], new_indent_for_children, is_root=False))
        return lines

    def generate_documentation(
        self, root_path_str: str, output_file_str: Optional[str] = None,
        include_patterns: Optional[List[str]] = None, user_exclude_patterns: Optional[List[str]] = None,
        use_gitignore: bool = True ) -> Dict[str, Any]: # pragma: no cover
        # (Unchanged from previous, uses self.get_agent_setting)
        root_path = pathlib.Path(root_path_str).resolve()
        effective_output_file: Path
        if output_file_str: effective_output_file = Path(output_file_str);
        if not effective_output_file.is_absolute(): effective_output_file = (PROJECT_ROOT / effective_output_file).resolve()
        else:
            input_dir_name = root_path.name; default_stem = self.get_agent_setting("default_output_filename_stem", "codebase_snapshot")
            output_subdir = self.get_agent_setting("output_subdir_relative_to_project", "data/generated_docs/codebase_snapshots")
            effective_output_file = PROJECT_ROOT / output_subdir / f"{input_dir_name}_{default_stem}.md"
        logger.info(f"{self.log_prefix} Doc gen: Input='{root_path}', Output='{effective_output_file}'")
        if not root_path.is_dir(): err_msg = f"Input path '{root_path}' not dir."; return {"status": "ERROR", "message": err_msg, "output_file": None, "files_included": 0}
        
        gitignore_spec = self._load_gitignore_specs(root_path) if use_gitignore else None
        included_files_rel: List[pathlib.Path] = []
        try:
            for item_abs in sorted(root_path.rglob("*")):
                if item_abs.is_file() and self._should_include_file(item_abs, root_path, include_patterns, user_exclude_patterns, gitignore_spec, use_gitignore):
                    included_files_rel.append(item_abs.relative_to(root_path))
        except Exception as e_scan: err_msg = f"Error scanning '{root_path}': {e_scan}"; return {"status": "ERROR", "message": err_msg, "output_file": None, "files_included": 0}

        tree_str = "[No files included]";
        if included_files_rel:
            try: tree_dict = self._build_tree_dict(root_path, included_files_rel); tree_lines = self._format_tree_lines(tree_dict, is_root=True); tree_str = "\n".join([root_path.name + "/"] + tree_lines) if tree_lines else root_path.name + "/"
            except Exception as e_tree: err_msg = f"Error building tree: {e_tree}"; return {"status": "ERROR", "message": err_msg, "output_file": None, "files_included": 0}

        md_lines = [f"# Codebase: {root_path.name}", f"Generated by: {self.agent_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"Source: `{root_path_str}`", "", "## Directory Structure", "", "```text", tree_str, "```", "", "## File Contents", ""]
        max_fsize_bytes = self.get_agent_setting("max_file_size_kb_for_inclusion", 1024) * 1024
        num_processed = 0
        for rel_p in included_files_rel:
            file_abs = root_path / rel_p; num_processed +=1
            md_lines.append(f"### `{rel_p.as_posix()}`\n\n```{self._guess_language(file_abs.suffix)}");
            try:
                if file_abs.stat().st_size > max_fsize_bytes: content = f"[File content omitted: Size ({file_abs.stat().st_size // 1024}KB) exceeds limit ({max_fsize_bytes // 1024}KB)]"
                else: content = file_abs.read_text(encoding="utf-8", errors="replace")
            except Exception as e_rc: content = f"[Error reading file '{rel_p.as_posix()}': {e_rc}]"
            md_lines.append(content.strip()); md_lines.append("```\n")
        try:
            effective_output_file.parent.mkdir(parents=True, exist_ok=True)
            with effective_output_file.open("w", encoding="utf-8") as f: f.write("\n".join(md_lines))
            msg = f"Markdown documentation generated: {effective_output_file}"; logger.info(f"{self.log_prefix} {msg}")
            return {"status": "SUCCESS", "message": msg, "output_file": str(effective_output_file), "files_included": num_processed}
        except Exception as e_w: err_msg = f"Error writing to '{effective_output_file}': {e_w}"; return {"status": "ERROR", "message": err_msg, "output_file": str(effective_output_file), "files_included": num_processed}

def main_cli(): # pragma: no cover
    """CLI entry point for BaseGenAgent (Codebase Documentation Generator)."""
    parser = argparse.ArgumentParser( description="Generate Markdown documentation for a codebase.", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument("input_dir", help="Path to the codebase root directory.")
    parser.add_argument("-o", "--output", default=None, help="Output Markdown file path. If None, a default is generated.")
    parser.add_argument("--include", nargs="+", help="Glob pattern(s) for files to include.")
    parser.add_argument("--exclude", nargs="+", help="Glob pattern(s) for files/dirs to exclude.")
    parser.add_argument("--no-gitignore", action="store_true", help="Disable .gitignore exclusions.")
    parser.add_argument("--config-file", default=None, help=f"Path to custom agent JSON config (default: {PROJECT_ROOT / BaseGenAgent.DEFAULT_AGENT_CONFIG_SUBDIR / BaseGenAgent.DEFAULT_AGENT_CONFIG_FILENAME}).")
    parser.add_argument("--update-config", default=None, help="JSON string of settings to update in agent's config. Ex: '{\"key.path\": \"value\"}'")
    
    args = parser.parse_args()
    Config() # Ensure global config loads for PROJECT_ROOT etc.

    try:
        agent = BaseGenAgent(config_file_path_override_str=args.config_file)
        
        if args.update_config:
            try:
                updates_dict = json.loads(args.update_config)
                if not isinstance(updates_dict, dict): raise ValueError("Update data must be a JSON object string.")
                all_ok = True
                for key_path, new_val in updates_dict.items():
                    if not agent.update_config_setting(key_path, new_val, save_after=False): all_ok = False
                if all_ok: 
                    agent._save_agent_config_to_file()
                    print(json.dumps({"status": "SUCCESS", "message": f"Agent config '{agent.agent_config_file_path}' updated."}))
                else: print(json.dumps({"status": "FAILURE", "message": "One or more config updates failed."}))
                sys.exit(0 if all_ok else 1)
            except Exception as e_cfg_upd: print(json.dumps({"status": "ERROR", "message": f"Error updating config: {e_cfg_upd}"}), file=sys.stderr); sys.exit(2)

        output_file_path = args.output
        if not output_file_path:
            input_dir_name_slug = Path(args.input_dir).resolve().name.replace(" ", "_")
            default_stem = agent.get_agent_setting("default_output_filename_stem", "codebase_doc")
            output_subdir = agent.get_agent_setting("output_subdir_relative_to_project", "data/generated_docs/codebase_snapshots")
            output_file_path = str(PROJECT_ROOT / output_subdir / f"{input_dir_name_slug}_{default_stem}.md")

        result = agent.generate_documentation(
            root_path_str=args.input_dir, output_file_str=output_file_path,
            include_patterns=args.include, user_exclude_patterns=args.exclude,
            use_gitignore=not args.no_gitignore,
        )
        print(json.dumps(result, indent=2)); sys.exit(0 if result["status"] == "SUCCESS" else 1)
    except Exception as e:
        print(json.dumps({"status": "FATAL_ERROR", "message": str(e), "error_type": type(e).__name__}), file=sys.stderr)
        logger.critical(f"Fatal error in BaseGenAgent (CodeDocGen) CLI: {e}", exc_info=True); sys.exit(2)

if __name__ == "__main__": # pragma: no cover
    # This ensures that if the script is run directly, PROJECT_ROOT is determined correctly by Config
    # before BaseGenAgent uses it.
    Config() 
    main_cli()
