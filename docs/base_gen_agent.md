# BaseGenAgent (`base_gen_agent.py`) - Configurable Codebase Documenter
optimized in markdown for LLM ingestion<br />
mindX/tools/base_gen_agent.py<br />
data/config/basegen_config.json<br />


## Introduction

The `BaseGenAgent` is a utility agent within the MindX toolkit (Augmentic Project). Its primary function is to automatically generate a comprehensive Markdown document that provides a snapshot of a given codebase directory. This documentation includes a visual directory tree of included files and the complete content of those files, with appropriate language tagging for Markdown syntax highlighting.

A key feature of this agent is its configurability. It intelligently filters files based on:
  `.gitignore` rules found within the target codebase.
  User-defined include/exclude glob patterns passed via CLI or programmatically.
  A central, modifiable JSON configuration file (`basegen_config.json`) that specifies hardcoded file/pattern exclusions and language mappings for syntax highlighting.

This makes `BaseGenAgent` a valuable tool for MindX itself (e.g., for the `SelfImprovementAgent` or `CoordinatorAgent` to understand code they are about to modify) or for developers needing a quick, shareable overview of a project or component.

## Explanation

### Core Functionality

  **Configuration Loading (`_load_agent_config`):**
    *   The agent's behavior is controlled by a JSON configuration file, typically `PROJECT_ROOT/data/config/basegen_config.json`. A custom path can also be provided during instantiation.
    *   If the external config file is not found, it falls back to internal `DEFAULT_CONFIG_DATA`.
    *   **Configuration Merging:** Values from an external `basegen_config.json` are merged with the internal defaults. For lists like `HARD_CODED_EXCLUDES`, entries are combined and deduplicated. For dictionaries like `LANGUAGE_MAPPING` and the `base_gen_agent` settings block, external values update or override internal defaults.
    *   **Key Configurable Sections:**
        -   `HARD_CODED_EXCLUDES`: A list of glob patterns for common binary files, lock files, IDE metadata, temporary files, and version control directories (e.g., `*.png`, `node_modules/`, `.git/`) that are generally excluded from code documentation.
        -   `LANGUAGE_MAPPING`: A dictionary mapping file extensions (e.g., `.py`, `.rs`) to language tags recognized by Markdown for syntax highlighting (e.g., `python`, `rust`).
        -   `base_gen_agent`: A sub-dictionary for settings specific to this agent:
            -   `max_file_size_kb_for_inclusion`: (Default: 1024KB) Files larger than this will have their content omitted with a warning.
            -   `default_output_filename`: Default name for the output Markdown if not specified by the caller.

  **File Discovery and Filtering Logic (`generate_documentation`, `_should_include_file`):**
    *   The agent recursively scans the target `root_path_str` directory.
    *   **`.gitignore` Processing (`_load_gitignore_specs`):** If `use_gitignore` is true (default), it finds all `.gitignore` files within the `root_path_str`, aggregates their patterns, and compiles them into a `pathspec.PathSpec` object. This spec is used to efficiently exclude any files or directories ignored by Git. The `.git/` directory itself is always implicitly ignored.
    *   **Filtering Precedence for `_should_include_file`:**
          If a file matches the `gitignore_spec` (and `use_gitignore` is true), it's **excluded**.
          The file is then checked against the combined exclude patterns (CLI/programmatic `user_exclude_patterns` + `HARD_CODED_EXCLUDES` from config). If it matches any, it's **excluded**.
          If `include_patterns` are provided (CLI/programmatic), the file **must match at least one** of these to be considered further. If it doesn't match any, it's **excluded**.
          If none of the above exclusion rules apply, the file is **included**.

  **Directory Tree Generation (`_build_tree_dict`, `_format_tree_lines`):**
    *   A list of included files (as relative paths from the `root_path_str`) is used.
    *   `_build_tree_dict`: Constructs a nested dictionary representing the directory hierarchy of these included files.
    *   `_format_tree_lines`: Recursively traverses this dictionary to create an indented, human-readable string representation of the tree structure, suitable for Markdown `text` code blocks. Directories are marked with a trailing `/`.

  **Markdown Document Generation (`generate_documentation`):**
    *   This is the main public method.
    *   It orchestrates scanning, filtering, and tree generation.
    *   It then iterates through the list of included files:
        *   For each file, a Markdown section is created with its POSIX-style relative path as a heading (e.g., `### \`src/module/file.py\``).
        *   The file's content is read (UTF-8, replacing errors). If a file exceeds `max_file_size_kb_for_inclusion`, its content is omitted, and a warning is included in the Markdown.
        *   `_guess_language()` determines the Markdown language tag for syntax highlighting based on the file extension and the `LANGUAGE_MAPPING` from the configuration.
        *   The file content is embedded within a fenced code block (e.g., \`\`\`python ... \`\`\`).
    *   The complete Markdown content (tree + file contents) is written to the specified output file. The default output path is `PROJECT_ROOT/data/generated_docs/<input_dir_name>_codebase_snapshot.md`.
    *   **Return Value:** Returns a dictionary summarizing the operation: `{"status": "SUCCESS"|"ERROR", "message": str, "output_file": str|None, "files_included": int}`. This structured return makes it suitable for programmatic use by other MindX agents.

### Agent Structure

-   The core logic is encapsulated within the `BaseGenAgent` class.
-   Helper methods are private (prefixed with `_`).
-   Configuration is loaded during instantiation.

## Technical Details

-   **Path Handling:** Uses `pathlib.Path` for robust, cross-platform path operations. Paths are generally resolved to absolute paths for internal consistency.
-   **Pattern Matching:**
    -   `.gitignore`: `pathspec` library (requires `pip install pathspec`).
    -   Include/Exclude Globs: Python's `fnmatch` module.
-   **Configuration:** Loads from an external `basegen_config.json` file, falling back to internal defaults. This external file is a target for potential self-modification by MindX.
-   **Error Handling:** Includes `try-except` blocks for file I/O, JSON parsing, and directory scanning. Errors are logged, and the main method returns an error status.
-   **Standalone CLI:** A `main_cli()` function using `argparse` allows direct command-line execution of the agent, outputting its status dictionary as JSON.

## Usage

### As a Standalone CLI Tool

The agent can be executed directly:

```bash
python mindx/tools/base_gen_agent.py <input_dir> [options]
```
Key Arguments:
input_dir: Path to the codebase root.
-o, --output <filename>: Output Markdown file. Defaults to PROJECT_ROOT/data/generated_docs/<input_dir_name>_codebase_snapshot.md.
--include <pattern>: Glob(s) for files to include.
--exclude <pattern>: Glob(s) for files to exclude (these are additional to config excludes).
--no-gitignore: Ignore .gitignore files.
--config-file <path/to/basegen_config.json>: Path to a custom agent configuration JSON file.
Example:
```bash
python mindx/tools/base_gen_agent.py ./mindx \
    -o ./data/generated_docs/mindx_core_docs.md \
    --include "mindx/core/**/*.py" \
    --exclude "**/__pycache__/*" \
    --config-file ./data/config/custom_basegen_config.json
```

The CLI will print a JSON object summarizing the result.
Programmatic Usage by Other MindX Agents
The CoordinatorAgent or StrategicEvolutionAgent can instantiate and call BaseGenAgent to get a structured understanding of a component they intend to analyze or modify
```python
# from mindx.tools.base_gen_agent import BaseGenAgent
# from mindx.utils.config import PROJECT_ROOT

# async def some_mindx_agent_task():
#     # Assuming 'config_for_basegen_tool.json' exists or using defaults
#     base_gen = BaseGenAgent(config_file_path_str=str(PROJECT_ROOT / "data" / "config" / "basegen_config.json"))
    
#     target_module_path = str(PROJECT_ROOT / "mindx" / "utils")
#     output_doc_path = str(PROJECT_ROOT / "data" / "temp_docs" / "utils_snapshot.md")
    
#     # Ensure the agent method can be awaited if it were async; currently it's sync.
#     # If BaseGenAgent methods become async, this call needs await.
#     # For now, assuming generate_documentation is synchronous.
#     # If used in an async agent, run it in an executor:
#     # loop = asyncio.get_running_loop()
#     # result = await loop.run_in_executor(None, base_gen.generate_documentation, 
#     #                                    target_module_path, output_doc_path, ...)
    
#     result = base_gen.generate_documentation(
#         root_path_str=target_module_path,
#         output_file_str=output_doc_path,
#         include_patterns=["*.py"],
#         user_exclude_patterns=["*test*"] 
#     )
    
#     if result["status"] == "SUCCESS":
#         logger.info(f"BaseGenAgent created doc: {result['output_file']}")
#         # Markdown content can now be read from result['output_file']
#         # and fed to an LLM for analysis, summarization, or as context for code generation.
#         markdown_content = Path(result['output_file']).read_text()
#         # ... further processing by the calling agent ...
#     else:
#         logger.error(f"BaseGenAgent failed: {result['message']}")
```
The BaseGenAgent, with its externalized and modifiable configuration, becomes a more integral and evolvable part of the MindX ecosystem, providing a standardized way to snapshot and understand codebases.

BaseGenAgent (Codebase Documentation Generator) CLI Usage
The BaseGenAgent can be run from the command line to generate Markdown documentation for a specified codebase directory.
Basic Command Structure:
python mindx/tools/base_gen_agent.py <input_dir> [options]
Use code with caution.
Bash
Positional Arguments:
<input_dir> (Required)
Path to the codebase root directory you want to document.
Example: mindx/core or . (for the current directory).
Optional Arguments:
-o <output_filepath>, --output <output_filepath>
Specifies the full path (including filename) for the output Markdown file.
If not provided, a default path and filename are generated based on the input_dir name and settings in the agent's configuration (typically in PROJECT_ROOT/data/generated_docs/codebase_snapshots/).
Example: --output my_project_docs.md
Example: --output docs/api/utils_snapshot.md
--include <pattern_1> [<pattern_2> ...]
One or more glob patterns for files to explicitly include.
If this option is used, only files matching at least one of these patterns (and not excluded by other rules) will be processed.
Patterns are relative to the <input_dir>.
Examples:
--include "*.py" (include all Python files)
--include "src/**/*.js" "tests/*.py" (include JavaScript files in src subdirectories and Python files in tests)
--exclude <pattern_1> [<pattern_2> ...]
One or more glob patterns for files or directories to explicitly exclude.
These exclusions are applied in addition to .gitignore rules (if enabled) and hardcoded excludes from the agent's configuration.
Patterns are relative to the <input_dir>.
Examples:
--exclude "*.log" "temp_files/"
--exclude "**/__pycache__/*" "docs/*"
--no-gitignore
Action flag. If present, the agent will not process any .gitignore files found within the <input_dir> to determine exclusions.
By default (if this flag is absent), .gitignore files are respected.
--config-file <path/to/agent_config.json>
Specifies the path to a custom JSON configuration file for the BaseGenAgent.
If not provided, the agent attempts to load its configuration from a default path (typically PROJECT_ROOT/mindx/data/config/basegen_config.json). If that's also not found, it uses internal fallback defaults.
This allows you to have different sets of HARD_CODED_EXCLUDES, LANGUAGE_MAPPING, and other agent settings for different documentation tasks without modifying the default config file.
Example: --config-file ./configs/python_only_doc_config.json
--update-config '<json_string>'
Allows direct modification of the agent's persistent default configuration file (the one at self.agent_config_file_path, which is PROJECT_ROOT/mindx/data/config/basegen_config.json unless overridden by --config-file in the same command).
The <json_string> must be a valid JSON object string where keys are dot-separated paths to settings and values are the new settings.
When this option is used, the agent updates the specified config file and then exits. It does not generate documentation in the same run.
Special list operations for "HARD_CODED_EXCLUDES":
To append unique items: '{"HARD_CODED_EXCLUDES": [{"_LIST_OP_":"APPEND_UNIQUE"}, "*.newext1", "*.newext2"]}'
To remove items: '{"HARD_CODED_EXCLUDES": [{"_LIST_OP_":"REMOVE"}, "*.log", "*.tmp"]}'
To replace the list entirely: '{"HARD_CODED_EXCLUDES": ["*.onlythis", "*.andthis"]}'
To deep merge a dictionary (e.g., LANGUAGE_MAPPING or base_gen_agent_settings): '{"LANGUAGE_MAPPING": {".foo": "bar", "_MERGE_DEEP_": true}}'
Examples:
python mindx/tools/base_gen_agent.py --update-config '{"base_gen_agent_settings.max_file_size_kb_for_inclusion": 500}'
python mindx/tools/base_gen_agent.py --config-file ./special_config.json --update-config '{"LANGUAGE_MAPPING..newlang": "mylang"}' (updates ./special_config.json)
-h, --help
Shows a help message listing all arguments and their descriptions, then exits.
Output:
The CLI will print a JSON object to standard output summarizing the result of the operation (either documentation generation or config update). This JSON includes:
"status": "SUCCESS" or "FAILURE" (or "ERROR" for pre-execution issues, "FATAL_ERROR" for unexpected crashes).
"message": A human-readable summary.
"output_file": Path to the generated Markdown file (if documentation was generated).
"files_included": Number of files included in the documentation (if generated).
"error_type": (On fatal error) The type of Python exception.
Exit Codes:
0: Success.
1: Operational failure (e.g., could not generate docs as specified, config update failed).
2: Fatal error (e.g., unexpected Python exception during CLI setup).
Examples:
Generate documentation for the mindx/core directory, output to core_docs.md:
python mindx/tools/base_gen_agent.py mindx/core -o core_docs.md
Use code with caution.
Bash
Generate documentation for the current directory, including only .py and .md files, excluding anything in build/ directories, and ignoring .gitignore:
python mindx/tools/base_gen_agent.py . --include "*.py" "*.md" --exclude "build/*" --no-gitignore
Use code with caution.
Bash
Update the agent's default configuration to change the max file size and then generate docs using this new default:
Step 1: Update config
python mindx/tools/base_gen_agent.py --update-config '{"base_gen_agent_settings.max_file_size_kb_for_inclusion": 256}'
Use code with caution.
Bash
Step 2: Generate docs (will use the updated default config if basegen_config.json was the one modified)
python mindx/tools/base_gen_agent.py ./my_project
Use code with caution.
Bash
Generate docs using a completely custom configuration file for this run only:
python mindx/tools/base_gen_agent.py ./my_project --config-file ./configs/strict_python_only.json -o project_strict_py.md
Use code with caution.
Bash
This provides a clear set of commands for using the BaseGenAgent (Codebase Documentation Generator) from the command line.


#################


# mindX Tool: BaseGenAgent (Codebase Documentation Generator)

**Module Path:** `tools.base_gen_agent`
**Class Name:** `BaseGenAgent`

## Overview

The `BaseGenAgent` is a specialized tool within the mindX (Augmentic Intelligence) framework designed to automatically generate comprehensive Markdown documentation from a given codebase directory. This documentation includes a visual directory tree structure and the concatenated content of specified files.

Its primary purpose is to create a "snapshot" of a codebase that is easily digestible by Large Language Models (LLMs) for tasks such as:
*   Code understanding and analysis.
*   Assisting in autonomous code improvement suggestions.
*   Providing context for developing new features or tools related to the analyzed codebase.
*   Generating human-readable summaries or overviews.

The agent is configurable, respecting `.gitignore` files and allowing for custom include/exclude patterns to fine-tune the scope of the documentation.

## Key Features

*   **Directory Tree Generation:** Creates a text-based representation of the codebase's directory structure.
*   **File Content Inclusion:** Concatenates the content of selected files into the Markdown output.
*   **Language Guessing:** Attempts to guess the language of code files based on their extension to apply appropriate Markdown code block formatting.
*   **Exclusion Handling:**
    *   Respects `.gitignore` files found within the target directory and its subdirectories.
    *   Uses a configurable list of hard-coded exclude patterns (e.g., for binary files, temporary files, common metadata directories).
    *   Allows users to specify additional custom exclude patterns via CLI or programmatic calls.
*   **Inclusion Filtering:** Allows users to specify glob patterns for files that *should* be explicitly included. If provided, only files matching these patterns (and not otherwise excluded) will be processed.
*   **File Size Limiting:** Avoids including overly large files by adhering to a configurable maximum file size limit.
*   **Configurable Output:** Output filename and location can be specified or defaults to a structured path within the project.
*   **Configuration Management:**
    *   Loads its primary configuration from a dedicated JSON file (default: `PROJECT_ROOT/data/config/basegen_config.json`).
    *   Provides internal fallback defaults if the config file is missing or invalid.
    *   Supports overriding the config file path.
    *   Allows dynamic updates to its configuration settings programmatically or via a CLI flag.

## Configuration

The `BaseGenAgent` relies on a JSON configuration file. The primary default location for this file is `<PROJECT_ROOT>/data/config/basegen_config.json`. This path can be overridden during instantiation or via the `--config-file` CLI argument.

Key configurable sections within its expected config structure (often part of a larger system config like `basegen_config.json`):

*   `HARD_CODED_EXCLUDES`: A list of glob patterns for files and directories to always exclude (e.g., `*.pyc`, `__pycache__/`, `.git/`).
*   `LANGUAGE_MAPPING`: A dictionary mapping file extensions to language identifiers for Markdown code blocks (e.g., `".py": "python"`).
*   `base_gen_agent_settings`: An object containing settings specific to this agent:
    *   `max_file_size_kb_for_inclusion`: Maximum size (in KB) for a file's content to be included directly. Larger files will have their content omitted with a note. (Default: 1024KB)
    *   `default_output_filename_stem`: The base name for the output Markdown file if not specified. (Default: "codebase_snapshot")
    *   `output_subdir_relative_to_project`: Default subdirectory (relative to `PROJECT_ROOT`) where generated snapshots are saved. (Default: "data/generated_docs/codebase_snapshots")

### Example `basegen_config.json` snippet (relevant parts):
```json
{
  "HARD_CODED_EXCLUDES": [
    "*.pyc",
    "__pycache__/",
    ".git/"
  ],
  "LANGUAGE_MAPPING": {
    ".py": "python",
    ".js": "javascript",
    ".md": "markdown"
  },
  "base_gen_agent_settings": {
    "max_file_size_kb_for_inclusion": 512,
    "default_output_filename_stem": "my_project_summary",
    "output_subdir_relative_to_project": "docs/generated/code_summaries"
  }
}
Use code with caution.
Markdown
Programmatic Usage
The BaseGenAgent is typically instantiated and used within other mindX agents, like MastermindAgent.
from pathlib import Path
from tools.base_gen_agent import BaseGenAgent # Assuming 'tools' is a top-level package
# from utils.config import PROJECT_ROOT # If needed for paths

# Initialize the agent (it will load its config)
# The path to basegen_config.json is resolved internally by BaseGenAgent
# or can be overridden.
code_analyzer = BaseGenAgent()
# or with a specific config file for this instance:
# code_analyzer = BaseGenAgent(config_file_path_override_str="path/to/specific_bga_config.json")


target_directory_to_scan = "/path/to/your/codebase" # or str(PROJECT_ROOT / "your_package")
output_markdown_file = "/path/to/output/summary.md"

# Optional patterns
include_patterns = ["*.py", "*.json"]
user_exclude_patterns = ["*test*", "temp/"]

result = code_analyzer.generate_markdown_summary(
    root_path_str=target_directory_to_scan,
    output_file_str=output_markdown_file,
    include_patterns=include_patterns,
    user_exclude_patterns=user_exclude_patterns,
    use_gitignore=True # Default
)

if result["status"] == "SUCCESS":
    print(f"Successfully generated documentation: {result['output_file']}")
    print(f"Files included: {result['files_included']}")
else:
    print(f"Error generating documentation: {result['message']}")
Use code with caution.
Python
Key Method: generate_markdown_summary
root_path_str (str): The absolute or relative path to the root directory of the codebase to document.
output_file_str (Optional[str]): The path where the output Markdown file will be saved. If None, a default path is constructed based on configuration.
include_patterns (Optional[List[str]]): A list of glob patterns. If provided, only files matching these patterns (and not excluded) are included. If None (default), all non-excluded files are considered. An empty list [] means no files will be included unless exclusions make an exception.
user_exclude_patterns (Optional[List[str]]): A list of additional glob patterns to exclude files or directories, beyond .gitignore and hardcoded excludes.
use_gitignore (bool): Whether to respect .gitignore files found in the root_path_str and its subdirectories (Default: True).
Returns (Dict[str, Any]): A dictionary indicating the status:
{"status": "SUCCESS", "message": "...", "output_file": "...", "files_included": count}
{"status": "ERROR", "message": "...", "output_file": "...", "files_included": count}
Command-Line Interface (CLI) Usage
The base_gen_agent.py script can also be run directly from the command line.
Prerequisites:
Ensure Python environment is set up with pathspec library installed (pip install pathspec).
Ensure the script can find its dependencies (like utils.config). Typically, run from the project root: python tools/base_gen_agent.py ...
Syntax:
python tools/base_gen_agent.py <input_dir> [options]
Use code with caution.
Bash
Arguments:
input_dir: Path to the codebase root directory to document.
Options:
-o OUTPUT, --output OUTPUT: Output Markdown file path. Defaults to a generated name in a configured output directory.
--include PATT [PATT ...]: Glob pattern(s) for files to explicitly include (e.g., *.py src/**/*.js).
--exclude PATT [PATT ...]: Glob pattern(s) to explicitly exclude (e.g., */temp/* .DS_Store).
--no-gitignore: Disable applying .gitignore file exclusions.
--config-file FILE_PATH: Path to a custom agent JSON configuration file. Overrides the default config file path resolution.
--update-config JSON_STRING: JSON string of settings to update in the agent's used config file. This allows modifying the configuration (like HARD_CODED_EXCLUDES or max_file_size_kb_for_inclusion) before generation. The script will save the updated config and then exit if this option is used successfully without also generating docs.
Example: --update-config '{"base_gen_agent_settings.max_file_size_kb_for_inclusion": 512}'
To append to HARD_CODED_EXCLUDES: --update-config '{"HARD_CODED_EXCLUDES": [{"_LIST_OP_":"APPEND_UNIQUE"}, "*.newignore", "docs_temp/"]}'
To remove from HARD_CODED_EXCLUDES: --update-config '{"HARD_CODED_EXCLUDES": [{"_LIST_OP_":"REMOVE"}, "*.log"]}'
CLI Output:
The CLI prints a JSON object to standard output indicating the status of the operation (SUCCESS or ERROR), a message, the path to the output file, and the number of files included.
Example CLI Call:
# From your project root (/home/luvai/mindX/)
python tools/base_gen_agent.py ./core -o ./data/generated_docs/core_summary.md --include "*.py" --exclude "__pycache__/*"
Use code with caution.
Bash
This command would analyze the ./core directory, output to ./data/generated_docs/core_summary.md, only include Python files, and additionally exclude any __pycache__ contents (on top of .gitignore and hardcoded excludes).
Integration with MastermindAgent
The MastermindAgent can instantiate and use BaseGenAgent internally via its BDI action ANALYZE_CODEBASE_FOR_STRATEGY. This allows Mastermind to:
Programmatically trigger codebase analysis of specific target paths (either within the mindX project itself or external codebases).
Receive the generated Markdown summary.
Use its own LLM to interpret the summary based on a strategic focus (e.g., identifying modules for improvement, understanding APIs for tool extrapolation).
Store this analysis and interpretation in its belief system to inform future strategic decisions, such as initiating new tool development or component improvement campaigns.
This makes BaseGenAgent a crucial internal tool for Mastermind's self-understanding and its ability to strategically plan the evolution of the mindX system and its toolset.
