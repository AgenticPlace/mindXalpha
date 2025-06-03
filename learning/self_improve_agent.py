# mindx/learning/self_improve_agent.py
import os
import sys
import asyncio
import json
import shutil
import stat # For file permissions (use with caution, platform-dependent)
import tempfile
import difflib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Use canonical PROJECT_ROOT from config for consistency if needed for base data dirs
from mindx.utils.config import Config # PROJECT_ROOT can be accessed via config instance
from mindx.utils.logging_config import get_logger
from mindx.llm.llm_factory import create_llm_handler, LLMHandler

logger = get_logger(__name__)

# --- Constants ---
# AGENT_ROOT_DIR is the directory *containing this self_improve_agent.py script*
_CURRENT_FILE_PATH = Path(__file__).resolve()
AGENT_ROOT_DIR = _CURRENT_FILE_PATH.parent
SELF_AGENT_ABSOLUTE_PATH = _CURRENT_FILE_PATH # Crucial for self-identification
SELF_AGENT_FILENAME = SELF_AGENT_ABSOLUTE_PATH.name # e.g., "self_improve_agent.py"

# SELF_IMPROVEMENT_BASE_DIR: where this SIA stores its iterations, fallbacks, archive.
_temp_config_for_paths = Config(test_mode=True) # Allows re-init if Config already exists
_project_root_from_config_str = _temp_config_for_paths.get("PROJECT_ROOT")
if not _project_root_from_config_str: # Fallback if PROJECT_ROOT not in config yet
    _project_root_from_config = AGENT_ROOT_DIR.parent.parent # Heuristic: mindx/learning/ -> mindx/ -> project_root/
else:
    _project_root_from_config = Path(_project_root_from_config_str)
# SIA's data is stored within the project's data directory, namespaced by the agent script's name stem.
SELF_IMPROVEMENT_BASE_DIR = _project_root_from_config / "data" / "self_improvement_work_sia" / SELF_AGENT_FILENAME.stem
Config.reset_instance() # Clean up temp instance if it was created just for this

IMPROVEMENT_ARCHIVE_DIR = SELF_IMPROVEMENT_BASE_DIR / "archive"
IMPROVEMENT_LOG_FILE = IMPROVEMENT_ARCHIVE_DIR / "improvement_history.jsonl"
FALLBACK_DIR_NAME = "fallback_versions"
ITERATION_DIR_NAME_PREFIX = "iteration_"
DIR_PERMISSIONS_OWNER_ONLY = stat.S_IRWXU # 0700 for rwx by owner only


class SelfImprovementAgent:
    """
    Agent responsible for analyzing, implementing, and evaluating code improvements
    for a target Python file, including its own source code.
    It operates with safety mechanisms like iteration directories, self-tests,
    and fallbacks for self-updates. Designed to be CLI callable.
    """
    def __init__(
        self,
        agent_id: str = "self_improve_agent_v_prod_cand", # Version up
        llm_provider_override: Optional[str] = None,
        llm_model_name_override: Optional[str] = None,
        max_cycles_override: Optional[int] = None,
        self_test_timeout_override: Optional[float] = None,
        critique_threshold_override: Optional[float] = None,
        is_iteration_instance: bool = False, # True if this instance is running from an iteration dir
        config_override: Optional[Config] = None # For testing
    ):
        self.agent_id = agent_id
        self.config = config_override or Config() # Uses global singleton if not overridden
        self.python_executable = sys.executable # Use the same interpreter that's running this

        # Configuration precedence: CLI override > Config file value > Hardcoded default in Config
        llm_provider = llm_provider_override or self.config.get("self_improvement_agent.llm.provider")
        llm_model_name = llm_model_name_override or self.config.get("self_improvement_agent.llm.model")
        self.llm_handler: LLMHandler = create_llm_handler(llm_provider, llm_model_name)
        
        self.max_self_improve_cycles = max_cycles_override if max_cycles_override is not None \
            else self.config.get("self_improvement_agent.default_max_cycles", 1)
        
        self.self_test_timeout_seconds = self_test_timeout_override if self_test_timeout_override is not None \
            else self.config.get("self_improvement_agent.self_test_timeout_seconds", 180.0)
        
        self.critique_threshold = critique_threshold_override if critique_threshold_override is not None \
            else self.config.get("self_improvement_agent.critique_threshold", 0.6)

        self.is_iteration_instance = is_iteration_instance
        self.current_iteration_dir: Optional[Path] = None # Set during a self-improvement cycle

        self._ensure_directories_exist()
        logger.info(
            f"SIA '{self.agent_id}' initialized. LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default'}. "
            f"MaxCycles: {self.max_self_improve_cycles}. WorkDir: {SELF_IMPROVEMENT_BASE_DIR}. "
            f"Self-Test Timeout: {self.self_test_timeout_seconds}s. Critique Threshold: {self.critique_threshold}. "
            f"Is Iteration Instance: {self.is_iteration_instance}."
        )

    def _ensure_directories_exist(self): # pragma: no cover
        """Ensures base, archive, and fallback directories exist with owner rwx permissions."""
        dirs_to_create = [SELF_IMPROVEMENT_BASE_DIR, IMPROVEMENT_ARCHIVE_DIR, 
                          SELF_IMPROVEMENT_BASE_DIR / FALLBACK_DIR_NAME]
        for dir_path in dirs_to_create:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if os.name != 'nt': # chmod is not effective/same on Windows
                    try: os.chmod(dir_path, DIR_PERMISSIONS_OWNER_ONLY)
                    except OSError as e_perm: logger.warning(f"SIA: Could not set 0700 permissions for {dir_path}: {e_perm}.")
            except Exception as e:
                logger.error(f"SIA: Failed to create/prepare directory {dir_path}: {e}")

    def _get_file_content(self, file_path: Path) -> Optional[str]: # pragma: no cover
        """Safely get the content of a file."""
        try:
            return file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning(f"SIA: File not found for reading: {file_path}")
            return None
        except Exception as e:
            logger.error(f"SIA: Error reading file {file_path}: {e}", exc_info=True)
            return None

    def _save_file_content(self, file_path: Path, content: str) -> bool: # pragma: no cover
        """Safely save content to a file, creating parent directories if needed."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"SIA: Successfully saved updated file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"SIA: Error saving file {file_path}: {e}", exc_info=True)
            return False

    def _record_improvement_attempt(self, attempt_data: Dict[str, Any]): # pragma: no cover
        """Records an improvement attempt to the archive JSONL file."""
        serializable_data: Dict[str, Any] = {}
        for k, v in attempt_data.items(): # Ensure all Path objects are str for JSON
            if isinstance(v, Path): serializable_data[k] = str(v)
            elif isinstance(v, dict): serializable_data[k] = {sk: (str(sv) if isinstance(sv, Path) else sv) for sk, sv in v.items()}
            else: serializable_data[k] = v
        
        serializable_data["timestamp"] = datetime.utcnow().isoformat()
        serializable_data["agent_id"] = self.agent_id
        serializable_data["iteration_dir"] = str(self.current_iteration_dir) if self.current_iteration_dir else None
        try:
            IMPROVEMENT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            with IMPROVEMENT_LOG_FILE.open("a", encoding="utf-8") as f:
                json.dump(serializable_data, f); f.write("\n")
        except Exception as e:
            logger.error(f"SIA: Failed to record improvement attempt to {IMPROVEMENT_LOG_FILE}: {e}", exc_info=True)

    def _create_iteration_dir(self) -> Optional[Path]: # pragma: no cover
        """Creates a unique directory for a self-improvement iteration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        iteration_dir = SELF_IMPROVEMENT_BASE_DIR / f"{ITERATION_DIR_NAME_PREFIX}{timestamp}"
        try:
            iteration_dir.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt':
                try: os.chmod(iteration_dir, DIR_PERMISSIONS_OWNER_ONLY)
                except OSError as e_perm: logger.warning(f"SIA: Could not set permissions for iter dir {iteration_dir}: {e_perm}.")
            logger.info(f"SIA: Created iteration directory: {iteration_dir}")
            return iteration_dir
        except Exception as e:
            logger.error(f"SIA: Failed to create iteration directory {iteration_dir}: {e}", exc_info=True)
            return None

    def _backup_current_self(self, backup_name_suffix: str = "") -> Optional[Path]: # pragma: no cover
        """Backs up SELF_AGENT_ABSOLUTE_PATH to the fallback directory."""
        fallback_dir = SELF_IMPROVEMENT_BASE_DIR / FALLBACK_DIR_NAME
        self._ensure_directories_exist() 

        if not SELF_AGENT_ABSOLUTE_PATH.exists():
            logger.error(f"SIA: Main agent file {SELF_AGENT_ABSOLUTE_PATH} not found for backup.")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{SELF_AGENT_FILENAME}.{timestamp}{backup_name_suffix}.bak"
        backup_path = fallback_dir / backup_filename

        try:
            shutil.copy2(SELF_AGENT_ABSOLUTE_PATH, backup_path)
            logger.info(f"SIA: Backed up main agent ({SELF_AGENT_FILENAME}) to: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"SIA: Failed to backup main agent to {backup_path}: {e}", exc_info=True)
            return None
            
    def _get_latest_fallback_version(self) -> Optional[Path]: # pragma: no cover
        """Gets the Path of the most recent fallback version of this agent's script."""
        fallback_dir = SELF_IMPROVEMENT_BASE_DIR / FALLBACK_DIR_NAME
        if not fallback_dir.is_dir(): return None
        
        backup_files = sorted(
            [f for f in fallback_dir.iterdir() if f.is_file() and f.name.startswith(SELF_AGENT_FILENAME) and f.name.endswith(".bak")],
            key=lambda p: p.stat().st_mtime, reverse=True )
        return backup_files[0] if backup_files else None

    def _revert_to_fallback(self, fallback_path: Optional[Path] = None) -> bool: # pragma: no cover
        """Reverts SELF_AGENT_ABSOLUTE_PATH from a specified fallback or the latest."""
        actual_fallback_path = fallback_path or self._get_latest_fallback_version()

        if not actual_fallback_path or not actual_fallback_path.exists():
            logger.error("SIA: No fallback version available or specified path does not exist for revert.")
            return False

        try:
            SELF_AGENT_ABSOLUTE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(actual_fallback_path, SELF_AGENT_ABSOLUTE_PATH)
            logger.warning(f"SIA: Reverted main agent script ({SELF_AGENT_FILENAME}) from fallback: {actual_fallback_path.name}")
            return True
        except Exception as e:
            logger.critical(f"SIA CRITICAL FAILURE: Could not revert main agent script from fallback {actual_fallback_path.name}: {e}", exc_info=True)
            return False

    async def analyze_target(
        self,
        target_file_path: Path,
        additional_context: Optional[str] = None,
        logs_for_analysis: Optional[List[str]] = None,
        improvement_goal_hint: Optional[str] = None
    ) -> Optional[str]: # pragma: no cover
        """Analyzes target file to propose ONE improvement description. Returns description or error string."""
        logger.info(f"SIA: Analyzing target '{target_file_path.name}'. Goal hint: {improvement_goal_hint or 'General analysis'}")
        file_content = self._get_file_content(target_file_path)
        if file_content is None and target_file_path.exists():
             return "Error: SIA could not read target file content for analysis."
        file_content = file_content or ""

        max_content_chars = self.config.get("self_improvement_agent.analysis.max_code_chars", 70000)
        content_snippet = file_content
        if len(file_content) > max_content_chars:
            content_snippet = file_content[:max_content_chars//2] + f"\n\n... (code truncated - original {len(file_content)} chars) ...\n\n" + file_content[-max_content_chars//2:]
            logger.warning(f"SIA: Target content for '{target_file_path.name}' truncated for analysis prompt.")

        context_parts = [f"Target Python file ('{target_file_path.name}') content (may be empty or truncated):\n```python\n{content_snippet}\n```"]
        if improvement_goal_hint: context_parts.append(f"\nSpecific Improvement Goal/Hint: {improvement_goal_hint}")
        if additional_context: context_parts.append(f"\nAdditional Context (truncated):\n{additional_context[:7000]}")
        if logs_for_analysis: context_parts.append(f"\nRelevant Logs Snippet (truncated):\n{('\n'.join(logs_for_analysis))[:7000]}")

        prompt = (
            "You are an AI specializing in Python code improvement (MindX System, Augmentic Project). Analyze the provided code and context. "
            "Identify ONE concrete, actionable, and implementable improvement for quality, performance, robustness, maintainability, or a specific goal.\n\n"
            f"{''.join(context_parts)}\n\n"
            "Describe this single improvement clearly: WHAT needs to change and WHY. This description will generate code. Avoid vagueness. "
            "Example: 'In `MyClass.process_data`, add error handling for `KeyError` when accessing `data['field']` and return `None` instead.'\n\n"
            "Proposed Improvement Description:"
        )
        try:
            max_desc_tokens = self.config.get("self_improvement_agent.analysis.max_description_tokens", 350)
            response_text = await self.llm_handler.generate_text(prompt, max_tokens=max_desc_tokens, temperature=0.2)
            if not response_text or response_text.startswith("Error:"):
                 logger.error(f"SIA: LLM analysis for '{target_file_path.name}' error/empty: {response_text}")
                 return f"Error: LLM analysis failed - {response_text}"
            description = response_text.strip()
            logger.info(f"SIA: LLM proposed for '{target_file_path.name}': {description}")
            return description
        except Exception as e:
            logger.error(f"SIA: LLM analysis call exception for '{target_file_path.name}': {e}", exc_info=True)
            return f"Error: LLM analysis exception - {type(e).__name__}: {e}"

    async def implement_improvement(
        self,
        target_file_path: Path,
        improvement_description: str,
        original_content: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]: # (success, new_content_or_error_msg, diff_patch_str)
        """Implements the improvement. Returns success, new content/error, and diff."""
        logger.info(f"SIA: Implementing for '{target_file_path.name}': {improvement_description[:120]}...")
        if original_content is None:
            original_content = self._get_file_content(target_file_path)
            if original_content is None and target_file_path.exists(): # File should exist or be new
                return False, f"Error: Could not read original content of existing file {target_file_path.name} for implementation.", None
            original_content = original_content or "" # If None (new file), treat as empty string

        prompt = (
            f"You are an AI coding assistant. Modify/create the Python code for file '{target_file_path.name}' "
            f"to implement ONLY this specific improvement goal:\n'{improvement_description}'\n\n"
            f"Current full code of `{target_file_path.name}` (this might be empty if it's a new file to be created based on the description):\n"
            f"```python\n{original_content}\n```\n\n"
            "IMPORTANT: Provide ONLY the complete, new Python code for the entire file '{target_file_path.name}'. "
            "Do NOT include any explanations, comments outside the code, or markdown code fences (like ```python or ```) at the beginning or end of the code block. "
            "Your entire response must be just the raw Python code."
        )
        modified_code_final: Optional[str] = None
        try:
            max_code_gen_tokens = self.config.get("self_improvement_agent.implementation.max_code_gen_tokens", 12000)
            temperature_code_gen = self.config.get("self_improvement_agent.implementation.temperature", 0.1) # Very low for precision
            modified_code_raw = await self.llm_handler.generate_text(prompt, temperature=temperature_code_gen, max_tokens=max_code_gen_tokens)
            
            if not modified_code_raw or modified_code_raw.startswith("Error:"):
                logger.error(f"SIA: LLM code generation returned error/empty: {modified_code_raw}")
                return False, f"LLM code generation failed: {modified_code_raw}", None
            
            modified_code_final = modified_code_raw.strip()
            # Robust cleaning of markdown code fences
            if modified_code_final.startswith("```python"): modified_code_final = modified_code_final[len("```python"):].strip()
            elif modified_code_final.startswith("```"): modified_code_final = modified_code_final[len("```"):].strip()
            if modified_code_final.endswith("```"): modified_code_final = modified_code_final[:-len("```")].strip()

            if not modified_code_final.strip() and original_content.strip():
                 logger.warning(f"SIA: LLM generated empty code for {target_file_path.name} when original was not. Failing implementation.")
                 return False, "LLM generated empty code for a previously non-empty file.", None
            if not modified_code_final.strip() and not original_content.strip():
                 logger.info(f"SIA: LLM generated empty code for new file {target_file_path.name}. This is allowed, evaluation will check if it's correct.")
                 # This is fine if the goal was e.g. "create an empty __init__.py"

            diff_patch_str = "No changes detected (content identical)."
            if original_content != modified_code_final:
                diff = difflib.unified_diff(
                    original_content.splitlines(keepends=True), modified_code_final.splitlines(keepends=True),
                    fromfile=f"a/{target_file_path.name}", tofile=f"b/{target_file_path.name}", lineterm='' )
                diff_patch_str = "".join(list(diff))
                if not diff_patch_str: # If unified_diff is empty but strings differ (e.g. only EOL or trailing whitespace)
                     diff_patch_str = "Content changed (likely only whitespace or EOL differences not captured by standard unified_diff without lineterm='')."
            
            if self._save_file_content(target_file_path, modified_code_final):
                logger.info(f"SIA: Applied LLM changes to {target_file_path} (in working dir). Diff status: {'Changes' if diff_patch_str != 'No changes detected (content identical).' else 'No changes'}")
                return True, modified_code_final, diff_patch_str
            else: # pragma: no cover
                return False, f"Failed to save modified code to {target_file_path}.", diff_patch_str
        except Exception as e: # pragma: no cover
            logger.error(f"SIA: Code generation/saving exception for {target_file_path.name}: {e}", exc_info=True)
            return False, f"Exception during code generation/saving: {type(e).__name__}: {e}", None

    async def _run_self_test_suite(self, iteration_agent_path: Path) -> Tuple[bool, str]: # pragma: no cover
        """Runs self-tests on iteration_agent_path. Returns (passed, full_output_str)."""
        logger.info(f"SIA: Running self-test suite for candidate agent: {iteration_agent_path}")
        cmd = [
            self.python_executable, str(iteration_agent_path), "--self-test-mode",
            "--llm-provider", self.llm_handler.provider_name,
            "--llm-model", self.llm_handler.model_name or "", # Ensure string
            "--cycles", str(self.config.get("self_improvement_agent.default_max_cycles",1)), # Pass a default
            "--self-test-timeout", str(self.self_test_timeout_seconds), 
            "--critique-threshold", str(self.critique_threshold)
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=AGENT_ROOT_DIR # Run self-test from where the main agent would run
            )
            effective_timeout = self.self_test_timeout_seconds + 20.0 # Add buffer for process overhead
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=effective_timeout)
            
            out_s = stdout.decode(errors='ignore').strip(); err_s = stderr.decode(errors='ignore').strip()
            full_output_str = f"SELF-TEST STDOUT:\n{out_s}\n\nSELF-TEST STDERR:\n{err_s}"
            logger.debug(f"SIA Self-Test Full Output ({iteration_agent_path.name}):\n{full_output_str[:2000]}...")

            if process.returncode == 0 and out_s:
                try:
                    test_json_result = json.loads(out_s)
                    if test_json_result.get("status") == "SUCCESS" and \
                       ("SELF TEST PASSED" in test_json_result.get("message", "") or \
                        any("SELF TEST PASSED" in m for m in test_json_result.get("data",{}).get("details",[]))):
                        logger.info(f"SIA Self-Test Suite PASSED for {iteration_agent_path.name}. Message: {test_json_result.get('message')}")
                        return True, full_output_str
                    else:
                        logger.warning(f"SIA Self-Test Suite FAILED (reported by test script JSON) for {iteration_agent_path.name}. Message: {test_json_result.get('message')}")
                except json.JSONDecodeError:
                    logger.error(f"SIA Self-Test for {iteration_agent_path.name}: STDOUT was not valid JSON despite exit code 0. Output: {out_s}")
            
            logger.warning(f"SIA Self-Test Suite FAILED for {iteration_agent_path.name}. Process Return Code: {process.returncode}.")
            return False, full_output_str
        except asyncio.TimeoutError:
            logger.error(f"SIA Self-Test Suite for {iteration_agent_path.name} TIMED OUT after {effective_timeout}s.")
            return False, "Self-test suite execution timed out."
        except Exception as e:
            logger.error(f"SIA Self-Test Suite for {iteration_agent_path.name} encountered an EXCEPTION: {e}", exc_info=True)
            return False, f"Exception during self-test suite execution: {type(e).__name__}: {e}"

    async def evaluate_improvement(
        self,
        target_file_path: Path, # Absolute path to the modified file
        old_content: Optional[str],
        new_content: str,
        improvement_description: str,
        is_self_improvement: bool = False
    ) -> Dict[str, Any]: # pragma: no cover
        """Evaluates the implemented improvement."""
        logger.info(f"SIA: Evaluating improvement for '{target_file_path.name}' (Self-Improve: {is_self_improvement})")
        eval_results: Dict[str, Any] = {
            "passed_syntax_check": "NOT_RUN",
            "passed_self_tests": "N/A" if not is_self_improvement else "NOT_RUN",
            "llm_self_critique_score": 0.0, # Default to 0.0
            "notes": "Evaluation initiated."
        }

        # 1. Syntax Check
        try:
            compile(new_content, str(target_file_path), 'exec')
            eval_results["passed_syntax_check"] = True
            eval_results["notes"] = "Python syntax check passed."
        except SyntaxError as se:
            logger.warning(f"SIA Eval: SyntaxError in new content for {target_file_path.name}: line {se.lineno}, offset {se.offset}: {se.msg}")
            eval_results["passed_syntax_check"] = False
            eval_results["notes"] = f"SyntaxError: line {se.lineno}, offset {se.offset}: {se.msg}"
            return eval_results # Fail fast

        # 2. Self-Tests (only for self-improvement attempts)
        if is_self_improvement:
            passed_st, st_output_summary = await self._run_self_test_suite(target_file_path) # target_file_path is the iteration script
            eval_results["passed_self_tests"] = passed_st
            eval_results["notes"] += f"\nSelf-Test Result: {'PASSED' if passed_st else 'FAILED'}. Output summary (truncated):\n{st_output_summary[:500]}..."
            if not passed_st:
                logger.warning(f"SIA Eval: Self-tests FAILED for candidate {target_file_path.name}.")
                eval_results["llm_self_critique_score"] = 0.0 # Force low score if self-tests fail
                return eval_results # Fail fast

        # 3. LLM Self-Critique
        run_critique = eval_results["passed_syntax_check"] and \
                       (eval_results["passed_self_tests"] if is_self_improvement else True)

        if run_critique:
            max_chars_critique = self.config.get("self_improvement_agent.evaluation.max_chars_for_critique", 4000)
            old_snippet = old_content[:max_chars_critique//2] if old_content else 'N/A (New file or no original content)'
            new_snippet = new_content[:max_chars_critique]

            critique_prompt = (
                f"Review an AI-generated code change. Goal: '{improvement_description}'\n"
                f"Old Code Snippet:\n```python\n{old_snippet}\n```\n"
                f"New Code Snippet:\n```python\n{new_snippet}\n```\n"
                "Assess how well 'New Code' achieves 'Goal' (0.0-1.0). Consider correctness, completeness, side effects. "
                "Respond ONLY in JSON: {\"score\": float, \"justification\": \"string (brief explanation)\"}."
            )
            try:
                max_crit_tokens = self.config.get("self_improvement_agent.evaluation.max_critique_tokens", 300)
                critique_response_str = await self.llm_handler.generate_text(critique_prompt, max_tokens=max_crit_tokens, temperature=0.0, json_mode=True)
                parsed_critique = {}
                if critique_response_str and not critique_response_str.startswith("Error:"):
                    try: parsed_critique = json.loads(critique_response_str)
                    except json.JSONDecodeError:
                        match = re.search(r"\{[\s\S]*\}", critique_response_str)
                        if match: parsed_critique = json.loads(match.group(0))
                        else: logger.warning(f"SIA Eval: LLM critique not valid JSON: {critique_response_str[:200]}")
                
                critique_score = float(parsed_critique.get("score", 0.0)); critique_score = max(0.0, min(1.0, critique_score))
                critique_just = parsed_critique.get("justification", "No justification from LLM critique.")
                eval_results["llm_self_critique_score"] = critique_score
                eval_results["notes"] += f"\nLLM Critique (Score: {critique_score:.2f}): {critique_just}"
            except Exception as e_crit:
                logger.error(f"SIA Eval: LLM self-critique call exception for {target_file_path.name}: {e_crit}", exc_info=True)
                eval_results["notes"] += f"\nLLM Critique processing failed: {e_crit}"
        else:
            eval_results["notes"] += "\nLLM Critique skipped due to prior critical evaluation failures."

        logger.info(f"SIA Eval complete for '{target_file_path.name}': Score={eval_results['llm_self_critique_score']:.2f}, SyntaxOK={eval_results['passed_syntax_check']}, SelfTestsOK={eval_results['passed_self_tests']}")
        return eval_results

    async def run_self_improvement_cycle(
        self,
        target_file_path_conceptual: Path,
        initial_analysis_context: Optional[str] = None,
        logs_for_analysis: Optional[List[str]] = None,
        improvement_goal_override: Optional[str] = None
    ) -> Dict[str, Any]: # pragma: no cover
        """Runs one full cycle: Analyze -> Implement -> Evaluate (-> Promote for self)."""
        is_self_improvement_attempt = (target_file_path_conceptual.resolve() == SELF_AGENT_ABSOLUTE_PATH)
        logger.info(f"SIA Cycle Start: Target='{target_file_path_conceptual.name}', Self-Improve={is_self_improvement_attempt}, GoalOverride='{bool(improvement_goal_override)}'")
        
        cycle_result: Dict[str, Any] = {
            "target_file_path_conceptual": str(target_file_path_conceptual), "is_self_improvement_attempt": is_self_improvement_attempt,
            "effective_target_path": None, "improvement_description": None, "implementation_status": "PENDING",
            "new_content": None, "diff_patch": None, "evaluation": None, "error_message": None,
            "promoted_to_main": False if is_self_improvement_attempt else "N/A", "code_updated_requires_restart": False
        }
        
        self.current_iteration_dir = None; effective_target_path: Path
        original_main_agent_content_for_revert: Optional[str] = None

        if is_self_improvement_attempt:
            if self.is_iteration_instance: # Safety check
                err_msg = "SIA CRITICAL: Recursive self-improvement call by iteration instance."; logger.fatal(err_msg)
                cycle_result.update({"error_message": err_msg, "implementation_status": "FAILED_RECURSION_GUARD"}); self._record_improvement_attempt(cycle_result); return cycle_result
            self.current_iteration_dir = self._create_iteration_dir()
            if not self.current_iteration_dir: cycle_result.update({"error_message": "Failed iteration dir create.", "implementation_status": "FAILED_SETUP"}); self._record_improvement_attempt(cycle_result); return cycle_result
            original_main_agent_content_for_revert = self._get_file_content(SELF_AGENT_ABSOLUTE_PATH)
            if original_main_agent_content_for_revert is None: cycle_result.update({"error_message": f"Could not read own source: {SELF_AGENT_ABSOLUTE_PATH}.", "implementation_status": "FAILED_SETUP"}); self._record_improvement_attempt(cycle_result); return cycle_result
            effective_target_path = self.current_iteration_dir / SELF_AGENT_FILENAME
            try: shutil.copy2(SELF_AGENT_ABSOLUTE_PATH, effective_target_path); logger.info(f"SIA: Copied self to iter dir: {effective_target_path}")
            except Exception as e_cp: cycle_result.update({"error_message": f"Failed copy to iter dir: {e_cp}", "implementation_status": "FAILED_SETUP"}); self._record_improvement_attempt(cycle_result); return cycle_result
        else: effective_target_path = target_file_path_conceptual.resolve()
        cycle_result["effective_target_path"] = str(effective_target_path)

        # 1. Analyze
        actual_improvement_desc = improvement_goal_override or await self.analyze_target(effective_target_path, initial_analysis_context, logs_for_analysis, improvement_goal_override)
        if not actual_improvement_desc or actual_improvement_desc.startswith("Error:"): cycle_result.update({"error_message": f"Analysis fail: {actual_improvement_desc}", "implementation_status": "FAILED_ANALYSIS"}); self._record_improvement_attempt(cycle_result); return cycle_result
        cycle_result["improvement_description"] = actual_improvement_desc
        
        original_content_for_impl = self._get_file_content(effective_target_path)
        if original_content_for_impl is None and effective_target_path.exists(): cycle_result.update({"error_message": f"Failed read working file {effective_target_path.name}", "implementation_status": "FAILED_SETUP"}); self._record_improvement_attempt(cycle_result); return cycle_result
        original_content_for_impl = original_content_for_impl or ""

        # 2. Implement
        implemented, new_content_or_err, diff = await self.implement_improvement(effective_target_path, actual_improvement_desc, original_content_for_impl)
        cycle_result["diff_patch"] = diff
        if not implemented: 
            cycle_result.update({"error_message": new_content_or_err, "implementation_status": "FAILED_IMPLEMENTATION"})
            if self._get_file_content(effective_target_path) != original_content_for_impl: self._save_file_content(effective_target_path, original_content_for_impl) # Revert working copy
            self._record_improvement_attempt(cycle_result); return cycle_result
        cycle_result["new_content"] = new_content_or_err; cycle_result["implementation_status"] = "SUCCESS_IMPLEMENTED"

        # 3. Evaluate
        eval_data = await self.evaluate_improvement(effective_target_path, original_content_for_impl, cycle_result["new_content"], actual_improvement_desc, is_self_improvement_attempt)
        cycle_result["evaluation"] = eval_data
        eval_passed = eval_data.get("passed_syntax_check", False) and \
                      (eval_data.get("passed_self_tests", True) if is_self_improvement_attempt else True) and \
                      eval_data.get("llm_self_critique_score", 0.0) >= self.critique_threshold

        if not eval_passed:
            cycle_result.update({"implementation_status": "FAILED_EVALUATION", "error_message": f"Evaluation failed. Details: {eval_data.get('notes', 'See evaluation data.')}"})
            if self._save_file_content(effective_target_path, original_content_for_impl): logger.info(f"SIA: Reverted working file {effective_target_path.name} due to poor eval.")
            else: cycle_result["error_message"] += " CRITICAL: Failed to revert working file."
            self._record_improvement_attempt(cycle_result); return cycle_result
        cycle_result["implementation_status"] = "SUCCESS_EVALUATED"

        # 4. Promote (only for self-improvement)
        if is_self_improvement_attempt: # And eval_passed is True
            logger.info(f"SIA: Self-improve candidate {effective_target_path.name} passed. Promoting.")
            backup_p = self._backup_current_self(f"_promo_{self.current_iteration_dir.name if self.current_iteration_dir else 'unk'}")
            if not backup_p: cycle_result.update({"error_message": "Backup fail before promo.", "implementation_status": "FAILED_PROMOTION_NO_BACKUP"}); self._record_improvement_attempt(cycle_result); return cycle_result
            try:
                shutil.copy2(effective_target_path, SELF_AGENT_ABSOLUTE_PATH); logger.warning(f"SIA: PROMOTED self-improvement from {effective_target_path.name} to {SELF_AGENT_FILENAME}. RESTART REQUIRED.")
                cycle_result.update({"promoted_to_main": True, "implementation_status": "SUCCESS_PROMOTED", "code_updated_requires_restart": True})
            except Exception as e_promo: # pragma: no cover
                logger.critical(f"SIA CRITICAL: Failed to promote {effective_target_path.name}: {e_promo}", exc_info=True)
                cycle_result.update({"promoted_to_main": False, "error_message": f"Promo fail: {e_promo}", "implementation_status": "FAILED_PROMOTION_COPY_ERROR"})
                if self._revert_to_fallback(backup_p): logger.info("SIA: Main agent restored from immediate backup after promo fail.")
                elif original_main_agent_content_for_revert and self._save_file_content(SELF_AGENT_ABSOLUTE_PATH, original_main_agent_content_for_revert): logger.info("SIA: Main agent restored to pre-cycle state after promo+backup_revert fail.")
                else: logger.critical("SIA: TRIPLE CRITICAL: PROMO FAIL, BACKUP REVERT FAIL, PRE-CYCLE REVERT FAIL. MANUAL INTERVENTION NEEDED.")
        
        self._record_improvement_attempt(cycle_result)
        return cycle_result

    async def improve_self(
        self, max_cycles_override: Optional[int] = None, initial_analysis_context: Optional[str] = None,
        logs_for_analysis: Optional[List[str]] = None, improvement_goal_override: Optional[str] = None
    ) -> Dict[str, Any]: # pragma: no cover
        """Orchestrates multiple self-improvement cycles for this agent's own code."""
        logger.info(f"SIA: Commencing self-improvement campaign for '{SELF_AGENT_FILENAME}'...")
        max_cycles = max_cycles_override if max_cycles_override is not None else self.max_self_improve_cycles
        overall_res = { "agent_id": self.agent_id, "operation_type": "improve_self", "target_script_path": str(SELF_AGENT_ABSOLUTE_PATH),
                        "cycles_configured": max_cycles, "cycles_attempted": 0, "cycles_promoted": 0,
                        "final_status": "NO_CYCLES_RUN", "message": "Self-improvement initiated.", "cycle_results": [] }
        current_context = initial_analysis_context; current_goal = improvement_goal_override
        for i in range(max_cycles):
            logger.info(f"SIA: Self-improve campaign cycle {i+1}/{max_cycles} for {SELF_AGENT_FILENAME}")
            overall_res["cycles_attempted"] += 1
            cycle_data = await self.run_self_improvement_cycle( SELF_AGENT_ABSOLUTE_PATH, current_context, logs_for_analysis if i==0 else None, current_goal)
            overall_res["cycle_results"].append(cycle_data); overall_res["final_status"] = cycle_data.get("implementation_status", "ERR_STAT")
            overall_res["message"] = f"Cycle {i+1}: {overall_res['final_status']}. Desc: {cycle_data.get('improvement_description', 'N/A')[:80]}"
            if cycle_data.get("promoted_to_main"):
                overall_res["cycles_promoted"] += 1; logger.warning(overall_res["message"] + " Restart advised.")
                current_context = f"Promo success: '{cycle_data.get('improvement_description', 'N/A')[:70]}...'. Code changed. Re-eval."
                current_goal = None # Reset specific goal
                # Could break here to force restart if desired
            elif overall_res["final_status"].startswith("FAILED"): logger.warning(overall_res["message"] + " Halting campaign."); break
            else: current_context = f"Last cycle ({cycle_data.get('improvement_description', 'N/A')[:70]}) status {overall_res['final_status']}. Next."; current_goal = None
        if overall_res["cycles_attempted"] == 0: overall_res["message"] = "No self-improve cycles run."
        elif overall_res["cycles_promoted"] > 0: overall_res["message"] = f"Self-improve campaign: {overall_res['cycles_promoted']} update(s) PROMOTED. RESTART CRITICAL."
        else: overall_res["message"] = f"Self-improve campaign: No updates promoted. Last cycle status: {overall_res['final_status']}"
        return overall_res

    async def improve_external_target(
        self, target_file_path: Path, max_cycles_override: Optional[int] = None,
        context: Optional[str] = None, logs: Optional[List[str]] = None, improvement_goal_override: Optional[str] = None
    ) -> Dict[str, Any]: # pragma: no cover
        """Improves an external target Python file, running multiple cycles."""
        logger.info(f"SIA: Commencing improvement campaign for external target: {target_file_path}...")
        if not target_file_path.is_absolute(): return {"final_status": "ERROR_BAD_PATH", "message": "Target path must be absolute."}
        max_cycles = max_cycles_override if max_cycles_override is not None else self.max_self_improve_cycles
        overall_res = { "agent_id": self.agent_id, "operation_type": "improve_external_target", "target_file_path": str(target_file_path),
                        "cycles_configured": max_cycles, "cycles_attempted": 0, "cycles_successful_eval": 0,
                        "final_status": "NO_CYCLES_RUN", "message": "External improvement initiated.", "cycle_results": []}
        current_context = context; current_goal = improvement_goal_override
        for i in range(max_cycles):
            logger.info(f"SIA: External improve cycle {i+1}/{max_cycles} for {target_file_path.name}")
            overall_res["cycles_attempted"] += 1
            cycle_data = await self.run_self_improvement_cycle(target_file_path, current_context, logs if i==0 else None, current_goal)
            overall_res["cycle_results"].append(cycle_data); overall_res["final_status"] = cycle_data.get("implementation_status", "ERR_STAT")
            overall_res["message"] = f"Cycle {i+1}: {overall_res['final_status']}. Desc: {cycle_data.get('improvement_description', 'N/A')[:80]}"
            if cycle_data.get("implementation_status") == "SUCCESS_EVALUATED":
                overall_res["cycles_successful_eval"] += 1
                current_context = f"Success: '{cycle_data.get('improvement_description', 'N/A')[:70]}...'. Target file changed. Re-eval for next."
                current_goal = None
            elif overall_res["final_status"].startswith("FAILED"): logger.warning(overall_res["message"] + " Halting campaign."); break
            else: current_context = f"Last cycle ({cycle_data.get('improvement_description', 'N/A')[:70]}) status {overall_res['final_status']}. Next."; current_goal = None
        if overall_res["cycles_attempted"] == 0: overall_res["message"] = "No external improve cycles run."
        elif overall_res["cycles_successful_eval"] > 0: overall_res["message"] = f"External improve campaign: {overall_res['cycles_successful_eval']} change(s) successfully evaluated."
        else: overall_res["message"] = f"External improve campaign: No changes successfully evaluated. Last status: {overall_res['final_status']}"
        return overall_res

async def main_cli(): # pragma: no cover
    """CLI entry point for the SelfImprovementAgent."""
    import argparse
    parser = argparse.ArgumentParser(
        description="MindX SelfImprovementAgent CLI. Modifies Python code. Always outputs JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument( "target_file", help="Path to Python file or 'self' for self-improvement.")
    parser.add_argument( "--context", default=None, help="Textual context for analysis.")
    parser.add_argument( "--context-file", type=Path, default=None, help="Path to file containing context.")
    parser.add_argument( "--logs", nargs="*", default=[], help="Paths to log files for context.")
    parser.add_argument( "--llm-provider", default=None, help="Override LLM provider.")
    parser.add_argument( "--llm-model", default=None, help="Override LLM model name.")
    parser.add_argument( "--cycles", type=int, default=None, help="Override number of improvement cycles.")
    parser.add_argument( "--self-test-timeout", type=float, default=None, help="Override self-test timeout seconds.")
    parser.add_argument( "--critique-threshold", type=float, default=None, help="Override critique score threshold (0.0-1.0).")
    parser.add_argument( "--self-test-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument( "--output-json", action="store_true", help="Output minified JSON (default is indented).")

    final_output_payload: Dict[str, Any] = {"status": "FAILURE", "message": "SIA CLI did not initialize correctly.", "data": {}}
    exit_code = 1
    args = None # Initialize args to None

    try:
        args = parser.parse_args()

        if args.self_test_mode:
            logger.info("SIA: Running in self-test mode via CLI.")
            test_ok = True; test_msgs = ["Self-test mode activated."]
            if not SELF_AGENT_ABSOLUTE_PATH.exists() or SELF_AGENT_FILENAME != "self_improve_agent.py": test_msgs.append(f"ERR: Path consts bad."); test_ok = False
            try: SELF_IMPROVEMENT_BASE_DIR.mkdir(parents=True, exist_ok=True); (SELF_IMPROVEMENT_BASE_DIR / "st.tmp").write_text("t"); (SELF_IMPROVEMENT_BASE_DIR / "st.tmp").unlink(); test_msgs.append("Dir write OK.")
            except Exception as e_st_dir: test_msgs.append(f"ERR: Dir write fail {SELF_IMPROVEMENT_BASE_DIR}: {e_st_dir}"); test_ok = False
            final_output_payload = {"status": "SUCCESS" if test_ok else "FAILURE", "message": "SELF TEST " + ("PASSED" if test_ok else "FAILED"), "data": {"details": test_msgs}}; exit_code = 0 if test_ok else 1
            print(json.dumps(final_output_payload, indent=2 if not args.output_json else None)); sys.exit(exit_code)

        ctx_str: Optional[str] = args.context
        if args.context_file:
            if args.context_file.exists() and args.context_file.is_file():
                try: ctx_str = args.context_file.read_text(encoding="utf-8"); logger.info(f"SIA CLI: Loaded context from: {args.context_file}")
                except Exception as e_f: final_output_payload = {"status": "FAILURE", "message": f"Err reading context file {args.context_file}: {e_f}", "data": {"err_type": "InputFileError"}}; raise SystemExit(1)
            else: final_output_payload = {"status": "FAILURE", "message": f"Context file not found: {args.context_file}", "data": {"err_type": "InputFileNotFound"}}; raise SystemExit(1)
        
        agent = SelfImprovementAgent( llm_provider_override=args.llm_provider, llm_model_name_override=args.llm_model, max_cycles_override=args.cycles, self_test_timeout_override=args.self_test_timeout, critique_threshold_override=args.critique_threshold )
        logs_list: List[str] = []
        if args.logs:
            for lp_str in args.logs:
                try: logs_list.append(Path(lp_str).read_text(encoding="utf-8"))
                except Exception as e_l: logger.warning(f"SIA CLI: Fail read log {lp_str}: {e_l}")
        
        sia_op_data: Dict[str, Any]
        if args.target_file.lower() == "self":
            sia_op_data = await agent.improve_self( max_cycles_override=args.cycles, initial_analysis_context=ctx_str, logs_for_analysis=logs_list if logs_list else None )
        else:
            target_p_abs = Path(args.target_file).resolve()
            # SIA's internal logic will check if target_p_abs exists if not a creation task
            sia_op_data = await agent.improve_external_target( target_p_abs, max_cycles_override=args.cycles, context=ctx_str, logs=logs_list if logs_list else None )
        
        final_status = sia_op_data.get("final_status", "UNKNOWN_SIA_OP_STATUS")
        final_output_payload["status"] = "SUCCESS" if final_status.startswith("SUCCESS") else "FAILURE"
        final_output_payload["message"] = sia_op_data.get("message", f"SIA op done. Internal status: {final_status}")
        final_output_payload["data"] = sia_op_data
        exit_code = 0 if final_output_payload["status"] == "SUCCESS" else 1

    except SystemExit as e_sys: # To allow sys.exit() to propagate from self-test or arg parsing errors
        # If final_output_payload was set before SystemExit, use it. Otherwise, it means arg parsing failed.
        if final_output_payload["message"] == "SIA CLI did not initialize correctly.": # Default message
             final_output_payload["message"] = f"SIA CLI Error: Argument parsing or setup failed. SystemExit code: {e_sys.code}"
        exit_code = e_sys.code if isinstance(e_sys.code, int) else 1 # Ensure int
    except Exception as e_top:
        logger.critical("SIA CLI: Top-level unhandled exception.", exc_info=True)
        final_output_payload["status"] = "FAILURE"
        final_output_payload["message"] = f"Critical SIA CLI error: {type(e_top).__name__}: {str(e_top)}"
        final_output_payload["data"] = {"error_type": type(e_top).__name__, "traceback_snippet": traceback.format_exc(limit=3).splitlines()}
        exit_code = 1
    finally:
        should_minify = getattr(args, "output_json", False) if args else False # getattr handles if args is None
        print(json.dumps(final_output_payload, indent=2 if not should_minify else None))
        sys.exit(exit_code)

if __name__ == "__main__": # pragma: no cover
    Config() # Ensure global Config is loaded, which loads .env for standalone SIA.
    asyncio.run(main_cli())
