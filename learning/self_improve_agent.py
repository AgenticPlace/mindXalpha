# mindx/learning/self_improve_agent.py
import os
import sys
import asyncio
import json
import shutil
import stat 
import tempfile
import difflib
import re # For robust JSON parsing in critique
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from mindx.utils.config import Config
from mindx.utils.logging_config import get_logger
from mindx.llm.llm_factory import create_llm_handler, LLMHandler

logger = get_logger(__name__)

# --- Constants ---
_CURRENT_FILE_PATH = Path(__file__).resolve()
AGENT_ROOT_DIR = _CURRENT_FILE_PATH.parent
SELF_AGENT_ABSOLUTE_PATH = _CURRENT_FILE_PATH 
SELF_AGENT_FILENAME = SELF_AGENT_ABSOLUTE_PATH.name

_temp_config_for_paths = Config(test_mode=True)
_project_root_from_config_str = _temp_config_for_paths.get("PROJECT_ROOT")
if not _project_root_from_config_str:
    _project_root_from_config = AGENT_ROOT_DIR.parent.parent
else:
    _project_root_from_config = Path(_project_root_from_config_str)
SELF_IMPROVEMENT_BASE_DIR = _project_root_from_config / "data" / "self_improvement_work_sia" / SELF_AGENT_FILENAME.stem
Config.reset_instance()

IMPROVEMENT_ARCHIVE_DIR = SELF_IMPROVEMENT_BASE_DIR / "archive"
IMPROVEMENT_LOG_FILE = IMPROVEMENT_ARCHIVE_DIR / "improvement_history.jsonl"
FALLBACK_DIR_NAME = "fallback_versions"
ITERATION_DIR_NAME_PREFIX = "iteration_"
DIR_PERMISSIONS_OWNER_ONLY = stat.S_IRWXU


class SelfImprovementAgent:
    """
    Agent responsible for analyzing, implementing, and evaluating code improvements
    for a target Python file, including its own source code.
    Employs safety mechanisms like iteration directories, self-tests, versioned backups,
    and fallbacks. Designed for robust CLI interaction.
    """
    def __init__(
        self,
        agent_id: str = "self_improve_agent_v_final_candidate", 
        llm_provider_override: Optional[str] = None,
        llm_model_name_override: Optional[str] = None,
        max_cycles_override: Optional[int] = None,
        self_test_timeout_override: Optional[float] = None,
        critique_threshold_override: Optional[float] = None,
        is_iteration_instance: bool = False, 
        config_override: Optional[Config] = None
    ):
        self.agent_id = agent_id
        self.config = config_override or Config()
        self.python_executable = sys.executable

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
        self.current_iteration_dir: Optional[Path] = None

        self._ensure_directories_exist()
        logger.info(
            f"SIA '{self.agent_id}' initialized. LLM: {self.llm_handler.provider_name}/{self.llm_handler.model_name or 'default'}. "
            f"MaxCycles: {self.max_self_improve_cycles}. WorkDir: {SELF_IMPROVEMENT_BASE_DIR}. "
            f"Self-Test Timeout: {self.self_test_timeout_seconds}s. Critique Threshold: {self.critique_threshold}. "
            f"Is Iteration Instance: {self.is_iteration_instance}."
        )

    def _ensure_directories_exist(self): # pragma: no cover
        dirs_to_create = [SELF_IMPROVEMENT_BASE_DIR, IMPROVEMENT_ARCHIVE_DIR, 
                          SELF_IMPROVEMENT_BASE_DIR / FALLBACK_DIR_NAME]
        for dir_path in dirs_to_create:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                if os.name != 'nt':
                    try: os.chmod(dir_path, DIR_PERMISSIONS_OWNER_ONLY)
                    except OSError as e_perm: logger.warning(f"SIA: Could not set 0700 perms for {dir_path}: {e_perm}.")
            except Exception as e: logger.error(f"SIA: Failed create dir {dir_path}: {e}")

    def _get_file_content(self, file_path: Path) -> Optional[str]: # pragma: no cover
        try: return file_path.read_text(encoding="utf-8")
        except FileNotFoundError: logger.warning(f"SIA: File not found for read: {file_path}"); return None
        except Exception as e: logger.error(f"SIA: Error reading {file_path}: {e}", exc_info=True); return None

    def _save_file_content(self, file_path: Path, content: str) -> bool: # pragma: no cover
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            logger.info(f"SIA: Saved updated file: {file_path}")
            return True
        except Exception as e: logger.error(f"SIA: Error saving {file_path}: {e}", exc_info=True); return False

    def _record_improvement_attempt(self, attempt_data: Dict[str, Any]): # pragma: no cover
        serializable_data = {k: (str(v) if isinstance(v, Path) else {sk: (str(sv) if isinstance(sv, Path) else sv) for sk, sv in v.items()} if isinstance(v, dict) else v) for k, v in attempt_data.items()}
        serializable_data["timestamp"] = datetime.utcnow().isoformat()
        serializable_data["agent_id"] = self.agent_id
        serializable_data["iteration_dir"] = str(self.current_iteration_dir) if self.current_iteration_dir else None
        try:
            IMPROVEMENT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            with IMPROVEMENT_LOG_FILE.open("a", encoding="utf-8") as f: json.dump(serializable_data, f); f.write("\n")
        except Exception as e: logger.error(f"SIA: Failed record attempt to {IMPROVEMENT_LOG_FILE}: {e}", exc_info=True)

    def _create_iteration_dir(self) -> Optional[Path]: # pragma: no cover
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        iter_dir = SELF_IMPROVEMENT_BASE_DIR / f"{ITERATION_DIR_NAME_PREFIX}{ts}"
        try:
            iter_dir.mkdir(parents=True, exist_ok=True)
            if os.name != 'nt':
                try: os.chmod(iter_dir, DIR_PERMISSIONS_OWNER_ONLY)
                except OSError as e: logger.warning(f"SIA: Chmod iter_dir {iter_dir} fail: {e}.")
            logger.info(f"SIA: Created iteration directory: {iter_dir}")
            return iter_dir
        except Exception as e: logger.error(f"SIA: Failed create iter_dir {iter_dir}: {e}", exc_info=True); return None

    def _backup_current_self(self, backup_name_suffix: str = "", reason: str = "pre_update_backup") -> Optional[Path]: # pragma: no cover
        logger.info(f"SIA: Creating snapshot of {SELF_AGENT_FILENAME}. Reason: {reason}.")
        fallback_dir = SELF_IMPROVEMENT_BASE_DIR / FALLBACK_DIR_NAME
        self._ensure_directories_exist() # Ensures fallback_dir exists
        if not SELF_AGENT_ABSOLUTE_PATH.exists(): logger.error(f"SIA: Main agent file {SELF_AGENT_ABSOLUTE_PATH} not found."); return None
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        reason_slug = re.sub(r'\W+', '_', reason)[:30] # Sanitize reason for filename
        backup_filename = f"{SELF_AGENT_FILENAME}.{ts}{backup_name_suffix}.{reason_slug}.bak"
        backup_path = fallback_dir / backup_filename
        try:
            shutil.copy2(SELF_AGENT_ABSOLUTE_PATH, backup_path)
            logger.info(f"SIA: Backed up main agent to: {backup_path} (Reason: {reason})")
            manifest_path = fallback_dir / "backup_manifest.jsonl"
            with manifest_path.open("a", encoding="utf-8") as f:
                json.dump({"timestamp": time.time(), "path": str(backup_path), "reason": reason, "id": backup_filename, "agent_version_info": self.agent_id}, f) # Add agent version
                f.write("\n")
            return backup_path
        except Exception as e: logger.error(f"SIA: Failed to backup main agent: {e}", exc_info=True); return None

    def _get_latest_fallback_version(self, N: int = 1) -> Optional[Path]: # pragma: no cover
        fallback_dir = SELF_IMPROVEMENT_BASE_DIR / FALLBACK_DIR_NAME
        manifest_path = fallback_dir / "backup_manifest.jsonl"
        if not manifest_path.exists(): logger.warning("SIA: Backup manifest not found."); return None
        backups = []
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try: backups.append(json.loads(line))
                    except json.JSONDecodeError: logger.warning(f"SIA: Corrupt line in manifest: {line.strip()}")
            backups.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            if N > 0 and len(backups) >= N:
                backup_entry = backups[N-1]; backup_path = Path(backup_entry["path"])
                return backup_path if backup_path.exists() else None
        except Exception as e: logger.error(f"SIA: Error reading backup manifest: {e}"); return None
        return None
        
    def _revert_to_nth_fallback(self, N: int = 1, reason: str = "revert_request") -> bool: # pragma: no cover
        logger.warning(f"SIA: Attempting revert main agent to {N}-th latest fallback. Reason: {reason}")
        fallback_p = self._get_latest_fallback_version(N=N)
        if not fallback_p: logger.error(f"SIA: No {N}-th fallback version found."); return False
        
        self._backup_current_self(backup_name_suffix="_before_revert", reason=f"pre_revert_to_{fallback_p.name}")
        try:
            SELF_AGENT_ABSOLUTE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fallback_p, SELF_AGENT_ABSOLUTE_PATH)
            logger.warning(f"SIA: SUCCESSFULLY REVERTED main agent ({SELF_AGENT_FILENAME}) from {fallback_p.name}. RESTART REQUIRED.")
            return True
        except Exception as e: logger.critical(f"SIA CRITICAL: Revert fail from {fallback_p.name}: {e}", exc_info=True); return False

    async def analyze_target(
        self, target_file_path: Path, additional_context: Optional[str] = None,
        logs_for_analysis: Optional[List[str]] = None, improvement_goal_hint: Optional[str] = None
    ) -> Optional[str]: # pragma: no cover
        logger.info(f"SIA: Analyzing target '{target_file_path.name}'. Goal hint: {improvement_goal_hint or 'General'}")
        file_content = self._get_file_content(target_file_path)
        if file_content is None and target_file_path.exists(): return "Error: SIA could not read target for analysis."
        file_content = file_content or ""
        max_chars = self.config.get("self_improvement_agent.analysis.max_code_chars", 70000)
        snippet = file_content[:max_chars//2] + f"\n... (truncated {len(file_content)}b) ...\n" + file_content[-max_chars//2:] if len(file_content) > max_chars else file_content
        ctx_parts = [f"Target ('{target_file_path.name}') code (may be empty/truncated):\n```python\n{snippet}\n```"]
        if improvement_goal_hint: ctx_parts.append(f"\nGoal Hint: {improvement_goal_hint}")
        if additional_context: ctx_parts.append(f"\nContext:\n{additional_context[:7000]}")
        if logs_for_analysis: ctx_parts.append(f"\nLogs:\n{('\n'.join(logs_for_analysis))[:7000]}")
        prompt = ("AI Python expert. Analyze MindX code (Augmentic Project) & context. "
                  "Identify ONE concrete, actionable improvement for quality, performance, robustness, maintainability, or a specific goal.\n\n"
                  f"{''.join(ctx_parts)}\n\n"
                  "Describe this single improvement: WHAT to change & WHY. Concise & clear for code generation. "
                  "Example: 'In `MyClass.method`, add `try-except ValueError` for `int()` conversion, log error, return default.'\n\n"
                  "Proposed Improvement Description:")
        try:
            max_tokens = self.config.get("self_improvement_agent.analysis.max_description_tokens", 350)
            resp = await self.llm_handler.generate_text(prompt, max_tokens=max_tokens, temperature=0.2)
            if not resp or resp.startswith("Error:"): logger.error(f"SIA: LLM analysis error for '{target_file_path.name}': {resp}"); return f"Error: LLM analysis fail - {resp}"
            desc = resp.strip(); logger.info(f"SIA: Proposed for '{target_file_path.name}': {desc}"); return desc
        except Exception as e: logger.error(f"SIA: LLM analysis call exception for '{target_file_path.name}': {e}", exc_info=True); return f"Error: LLM analysis exception - {e}"

    async def implement_improvement(
        self, target_file_path: Path, improvement_description: str, original_content: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]: # (success, new_content_or_error, diff_patch)
        logger.info(f"SIA: Implementing for '{target_file_path.name}': {improvement_description[:120]}...")
        if original_content is None:
            original_content = self._get_file_content(target_file_path)
            if original_content is None and target_file_path.exists(): return False, f"Error: Could not read original of {target_file_path.name}.", None
            original_content = original_content or ""

        prompt = (f"AI Python coder. Modify/create file '{target_file_path.name}' for this goal:\n'{improvement_description}'\n\n"
                  f"Current code (empty if new):\n```python\n{original_content}\n```\n\n"
                  "Provide ONLY the complete, new Python code for the entire file. NO explanations, NO markdown fences (```python or ```)." )
        new_code: Optional[str] = None
        try:
            max_tokens = self.config.get("self_improvement_agent.implementation.max_code_gen_tokens", 12000)
            temp = self.config.get("self_improvement_agent.implementation.temperature", 0.05)
            raw_code = await self.llm_handler.generate_text(prompt, temperature=temp, max_tokens=max_tokens)
            if not raw_code or raw_code.startswith("Error:"): logger.error(f"SIA: LLM code gen error: {raw_code}"); return False, f"LLM code gen fail: {raw_code}", None
            
            new_code = raw_code.strip()
            if new_code.startswith("```python"): new_code = new_code[len("```python"):].strip()
            elif new_code.startswith("```"): new_code = new_code[len("```"):].strip()
            if new_code.endswith("```"): new_code = new_code[:-len("```")].strip()

            if not new_code.strip() and original_content.strip(): logger.warning(f"SIA: LLM made non-empty file empty: {target_file_path.name}. Failing."); return False, "LLM made non-empty file empty.", None
            if not new_code.strip() and not original_content.strip(): logger.info(f"SIA: LLM made new file empty: {target_file_path.name}. Allowed.")

            diff = ""
            if original_content != new_code:
                diff_lines = difflib.unified_diff(original_content.splitlines(keepends=True), new_code.splitlines(keepends=True), fromfile=f"a/{target_file_path.name}", tofile=f"b/{target_file_path.name}", lineterm='')
                diff = "".join(list(diff_lines))
                if not diff and (original_content.strip() != new_code.strip() or original_content != new_code): diff = "Content changed (whitespace/EOL only)."
            else: diff = "No functional code changes generated (content identical)."
            
            if self._save_file_content(target_file_path, new_code): logger.info(f"SIA: Applied LLM changes to {target_file_path} (working dir). Diff: {'Yes' if diff != 'No functional code changes generated (content identical).' else 'No'}"); return True, new_code, diff
            else: return False, f"Failed save modified code to {target_file_path}.", diff
        except Exception as e: logger.error(f"SIA: Code gen/save exception for {target_file_path.name}: {e}", exc_info=True); return False, f"Exception in code gen: {e}", None

    async def _run_self_test_suite(self, iteration_agent_path: Path) -> Tuple[bool, str]: # pragma: no cover
        logger.info(f"SIA: Running self-test suite for candidate: {iteration_agent_path}")
        cmd = [ self.python_executable, str(iteration_agent_path), "--self-test-mode",
                "--llm-provider", self.llm_handler.provider_name, "--llm-model", self.llm_handler.model_name or "",
                "--cycles", str(self.max_self_improve_cycles), "--self-test-timeout", str(self.self_test_timeout_seconds),
                "--critique-threshold", str(self.critique_threshold) ]
        try:
            process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=AGENT_ROOT_DIR)
            timeout = self.self_test_timeout_seconds + 20.0
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            out_s = stdout.decode(errors='ignore').strip(); err_s = stderr.decode(errors='ignore').strip()
            full_out = f"STDOUT:\n{out_s}\n\nSTDERR:\n{err_s}"; logger.debug(f"SIA Self-Test Output ({iteration_agent_path.name}):\n{full_out[:2000]}...")
            if process.returncode == 0 and out_s:
                try:
                    res_json = json.loads(out_s)
                    if res_json.get("status") == "SUCCESS" and ("SELF TEST PASSED" in res_json.get("message","") or any("SELF TEST PASSED" in m for m in res_json.get("data",{}).get("details",[]))):
                        logger.info(f"SIA Self-Test PASSED for {iteration_agent_path.name}."); return True, full_out
                except json.JSONDecodeError: logger.error(f"SIA Self-Test: STDOUT not JSON: {out_s}")
            logger.warning(f"SIA Self-Test FAILED for {iteration_agent_path.name}. RC: {process.returncode}."); return False, full_out
        except asyncio.TimeoutError: logger.error(f"SIA Self-Test TIMED OUT after {timeout}s."); return False, "Self-test suite timed out."
        except Exception as e: logger.error(f"SIA Self-Test EXCEPTION: {e}", exc_info=True); return False, f"Exception: {e}"

    async def evaluate_improvement( self, target_file_path: Path, old_content: Optional[str], new_content: str,
                                   improvement_description: str, is_self_improvement: bool = False ) -> Dict[str, Any]: # pragma: no cover
        logger.info(f"SIA: Evaluating improvement for '{target_file_path.name}' (Self-Improve: {is_self_improvement})")
        eval_res: Dict[str, Any] = {"passed_syntax_check": "NOT_RUN", "passed_self_tests": "N/A" if not is_self_improvement else "NOT_RUN", "llm_self_critique_score": 0.0, "notes": "Eval initiated."}
        try: compile(new_content, str(target_file_path), 'exec'); eval_res["passed_syntax_check"] = True; eval_res["notes"] = "Syntax OK."
        except SyntaxError as se: eval_res["passed_syntax_check"] = False; eval_res["notes"] = f"SyntaxError: L{se.lineno} Off{se.offset}: {se.msg}"; return eval_res
        if is_self_improvement:
            passed_st, st_out = await self._run_self_test_suite(target_file_path)
            eval_res["passed_self_tests"] = passed_st; eval_res["notes"] += f"\nSelf-Test: {'PASS' if passed_st else 'FAIL'}. Output: {st_out[:300]}..."
            if not passed_st: eval_res["llm_self_critique_score"] = 0.0; return eval_res
        
        run_critique = eval_res["passed_syntax_check"] and (eval_res["passed_self_tests"] if is_self_improvement else True)
        if run_critique:
            max_chars_crit = self.config.get("self_improvement_agent.evaluation.max_chars_for_critique", 4000)
            old_snip = old_content[:max_chars_crit//2] if old_content else 'N/A (New file)'; new_snip = new_content[:max_chars_crit]
            crit_prompt = (f"Review AI code change. Goal: '{improvement_description}'\nOld:\n```python\n{old_snip}\n```\nNew:\n```python\n{new_snip}\n```\n"
                           f"Score (0.0-1.0) how well New achieves Goal (correctness, completeness, side effects). JSON: {{\"score\": float, \"justification\": \"string\"}} ONLY.")
            try:
                max_crit_tokens = self.config.get("self_improvement_agent.evaluation.max_critique_tokens", 300)
                crit_resp_str = await self.llm_handler.generate_text(crit_prompt, max_tokens=max_crit_tokens, temperature=0.0, json_mode=True)
                parsed_crit = {}; 
                if crit_resp_str and not crit_resp_str.startswith("Error:"):
                    try: parsed_crit = json.loads(crit_resp_str)
                    except json.JSONDecodeError: match = re.search(r"\{[\s\S]*\}", crit_resp_str);
                    if match: parsed_crit = json.loads(match.group(0))
                    else: logger.warning(f"SIA Eval: LLM critique not JSON: {crit_resp_str[:100]}")
                score = float(parsed_crit.get("score",0.0)); score = max(0.0, min(1.0, score)); just = parsed_crit.get("justification", "No LLM justification.")
                eval_res["llm_self_critique_score"] = score; eval_res["notes"] += f"\nLLM Critique (Score: {score:.2f}): {just}"
            except Exception as e_crit: logger.error(f"SIA Eval: LLM critique fail for {target_file_path.name}: {e_crit}", exc_info=True); eval_res["notes"] += f"\nLLM Critique fail: {e_crit}"
        else: eval_res["notes"] += "\nLLM Critique skipped."
        logger.info(f"SIA Eval for '{target_file_path.name}': Score={eval_res['llm_self_critique_score']:.2f}, Syntax={eval_res['passed_syntax_check']}, SelfTests={eval_res['passed_self_tests']}")
        return eval_res

    async def run_self_improvement_cycle(
        self, target_file_path_conceptual: Path, initial_analysis_context: Optional[str] = None,
        logs_for_analysis: Optional[List[str]] = None, improvement_goal_override: Optional[str] = None
    ) -> Dict[str, Any]: # pragma: no cover
        is_self_attempt = (target_file_path_conceptual.resolve() == SELF_AGENT_ABSOLUTE_PATH)
        logger.info(f"SIA Cycle Start: Target='{target_file_path_conceptual.name}', Self-Improve={is_self_attempt}, GoalOverride='{bool(improvement_goal_override)}'")
        cycle_res: Dict[str, Any] = { "target_file_path_conceptual": str(target_file_path_conceptual), "is_self_improvement_attempt": is_self_attempt, "implementation_status": "PENDING", "promoted_to_main": False if is_self_attempt else "N/A", "code_updated_requires_restart": False }
        self.current_iteration_dir = None; effective_target_path: Path; orig_main_agent_content: Optional[str] = None

        if is_self_attempt:
            if self.is_iteration_instance: cycle_res.update({"error_message": "Recursive self-improve", "implementation_status": "FAILED_RECURSION"}); self._record_improvement_attempt(cycle_res); return cycle_res
            self.current_iteration_dir = self._create_iteration_dir()
            if not self.current_iteration_dir: cycle_res.update({"error_message": "Iter dir fail", "implementation_status": "FAILED_SETUP"}); self._record_improvement_attempt(cycle_res); return cycle_res
            orig_main_agent_content = self._get_file_content(SELF_AGENT_ABSOLUTE_PATH)
            if orig_main_agent_content is None: cycle_res.update({"error_message": f"Read self source fail: {SELF_AGENT_ABSOLUTE_PATH}", "implementation_status": "FAILED_SETUP"}); self._record_improvement_attempt(cycle_res); return cycle_res
            effective_target_path = self.current_iteration_dir / SELF_AGENT_FILENAME
            try: shutil.copy2(SELF_AGENT_ABSOLUTE_PATH, effective_target_path); logger.info(f"SIA: Copied self to iter dir: {effective_target_path}")
            except Exception as e: cycle_res.update({"error_message": f"Copy to iter fail: {e}", "implementation_status": "FAILED_SETUP"}); self._record_improvement_attempt(cycle_res); return cycle_res
        else: effective_target_path = target_file_path_conceptual.resolve()
        cycle_res["effective_target_path"] = str(effective_target_path)

        actual_imp_desc = improvement_goal_override or await self.analyze_target(effective_target_path, initial_analysis_context, logs_for_analysis, improvement_goal_override)
        if not actual_imp_desc or actual_imp_desc.startswith("Error:"): cycle_res.update({"error_message": f"Analysis fail: {actual_imp_desc}", "implementation_status": "FAILED_ANALYSIS", "improvement_description": actual_imp_desc}); self._record_improvement_attempt(cycle_res); return cycle_res
        cycle_res["improvement_description"] = actual_imp_desc
        
        orig_content_for_impl = self._get_file_content(effective_target_path) or "" # Ensure string

        implemented, new_code_or_err, diff = await self.implement_improvement(effective_target_path, actual_imp_desc, orig_content_for_impl)
        cycle_res["diff_patch"] = diff
        if not implemented: cycle_res.update({"error_message": new_code_or_err, "implementation_status": "FAILED_IMPLEMENTATION"}); 
        if self._get_file_content(effective_target_path) != orig_content_for_impl: self._save_file_content(effective_target_path, orig_content_for_impl)
        self._record_improvement_attempt(cycle_res); return cycle_res
        cycle_res["new_content"] = new_code_or_err; cycle_res["implementation_status"] = "SUCCESS_IMPLEMENTED"

        eval_data = await self.evaluate_improvement(effective_target_path, orig_content_for_impl, cycle_res["new_content"], actual_imp_desc, is_self_attempt)
        cycle_res["evaluation"] = eval_data
        eval_passed = eval_data.get("passed_syntax_check",False) and (eval_data.get("passed_self_tests",True) if is_self_attempt else True) and eval_data.get("llm_self_critique_score",0.0) >= self.critique_threshold

        if not eval_passed:
            cycle_res.update({"implementation_status": "FAILED_EVALUATION", "error_message": f"Eval fail. Details: {eval_data.get('notes', '')}"})
            if self._save_file_content(effective_target_path, orig_content_for_impl): logger.info(f"SIA: Reverted working file {effective_target_path.name} due to poor eval.")
            else: cycle_res["error_message"] += " CRITICAL: Fail revert working file."
            self._record_improvement_attempt(cycle_res); return cycle_res
        cycle_res["implementation_status"] = "SUCCESS_EVALUATED"

        if is_self_attempt:
            logger.info(f"SIA: Self-improve candidate {effective_target_path.name} passed. Promoting.")
            iter_name = self.current_iteration_dir.name if self.current_iteration_dir else "unknown"
            backup_p = self._backup_current_self(f"_promo_{iter_name}", f"pre_promo_{iter_name}")
            if not backup_p: cycle_res.update({"error_message": "Backup fail pre-promo.", "implementation_status": "FAILED_PROMOTION_NO_BACKUP"}); self._record_improvement_attempt(cycle_res); return cycle_res
            try:
                shutil.copy2(effective_target_path, SELF_AGENT_ABSOLUTE_PATH); logger.warning(f"SIA: PROMOTED self-improve from {effective_target_path.name} to {SELF_AGENT_FILENAME}. RESTART REQUIRED.")
                cycle_res.update({"promoted_to_main": True, "implementation_status": "SUCCESS_PROMOTED", "code_updated_requires_restart": True})
            except Exception as e_promo:
                logger.critical(f"SIA CRITICAL: Failed to promote {effective_target_path.name}: {e_promo}", exc_info=True)
                cycle_res.update({"promoted_to_main": False, "error_message": f"Promo fail: {e_promo}", "implementation_status": "FAILED_PROMOTION_COPY_ERROR"})
                if self._revert_to_fallback(backup_p): logger.info("SIA: Main agent restored from immediate backup post-promo fail.")
                elif orig_main_agent_content and self._save_file_content(SELF_AGENT_ABSOLUTE_PATH, orig_main_agent_content): logger.info("SIA: Main agent restored pre-cycle content post-promo+backup_revert fail.")
                else: logger.critical("SIA: TRIPLE CRITICAL: PROMO & ALL RESTORES FAILED. MANUAL INTERVENTION.")
        
        self._record_improvement_attempt(cycle_res)
        return cycle_res

    async def improve_self(self, max_cycles_override: Optional[int]=None, initial_analysis_context: Optional[str]=None, logs_for_analysis: Optional[List[str]]=None, improvement_goal_override: Optional[str]=None) -> Dict[str,Any]: # pragma: no cover
        logger.info(f"SIA: Starting self-improvement campaign for '{SELF_AGENT_FILENAME}'...")
        max_cycles = max_cycles_override if max_cycles_override is not None else self.max_self_improve_cycles
        overall_res = {"agent_id": self.agent_id, "operation_type": "improve_self", "target_script_path": str(SELF_AGENT_ABSOLUTE_PATH), "cycles_configured": max_cycles, "cycles_attempted": 0, "cycles_promoted": 0, "final_status": "NO_CYCLES_RUN", "message": "Self-improvement initiated.", "cycle_results": []}
        current_ctx = initial_analysis_context; current_goal = improvement_goal_override
        for i in range(max_cycles):
            logger.info(f"SIA: Self-improve campaign cycle {i+1}/{max_cycles}")
            overall_res["cycles_attempted"] += 1
            cycle_data = await self.run_self_improvement_cycle(SELF_AGENT_ABSOLUTE_PATH, current_ctx, logs_for_analysis if i==0 else None, current_goal)
            overall_res["cycle_results"].append(cycle_data); overall_res["final_status"] = cycle_data.get("implementation_status", "ERR_STAT")
            overall_res["message"] = f"Cycle {i+1}: {overall_res['final_status']}. Desc: {cycle_data.get('improvement_description', 'N/A')[:80]}"
            if cycle_data.get("promoted_to_main"): overall_res["cycles_promoted"] += 1; logger.warning(overall_res["message"] + " Restart required."); current_ctx = f"Promo success: '{cycle_data.get('improvement_description', 'N/A')[:70]}...'. Code changed. Re-eval."; current_goal = None; # break # Option: stop after 1st promo
            elif overall_res["final_status"].startswith("FAILED"): logger.warning(overall_res["message"] + " Halting."); break
            else: current_ctx = f"Last cycle ({cycle_data.get('improvement_description', 'N/A')[:70]}) status {overall_res['final_status']}. Next."; current_goal = None
        if overall_res["cycles_attempted"] == 0: overall_res["message"] = "No self-improve cycles run."
        elif overall_res["cycles_promoted"] > 0: overall_res["message"] = f"Self-improve: {overall_res['cycles_promoted']} update(s) PROMOTED. RESTART CRITICAL."
        else: overall_res["message"] = f"Self-improve: No updates promoted. Last status: {overall_res['final_status']}"
        return overall_res

    async def improve_external_target(self, target_file_path: Path, max_cycles_override: Optional[int]=None, context: Optional[str]=None, logs: Optional[List[str]]=None, improvement_goal_override: Optional[str]=None) -> Dict[str,Any]: # pragma: no cover
        logger.info(f"SIA: Starting improvement campaign for external target: {target_file_path}...")
        if not target_file_path.is_absolute(): return {"final_status": "ERROR_BAD_PATH", "message": "Target path must be absolute."}
        max_cycles = max_cycles_override if max_cycles_override is not None else self.max_self_improve_cycles
        overall_res = {"agent_id": self.agent_id, "operation_type": "improve_external_target", "target_file_path": str(target_file_path), "cycles_configured": max_cycles, "cycles_attempted": 0, "cycles_successful_eval": 0, "final_status": "NO_CYCLES_RUN", "message": "External improvement initiated.", "cycle_results": []}
        current_ctx = context; current_goal = improvement_goal_override
        for i in range(max_cycles):
            logger.info(f"SIA: External improve cycle {i+1}/{max_cycles} for {target_file_path.name}")
            overall_res["cycles_attempted"] += 1
            cycle_data = await self.run_self_improvement_cycle(target_file_path, current_ctx, logs if i==0 else None, current_goal)
            overall_res["cycle_results"].append(cycle_data); overall_res["final_status"] = cycle_data.get("implementation_status", "ERR_STAT")
            overall_res["message"] = f"Cycle {i+1}: {overall_res['final_status']}. Desc: {cycle_data.get('improvement_description', 'N/A')[:80]}"
            if cycle_data.get("implementation_status") == "SUCCESS_EVALUATED": overall_res["cycles_successful_eval"] += 1; current_ctx = f"Success: '{cycle_data.get('improvement_description', 'N/A')[:70]}...'. Target changed. Re-eval."; current_goal = None
            elif overall_res["final_status"].startswith("FAILED"): logger.warning(overall_res["message"] + " Halting."); break
            else: current_ctx = f"Last cycle ({cycle_data.get('improvement_description', 'N/A')[:70]}) status {overall_res['final_status']}. Next."; current_goal = None
        if overall_res["cycles_attempted"] == 0: overall_res["message"] = "No external improve cycles run."
        elif overall_res["cycles_successful_eval"] > 0: overall_res["message"] = f"External improve: {overall_res['cycles_successful_eval']} change(s) successfully evaluated."
        else: overall_res["message"] = f"External improve: No changes evaluated. Last status: {overall_res['final_status']}"
        return overall_res

async def main_cli(): # pragma: no cover
    """CLI entry point for the SelfImprovementAgent. Ensures structured JSON output."""
    import argparse 
    config_instance_for_cli_defaults = Config() # Load config for CLI defaults

    parser = argparse.ArgumentParser(
        description="MindX SelfImprovementAgent CLI. Modifies Python code. Always outputs JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument("target_file", help="Path to Python file or 'self' for self-improvement.")
    parser.add_argument("--context", default=None, help="Textual context for analysis.")
    parser.add_argument("--context-file", type=Path, default=None, help="Path to file containing context.")
    parser.add_argument("--logs", nargs="*", default=[], help="Paths to log files for context.")
    parser.add_argument("--llm-provider", default=config_instance_for_cli_defaults.get("self_improvement_agent.llm.provider"), help="Override LLM provider.")
    parser.add_argument("--llm-model", default=config_instance_for_cli_defaults.get("self_improvement_agent.llm.model"), help="Override LLM model name.")
    parser.add_argument("--cycles", type=int, default=config_instance_for_cli_defaults.get("self_improvement_agent.default_max_cycles"), help="Override number of improvement cycles.")
    parser.add_argument("--self-test-timeout", type=float, default=config_instance_for_cli_defaults.get("self_improvement_agent.self_test_timeout_seconds"), help="Override self-test timeout.")
    parser.add_argument("--critique-threshold", type=float, default=config_instance_for_cli_defaults.get("self_improvement_agent.critique_threshold"), help="Override critique score threshold.")
    parser.add_argument("--rollback", type=int, nargs='?', const=1, default=None, help="Rollback self to Nth latest backup (default: 1, most recent). Skips improvement.")
    parser.add_argument("--self-test-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", action="store_true", help="Output minified JSON (default is indented).")

    final_output_payload: Dict[str, Any] = {"status": "FAILURE", "message": "SIA CLI did not initialize correctly.", "data": {}}
    exit_code = 1; args_parsed: Optional[argparse.Namespace] = None

    try:
        args_parsed = parser.parse_args()

        if args_parsed.self_test_mode:
            logger.info("SIA: Running in self-test mode via CLI command.")
            test_ok = True; test_messages = ["Self-test mode activated via CLI."]
            if not SELF_AGENT_ABSOLUTE_PATH.exists(): test_messages.append(f"ERROR: SELF_AGENT_ABSOLUTE_PATH '{SELF_AGENT_ABSOLUTE_PATH}' does not exist."); test_ok = False
            if SELF_AGENT_FILENAME != "self_improve_agent.py": test_messages.append(f"ERROR: SELF_AGENT_FILENAME is '{SELF_AGENT_FILENAME}', expected 'self_improve_agent.py'."); test_ok = False
            try:
                SELF_IMPROVEMENT_BASE_DIR.mkdir(parents=True, exist_ok=True)
                test_file = SELF_IMPROVEMENT_BASE_DIR / "selftest_permissions_check.tmp"
                test_file.write_text("test_permissions")
                test_file.unlink()
                test_messages.append(f"Basic directory write access OK to '{SELF_IMPROVEMENT_BASE_DIR}'.")
            except Exception as e_dir_test: test_messages.append(f"ERROR: Directory write test failed in '{SELF_IMPROVEMENT_BASE_DIR}': {e_dir_test}"); test_ok = False
            
            final_output_payload = {"status": "SUCCESS" if test_ok else "FAILURE", "message": "SELF TEST " + ("PASSED" if test_ok else "FAILED"), "data": {"details": test_messages}}
            exit_code = 0 if test_ok else 1
            # print is handled in finally
            sys.exit(exit_code) # Exit after self-test completes and output is prepared
        
        # Create agent instance AFTER parsing args, so it picks up CLI overrides for its config
        agent = SelfImprovementAgent(
            llm_provider_override=args_parsed.llm_provider, llm_model_name_override=args_parsed.llm_model,
            max_cycles_override=args_parsed.cycles, self_test_timeout_override=args_parsed.self_test_timeout,
            critique_threshold_override=args_parsed.critique_threshold
        )

        if args_parsed.target_file.lower() == "self" and args_parsed.rollback is not None:
            if agent.is_iteration_instance: # Should not happen if called from main CLI
                final_output_payload = {"status": "FAILURE", "message": "Rollback cannot be run by an iteration instance.", "data":{}}
            else:
                rollback_n = args_parsed.rollback
                logger.warning(f"SIA CLI: Received rollback command for 'self' to version N={rollback_n}.")
                success = agent._revert_to_nth_fallback(N=rollback_n, reason=f"cli_rollback_to_N{rollback_n}")
                if success: final_output_payload = {"status": "SUCCESS", "message": f"Rollback to version N={rollback_n} successful. RESTART AGENT PROCESS REQUIRED.", "data": {"rollback_version_n": rollback_n, "requires_restart": True}}; exit_code = 0
                else: final_output_payload = {"status": "FAILURE", "message": f"Rollback to version N={rollback_n} FAILED.", "data": {"rollback_version_n": rollback_n}}
            # print and exit are handled in finally
            sys.exit(exit_code) # Exit after rollback attempt

        cli_context_str_val: Optional[str] = args_parsed.context
        if args_parsed.context_file:
            if args_parsed.context_file.exists() and args_parsed.context_file.is_file():
                try: cli_context_str_val = args_parsed.context_file.read_text(encoding="utf-8"); logger.info(f"SIA CLI: Loaded context from file: {args_parsed.context_file}")
                except Exception as e_f: final_output_payload = {"status": "FAILURE", "message": f"Error reading context file {args_parsed.context_file}: {e_f}", "data": {"error_type": "InputFileError"}}; raise SystemExit(1)
            else: final_output_payload = {"status": "FAILURE", "message": f"Context file not found or not a file: {args_parsed.context_file}", "data": {"error_type": "InputFileNotFound"}}; raise SystemExit(1)
        
        logs_list_val: List[str] = []
        if args_parsed.logs:
            for lp_str_cli in args_parsed.logs:
                try: logs_list_val.append(Path(lp_str_cli).read_text(encoding="utf-8"))
                except Exception as e_l: logger.warning(f"SIA CLI: Failed to read log file {lp_str_cli}: {e_l}")
        
        sia_op_detailed_result: Dict[str, Any]
        if args_parsed.target_file.lower() == "self":
            sia_op_detailed_result = await agent.improve_self(max_cycles_override=args_parsed.cycles, initial_analysis_context=cli_context_str_val, logs_for_analysis=logs_list_val if logs_list_val else None)
        else:
            target_p_abs_val = Path(args_parsed.target_file).resolve()
            sia_op_detailed_result = await agent.improve_external_target(target_p_abs_val, max_cycles_override=args_parsed.cycles, context=cli_context_str_val, logs=logs_list_val if logs_list_val else None)
        
        final_sia_op_status_val = sia_op_detailed_result.get("final_status", "UNKNOWN_SIA_OP_STATUS_FINAL")
        if final_sia_op_status_val.startswith("SUCCESS"): final_output_payload["status"] = "SUCCESS"; exit_code = 0
        else: final_output_payload["status"] = "FAILURE"; exit_code = 1
        
        final_output_payload["message"] = sia_op_detailed_result.get("message", f"SIA operation completed. Final internal status from SIA: {final_sia_op_status_val}")
        final_output_payload["data"] = sia_op_detailed_result

    except SystemExit as e_sys_exit_main: # To allow sys.exit() to propagate correctly
        # If final_output_payload wasn't set by the specific SystemExit branch (e.g. self-test, rollback),
        # it means argparse itself failed, or some other pre-agent-call SystemExit.
        if final_output_payload["message"] == "SIA CLI did not initialize correctly.":
             final_output_payload["message"] = f"SIA CLI argument parsing or pre-run setup failed. SystemExit code: {e_sys_exit_main.code}"
        # exit_code is already set if it came from self-test or rollback.
        # If it's a new SystemExit (e.g. from argparse failure), set exit_code.
        if exit_code == 1 and getattr(e_sys_exit_main, "code", None) is not None : # Check if SystemExit has a code
            exit_code = e_sys_exit_main.code if isinstance(e_sys_exit_main.code, int) else 1
    except Exception as e_top_main:
        logger.critical("SIA CLI: Top-level unhandled exception during main execution.", exc_info=True)
        final_output_payload["status"] = "FAILURE"
        final_output_payload["message"] = f"Critical unhandled error in SIA CLI: {type(e_top_main).__name__}: {str(e_top_main)}"
        final_output_payload["data"] = {"error_type": type(e_top_main).__name__, "traceback_snippet": traceback.format_exc(limit=3).splitlines()}
        exit_code = 1 # Ensure failure exit code
    finally:
        should_minify_final = getattr(args_parsed, "output_json", False) if args_parsed else False
        print(json.dumps(final_output_payload, indent=2 if not should_minify_final else None))
        sys.exit(exit_code)

if __name__ == "__main__": # pragma: no cover
    Config() # Ensure global Config is loaded, which loads .env for standalone SIA.
    asyncio.run(main_cli())
