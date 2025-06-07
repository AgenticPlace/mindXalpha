# orchestration/coordinator_agent.py
import os
# import logging # Using get_logger
import asyncio
import json
import time
import uuid
import traceback
import subprocess # For calling SIA CLI
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Coroutine
from enum import Enum
import tempfile # For context file for SIA
import ast # For codebase scanning
import sys # For sys.executable

# Assuming these are top-level packages found via project_root in sys.path
from utils.config import Config, PROJECT_ROOT
from utils.logging_config import get_logger
from core.belief_system import BeliefSystem, BeliefSource
from monitoring.resource_monitor import get_resource_monitor_async, ResourceMonitor, ResourceType
from monitoring.performance_monitor import get_performance_monitor_async, PerformanceMonitor
from llm.llm_factory import create_llm_handler
from llm.llm_interface import LLMHandlerInterface # Using full name for clarity

logger = get_logger(__name__)

class InteractionType(Enum): # pragma: no cover
    QUERY = "query"
    SYSTEM_ANALYSIS = "system_analysis"
    COMPONENT_IMPROVEMENT = "component_improvement"
    APPROVE_IMPROVEMENT = "approve_improvement"
    REJECT_IMPROVEMENT = "reject_improvement"
    ROLLBACK_COMPONENT = "rollback_component"

class InteractionStatus(Enum): # pragma: no cover
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_APPROVAL = "pending_approval"

class Interaction: # pragma: no cover
    def __init__( self,
                  interaction_id: str,
                  interaction_type: InteractionType,
                  content: str,
                  user_id: Optional[str] = None,
                  agent_id: Optional[str] = None,
                  metadata: Optional[Dict[str, Any]] = None ):
        self.interaction_id = interaction_id; self.interaction_type = interaction_type
        self.content = content; self.user_id = user_id; self.agent_id = agent_id
        self.metadata = metadata or {}; self.status = InteractionStatus.PENDING
        self.response: Any = None; self.error: Optional[str] = None
        self.created_at: float = time.time(); self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None; self.history: List[Dict[str, Any]] = []

    def add_to_history(self, role: str, message: str, data: Optional[Dict[str, Any]] = None):
        entry: Dict[str, Any] = { "role": role, "message": message, "timestamp": time.time() }
        if data is not None: entry["data"] = data
        self.history.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id, "interaction_type": self.interaction_type.value,
            "content": self.content, "user_id": self.user_id, "agent_id": self.agent_id,
            "metadata": self.metadata, "status": self.status.value, "response": self.response,
            "error": self.error, "created_at": self.created_at, "started_at": self.started_at,
            "completed_at": self.completed_at, "history": self.history
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction':
        interaction = cls( interaction_id=data["interaction_id"], interaction_type=InteractionType(data["interaction_type"]), content=data["content"], user_id=data.get("user_id"), agent_id=data.get("agent_id"), metadata=data.get("metadata", {}) )
        interaction.status = InteractionStatus(data.get("status", InteractionStatus.PENDING.value)); interaction.response = data.get("response"); interaction.error = data.get("error"); interaction.created_at = data.get("created_at", time.time()); interaction.started_at = data.get("started_at"); interaction.completed_at = data.get("completed_at"); interaction.history = data.get("history", []); return interaction
    def __repr__(self): return f"<Interaction id='{self.interaction_id}' type={self.interaction_type.name} status={self.status.name}>"


class CoordinatorAgent:
    _instance = None
    _lock = asyncio.Lock()
    # _llm_handler_instance is not used as self.llm_handler is preferred for instance-specific handler
    # LLM handler is now instance-specific and initialized asynchronously

    def __new__(cls, *args, **kwargs): # pragma: no cover
        # Basic singleton, but factory method get_coordinator_agent_mindx_async is preferred
        if not cls._instance:
            cls._instance = super(CoordinatorAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self, # pragma: no cover
                 belief_system: BeliefSystem,
                 resource_monitor: ResourceMonitor,
                 performance_monitor: PerformanceMonitor,
                 config_override: Optional[Config] = None,
                 test_mode: bool = False):

        if hasattr(self, '_initialized') and self._initialized and not test_mode:
            return

        self.config = config_override or Config()
        self.belief_system = belief_system
        self.resource_monitor = resource_monitor
        self.performance_monitor = performance_monitor

        self.llm_handler: Optional[LLMHandlerInterface] = None # Initialized by _async_init_llm
        self._llm_init_lock = asyncio.Lock()

        self.interactions: Dict[str, Interaction] = {}
        self.active_interactions: Dict[str, Interaction] = {}
        self.completed_interactions: Dict[str, Interaction] = {}

        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.system_capabilities_cache: Optional[Dict[str, Any]] = None

        self.improvement_campaign_history: List[Dict[str, Any]] = self._load_json_file("improvement_campaign_history.json", [])
        self.improvement_backlog: List[Dict[str, Any]] = self._load_backlog()

        self.callbacks: Dict[str, List[Callable[..., Coroutine[Any,Any,None]]]] = {
            "on_interaction_created": [], "on_interaction_started": [],
            "on_interaction_completed": [], "on_interaction_failed": [],
            "on_improvement_campaign_result": [],
            "on_new_improvement_suggestion": []
        }

        self.self_improve_agent_script_path: Optional[Path] = None
        potential_sia_path = PROJECT_ROOT / "learning" / "self_improve_agent.py" # Assumes 'learning' is top-level
        if potential_sia_path.exists() and potential_sia_path.is_file():
            self.self_improve_agent_script_path = potential_sia_path.resolve()
        else: # pragma: no cover
            logger.critical(f"CRITICAL: SelfImprovementAgent script not found at {potential_sia_path}. Component improvement features will be disabled.")

        self._register_default_agents()
        self.sia_concurrency_limit = asyncio.Semaphore(
            self.config.get("coordinator.max_concurrent_sia_tasks", 1)
        )

        if self.resource_monitor and self.resource_monitor.monitoring:
            self._register_monitor_callbacks()

        self.autonomous_improvement_task: Optional[asyncio.Task] = None
        if self.config.get("coordinator.autonomous_improvement.enabled", False) and not test_mode: # pragma: no cover
            self.start_autonomous_improvement_loop()

        self.critical_components_for_approval: List[str] = self.config.get(
            "coordinator.autonomous_improvement.critical_components",
            ["learning.self_improve_agent", "orchestration.coordinator_agent"] # Module paths like 'package.module_stem'
        )
        self.require_human_approval_for_critical: bool = self.config.get(
            "coordinator.autonomous_improvement.require_human_approval_for_critical", True
        )

        logger.info(
            f"CoordinatorAgent mindX initialized. "
            f"SIA Script: {self.self_improve_agent_script_path or 'NOT FOUND - Improvement Disabled'}. "
            f"Autonomous Mode: {self.config.get('coordinator.autonomous_improvement.enabled', False)}. "
            f"HITL for Critical: {self.require_human_approval_for_critical}."
        )
        self._initialized = True

    async def _async_init_llm(self): # pragma: no cover
        async with self._llm_init_lock:
            if self.llm_handler is None:
                coord_llm_provider = self.config.get("coordinator.llm.provider", self.config.get("llm.default_provider"))
                coord_llm_model = self.config.get("coordinator.llm.model", self.config.get(f"llm.{coord_llm_provider}.default_model_for_orchestration"))
                try:
                    self.llm_handler = await create_llm_handler(coord_llm_provider, coord_llm_model)
                    logger.info(f"CoordinatorAgent LLM Handler initialized: {self.llm_handler.provider_name}/{self.llm_handler.model_name_for_api}")
                except Exception as e:
                    logger.error(f"CoordinatorAgent failed to initialize LLM handler: {e}", exc_info=True)
                    from llm.mock_llm_handler import MockLLMHandler
                    self.llm_handler = MockLLMHandler(model_name="mock_coordinator_llm_due_to_error")
                    logger.warning("CoordinatorAgent is using a MockLLMHandler due to initialization error.")
            return self.llm_handler

    def _register_default_agents(self): # pragma: no cover
        self.register_agent(
            agent_id="coordinator_agent_mindx", agent_type="coordinator",
            description="mindX Central Coordinator Agent",
            capabilities=["orchestration", "system_analysis", "component_improvement", "query", "backlog_management", "rollback_trigger"],
            instance=self,
            metadata={"file_path": str(Path(__file__).resolve())}
        )
        if self.self_improve_agent_script_path:
            self.register_agent(
                agent_id="self_improve_agent_cli_mindx", agent_type="self_improvement_worker",
                description="mindX Self-Improvement Worker Agent (CLI based)",
                capabilities=["code_modification", "code_evaluation", "self_update_atomic", "rollback_self"],
                metadata={"script_path": str(self.self_improve_agent_script_path)}
            )
        if self.resource_monitor:
            self.register_agent(agent_id="resource_monitor_mindx", agent_type="monitor", description="mindX System Resource Monitor", capabilities=["system_resource_tracking", "alerting"], instance=self.resource_monitor)
        if self.performance_monitor:
            self.register_agent(agent_id="performance_monitor_mindx", agent_type="monitor", description="mindX LLM Performance Monitor", capabilities=["llm_performance_tracking", "reporting"], instance=self.performance_monitor)

    def _load_json_file(self, file_name: str, default_value: Union[List, Dict]) -> Union[List, Dict]: # pragma: no cover
        file_path = PROJECT_ROOT / "data" / file_name
        if file_path.exists():
            try:
                with file_path.open("r", encoding="utf-8") as f: return json.load(f)
            except Exception as e: logger.error(f"Coordinator: Error loading {file_name} from {file_path}: {e}")
        return default_value

    def _save_json_file(self, file_name: str, data: Union[List, Dict]): # pragma: no cover
        file_path = PROJECT_ROOT / "data" / file_name
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            logger.debug(f"Coordinator: Saved data to {file_path}")
        except Exception as e: logger.error(f"Coordinator: Error saving {file_name} to {file_path}: {e}")

    def _load_backlog(self) -> List[Dict[str, Any]]: # pragma: no cover
        loaded = self._load_json_file("improvement_backlog.json", [])
        valid = [];
        for item in loaded:
            if isinstance(item,dict) and all(k in item for k in ["target_component_path","suggestion"]):
                item.setdefault("id", str(uuid.uuid4())[:8])
                item.setdefault("priority",0); item.setdefault("status",InteractionStatus.PENDING.value)
                item.setdefault("added_at",time.time()); item.setdefault("attempt_count",0)
                item.setdefault("is_critical_target", item.get("is_critical_target", False))
                valid.append(item)
            else: logger.warning(f"Coordinator: Discarding malformed backlog item: {str(item)[:100]}")
        valid.sort(key=lambda x: (x.get("status") == InteractionStatus.PENDING_APPROVAL.value, -int(x.get("priority",0)), x.get("added_at",0)), reverse=False)
        return valid

    def _save_backlog(self): self._save_json_file("improvement_backlog.json", self.improvement_backlog) # pragma: no cover
    def _load_campaign_history(self) -> List[Dict[str, Any]]: return self._load_json_file("improvement_campaign_history.json", []) # pragma: no cover
    def _save_campaign_history(self): self._save_json_file("improvement_campaign_history.json", self.improvement_campaign_history) # pragma: no cover

    def _register_monitor_callbacks(self): # pragma: no cover
        async def handle_resource_alert(monitor_instance: ResourceMonitor, rtype: ResourceType, value: float, path: Optional[str] = None):
            alert_key_base = f"system_health.{rtype.value}.alert_active"; alert_key = f"{alert_key_base}.{Path(path).name.replace('.','_')}" if path else alert_key_base
            logger.warning(f"CoordCB: HIGH RESOURCE: {rtype.name} at {value:.1f}%" + (f" for '{path}'" if path else ""))
            await self.belief_system.add_belief(alert_key,{"percent":value,"path":path,"ts":time.time()},0.85,BeliefSource.PERCEPTION,ttl_seconds=3600*2)
        async def handle_resource_resolve(monitor_instance: ResourceMonitor, rtype: ResourceType, value: float, path: Optional[str] = None):
            alert_key_base = f"system_health.{rtype.value}.alert_active"; alert_key_to_clear = f"{alert_key_base}.{Path(path).name.replace('.','_')}" if path else alert_key_base
            logger.info(f"CoordCB: RESOURCE RESOLVED: {rtype.name} at {value:.1f}%" + (f" for '{path}'" if path else ""))
            await self.belief_system.remove_belief(alert_key_to_clear)
            await self.belief_system.add_belief(alert_key_to_clear.replace("alert_active","resolved_event"),{"percent":value,"path":path,"ts":time.time()},0.9,BeliefSource.PERCEPTION,ttl_seconds=600)
        self.resource_monitor.register_alert_callback(handle_resource_alert)
        self.resource_monitor.register_resolve_callback(handle_resource_resolve)
        logger.info("Coordinator: Registered internal callbacks for resource monitor.")

    def register_agent( self, agent_id: str, agent_type: str, description: str, capabilities: List[str], metadata: Optional[Dict[str, Any]] = None, instance: Any = None ): # pragma: no cover
        self.agent_registry[agent_id] = { "agent_id": agent_id, "agent_type": agent_type, "description": description, "capabilities": capabilities, "metadata": metadata or {}, "status": "available", "registered_at": time.time(), "instance": instance }; logger.info(f"Coordinator: Registered agent {agent_id} (Type: {agent_type})")
        if "coordinator_agent_mindx" in self.agent_registry: self.agent_registry["coordinator_agent_mindx"]["metadata"]["managed_agents"] = list(self.agent_registry.keys())

    async def create_interaction( self, interaction_type: Union[InteractionType, str], content: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None ) -> Interaction: # pragma: no cover
        if isinstance(interaction_type, str):
             try: interaction_type_enum = InteractionType(interaction_type.lower())
             except ValueError as e: logger.error(f"Coord: Invalid interaction_type string '{interaction_type}' for content '{content[:50]}...'."); raise ValueError(f"Invalid interaction_type: {interaction_type}") from e
        elif isinstance(interaction_type, InteractionType): interaction_type_enum = interaction_type
        else: raise TypeError(f"interaction_type must be Enum or str, not {type(interaction_type)}")

        interaction_id = str(uuid.uuid4()); interaction = Interaction( interaction_id, interaction_type_enum, content, user_id, agent_id, metadata )
        initiator = "user" if user_id else ("agent" if agent_id else "system_internal")
        interaction.add_to_history(initiator, f"Interaction created. Type: {interaction_type_enum.name}. Content (start): {content[:100]}...")
        self.interactions[interaction_id] = interaction
        logger.info(f"Coordinator: Created interaction {interaction_id} type {interaction_type_enum.value}"); return interaction

    async def process_interaction(self, interaction: Interaction) -> Interaction: # pragma: no cover
        if self.llm_handler is None:
            await self._async_init_llm()
            if self.llm_handler is None:
                interaction.status = InteractionStatus.FAILED
                interaction.error = "Coordinator LLM handler not available."
                interaction.add_to_history("system_error", interaction.error)
                logger.critical("Coordinator: LLM Handler is None, cannot process interaction.")
                return interaction

        if not isinstance(interaction, Interaction): logger.error("Coord: process_interaction invalid type."); raise TypeError("Invalid object to process_interaction")
        if interaction.status not in [InteractionStatus.PENDING, InteractionStatus.PENDING_APPROVAL]: logger.warning(f"Coord: Interaction {interaction.interaction_id} not PENDING/PENDING_APPROVAL (Status: {interaction.status.name}). Returning."); return interaction
        logger.info(f"Coord: Processing interaction {interaction.interaction_id} (Type: {interaction.interaction_type.name}, Status: {interaction.status.name})")
        interaction.status = InteractionStatus.IN_PROGRESS; interaction.started_at = time.time()
        if interaction.interaction_id not in self.active_interactions: self.active_interactions[interaction.interaction_id] = interaction
        response_data: Any = None
        try:
            if interaction.interaction_type == InteractionType.QUERY: response_data = await self.llm_handler.generate_text(interaction.content, model=self.llm_handler.model_name_for_api, max_tokens=1024, temperature=0.5); interaction.add_to_history("coord_llm", "Query by Coord LLM.")
            elif interaction.interaction_type == InteractionType.SYSTEM_ANALYSIS:
                response_data = await self._process_system_analysis(interaction)
                if isinstance(response_data, dict) and "improvement_suggestions" in response_data:
                    source = interaction.metadata.get("source", "user_request" if interaction.user_id else (f"agent_request:{interaction.agent_id}" if interaction.agent_id else "unknown_source"))
                    added_to_backlog_count = 0
                    for sugg in response_data.get("improvement_suggestions",[]): self.add_to_improvement_backlog(sugg, source=source); added_to_backlog_count+=1
                    interaction.add_to_history("backlog_update", f"{added_to_backlog_count} suggs from analysis to backlog.")
            elif interaction.interaction_type == InteractionType.COMPONENT_IMPROVEMENT: response_data = await self._process_component_improvement_cli(interaction)
            elif interaction.interaction_type == InteractionType.APPROVE_IMPROVEMENT: response_data = self._process_backlog_approval(interaction.metadata.get("backlog_item_id"), approve=True)
            elif interaction.interaction_type == InteractionType.REJECT_IMPROVEMENT: response_data = self._process_backlog_approval(interaction.metadata.get("backlog_item_id"), approve=False)
            elif interaction.interaction_type == InteractionType.ROLLBACK_COMPONENT: response_data = await self._process_component_rollback_cli(interaction)
            else: response_data = {"error": f"Unsupported type: {interaction.interaction_type.name}"}; interaction.status = InteractionStatus.FAILED; interaction.error = response_data["error"]

            if interaction.status != InteractionStatus.FAILED:
                interaction.response = response_data
                interaction.status = InteractionStatus.COMPLETED
            interaction.add_to_history("coordinator", f"Interaction processing finished. Final Status: {interaction.status.name}.")
        except Exception as e:
            logger.error(f"Error processing interaction {interaction.interaction_id}: {e}", exc_info=True)
            interaction.status = InteractionStatus.FAILED; interaction.error = f"{type(e).__name__}: {str(e)}"
            interaction.add_to_history("system_error", f"Unhandled exception: {interaction.error}", {"traceback": traceback.format_exc(limit=3).splitlines()})
        finally:
            interaction.completed_at = time.time();
            if interaction.interaction_id in self.active_interactions: del self.active_interactions[interaction.interaction_id]
            self.completed_interactions[interaction.interaction_id] = interaction
        return interaction

    async def _scan_codebase_capabilities(self) -> Dict[str, Any]: # pragma: no cover
        src_dir = PROJECT_ROOT
        capabilities: Dict[str, Any] = {};
        logger.info(f"Coordinator: Scanning capabilities in subdirectories of: {src_dir}")
        
        packages_to_scan = ["core", "llm", "orchestration", "learning", "monitoring", "utils", "tools"]

        for pkg_name in packages_to_scan:
            pkg_dir = src_dir / pkg_name
            if not pkg_dir.is_dir():
                continue
            
            logger.debug(f"Coordinator: Scanning package directory: {pkg_dir}")
            for item in pkg_dir.rglob("*.py"):
                if item.name.startswith("__"): continue
                try:
                    rel_path_from_pkg = item.relative_to(pkg_dir)
                    module_parts = [pkg_name] + list(rel_path_from_pkg.parts[:-1]) + [rel_path_from_pkg.stem]
                    mod_name = ".".join(module_parts)

                    with item.open("r", encoding="utf-8") as f_handle:
                        tree = ast.parse(f_handle.read())
                    for node in ast.walk(tree):
                        n_name = getattr(node, 'name', None)
                        if not n_name or n_name.startswith("_"): continue

                        cap_k: Optional[str] = None; cap_t: Optional[str] = None
                        doc_s = ast.get_docstring(node)

                        if isinstance(node, ast.ClassDef):
                            cap_k=f"{mod_name}.{n_name}"; cap_t="class"
                        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            is_method = False
                            # Simplified check for methods (might not catch all edge cases of nested functions)
                            for parent_node in ast.walk(tree):
                                if isinstance(parent_node, ast.ClassDef) and node in parent_node.body:
                                    is_method = True; break
                            if not is_method:
                                cap_k=f"{mod_name}.{n_name}"; cap_t="function"
                        
                        if cap_k and cap_t:
                            capabilities[cap_k] = {
                                "type": cap_t, "name": n_name, "module": mod_name,
                                "path": str(item.relative_to(PROJECT_ROOT)),
                                "docstring_snippet": (doc_s[:150] + "..." if doc_s and len(doc_s) > 150 else doc_s)
                            }
                except Exception as e_ast: # pragma: no cover
                    logger.warning(f"Coordinator: AST scan error for file {item}: {e_ast}")
        
        self.system_capabilities_cache = capabilities
        logger.info(f"Coordinator: Scanned {len(capabilities)} capabilities from defined packages.")
        return capabilities


    def add_to_improvement_backlog(self, suggestion: Dict[str, Any], source: str = "system_analysis"): # pragma: no cover
        if not all(k in suggestion for k in ["target_component_path", "suggestion"]) or not suggestion["suggestion"]:
            logger.warning(f"Coord: Skipping invalid suggestion for backlog (missing fields or empty suggestion): {suggestion}")
            return
        
        suggestion.setdefault("priority", 5)
        suggestion.setdefault("id", str(uuid.uuid4())[:8])
        sugg_id = suggestion["id"]

        for item in self.improvement_backlog:
            if item.get("target_component_path") == suggestion["target_component_path"] and \
               item.get("suggestion","").strip()[:70] == suggestion.get("suggestion","").strip()[:70] and \
               item.get("status") == InteractionStatus.PENDING.value:
                logger.info(f"Coord: Similar PENDING suggestion for '{suggestion['target_component_path']}' exists (ID {item.get('id','N/A')[:8]}). Updating priority if higher, and suggestion text.")
                if suggestion['priority'] > item.get('priority',0):
                    item['priority'] = suggestion['priority']
                item['suggestion'] = suggestion['suggestion']
                item['added_at'] = time.time()
                item['source'] = source
                self._save_backlog()
                return

        suggestion["added_at"] = time.time()
        suggestion["status"] = InteractionStatus.PENDING.value
        suggestion["source"] = source
        suggestion["attempt_count"] = 0
        suggestion["last_attempted_at"] = None
        suggestion.setdefault("is_critical_target", False)

        self.improvement_backlog.append(suggestion)
        self.improvement_backlog.sort(key=lambda x: (
            x.get("status") == InteractionStatus.PENDING_APPROVAL.value,
            -int(x.get("priority", 0)),
            x.get("added_at",0)
        ), reverse=False)

        logger.info(f"Coord: Added suggestion (ID:{sugg_id}) for '{suggestion['target_component_path']}' to backlog from '{source}'. Backlog size: {len(self.improvement_backlog)}")
        self._save_backlog()
        for callback in self.callbacks.get("on_new_improvement_suggestion", []): # pragma: no cover
            try: asyncio.create_task(callback(suggestion))
            except Exception as e_cb: logger.error(f"Error in on_new_improvement_suggestion callback: {e_cb}")


    def _process_backlog_approval(self, item_id: Optional[str], approve: bool) -> Dict[str, Any]: # pragma: no cover
        if not item_id: return {"status": "FAILURE", "message": "No backlog_item_id provided."}
        for item in self.improvement_backlog:
            if item.get("id") == item_id and item.get("status") == InteractionStatus.PENDING_APPROVAL.value:
                if approve:
                    item["status"] = InteractionStatus.PENDING.value
                    item["approved_at"] = time.time()
                    item["approved_by"] = "manual_cli"
                    msg=f"Item '{item_id[:8]}' approved and moved to PENDING."
                else:
                    item["status"] = "rejected_manual"
                    item["rejected_at"] = time.time()
                    item["rejected_by"] = "manual_cli"
                    msg=f"Item '{item_id[:8]}' rejected."
                logger.info(msg); self._save_backlog()
                self.improvement_backlog.sort(key=lambda x: (
                    x.get("status") == InteractionStatus.PENDING_APPROVAL.value,
                    -int(x.get("priority", 0)),
                    x.get("added_at",0)
                ), reverse=False)
                return {"status": "SUCCESS", "message": msg}
        return {"status": "FAILURE", "message": f"Item '{item_id[:8]}' not found or not pending approval."}

    async def _process_system_analysis(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        if self.llm_handler is None: await self._async_init_llm()
        if self.llm_handler is None: return {"error": "LLM handler not available for system analysis.", "improvement_suggestions": []}

        interaction.add_to_history("coordinator", "System analysis: Gathering data...")
        if self.system_capabilities_cache is None:
            await self._scan_codebase_capabilities()

        res_usage = self.resource_monitor.get_resource_usage() if self.resource_monitor else {}
        resource_summary = f"CPU: {res_usage.get('cpu_percent',0):.1f}%, Mem: {res_usage.get('memory_percent',0):.1f}%"
        
        perf_metrics = self.performance_monitor.get_summary_metrics() if self.performance_monitor else {}
        performance_summary = f"LLM Calls: {perf_metrics.get('total_calls',0)}, Avg Latency: {perf_metrics.get('avg_latency_ms',0):.0f}ms"
        
        system_structure_summary = f"{len(self.system_capabilities_cache or {})} capabilities scanned. {len(self.agent_registry)} agents registered."
        history_summary = f"{len(self.improvement_campaign_history)} past improvement campaigns logged. {len(self.improvement_backlog)} items in backlog."
        analysis_focus_hint = interaction.metadata.get("analysis_context", "overall system health, identify bottlenecks, and suggest concrete, actionable improvements to code or architecture")

        prompt = (
            f"You are an AI System Architect for the 'mindX' Augmentic Intelligence platform. "
            f"Your task is to perform a system analysis based on the provided telemetry and structure summary. "
            f"Focus on: {analysis_focus_hint}.\n\n"
            f"Current System State:\n"
            f"- Structure: {system_structure_summary}\n"
            f"- Resources: {resource_summary}\n"
            f"- LLM Performance: {performance_summary}\n"
            f"- Improvement History: {history_summary}\n\n"
            f"Instruction: Based on this data, provide 1-2 specific, actionable 'improvement_suggestions'. "
            f"Each suggestion MUST be a dictionary with these keys:\n"
            f"  - 'target_component_path': Python module path (e.g., 'utils.config', 'learning.self_improve_agent').\n"
            f"  - 'suggestion': A clear, concise description of the proposed improvement (max 150 chars).\n"
            f"  - 'priority': An integer from 1 (lowest) to 10 (highest).\n"
            f"  - 'is_critical_target': A boolean (true/false) if this component is considered critical and changes might need review.\n"
            f"  - (Optional) 'details': Further elaboration if needed.\n"
            f"Respond ONLY with a valid JSON object: {{\"improvement_suggestions\": [{{...}}]}}.\n"
            f"If no clear suggestions, return an empty list: {{\"improvement_suggestions\": []}}."
        )
        try:
            max_tok = self.config.get("coordinator.system_analysis.max_tokens", 1024)
            temp = self.config.get("coordinator.system_analysis.temperature", 0.3)
            response_str = await self.llm_handler.generate_text(prompt, model=self.llm_handler.model_name_for_api, max_tokens=max_tok, temperature=temp, json_mode=True)
            
            if not response_str or response_str.startswith("Error:"):
                raise ValueError(f"LLM system analysis error: {response_str}")

            analysis_result: Dict[str, Any] = {"improvement_suggestions": []}
            try:
                parsed_json = json.loads(response_str)
                if isinstance(parsed_json, dict) and "improvement_suggestions" in parsed_json and isinstance(parsed_json["improvement_suggestions"], list):
                    analysis_result = parsed_json
                else:
                    if isinstance(parsed_json, dict):
                        for val in parsed_json.values():
                            if isinstance(val, dict) and "improvement_suggestions" in val and isinstance(val["improvement_suggestions"], list):
                                analysis_result = val; break
                    if not analysis_result["improvement_suggestions"] and isinstance(parsed_json, list):
                         analysis_result["improvement_suggestions"] = parsed_json
            except json.JSONDecodeError:
                logger.warning(f"Coordinator: System analysis LLM response not perfect JSON. Raw: {response_str[:200]}...")
                suggestions_match = re.search(r"\[\s*(\{[\s\S]*?\}(?:\s*,\s*\{[\s\S]*?\})*)\s*\]", response_str, re.DOTALL)
                if suggestions_match:
                    try: analysis_result["improvement_suggestions"] = json.loads(suggestions_match.group(0))
                    except: logger.error("Coordinator: Could not parse extracted suggestions list from LLM response.")

            valid_suggestions = []
            for sugg in analysis_result.get("improvement_suggestions", []):
                if isinstance(sugg, dict) and all(k in sugg for k in ["target_component_path", "suggestion", "priority"]):
                    sugg.setdefault("is_critical_target", False)
                    valid_suggestions.append(sugg)
                else:
                    logger.warning(f"Coordinator: Invalid suggestion format from LLM: {str(sugg)[:100]}")
            analysis_result["improvement_suggestions"] = valid_suggestions

            interaction.add_to_history("llm_analysis", "System analysis generated by LLM.", analysis_result)
            return analysis_result
        except Exception as e: # pragma: no cover
            logger.error(f"Coordinator: Error during LLM system analysis: {e}", exc_info=True)
            return {"error": f"System analysis LLM call failed: {str(e)}", "improvement_suggestions": []}

    async def _resolve_component_path_for_sia(self, component_identifier: str) -> Optional[Path]: # pragma: no cover
        if not component_identifier:
            logger.warning("Coordinator: _resolve_component_path_for_sia called with empty identifier.")
            return None
        
        logger.debug(f"Coordinator: Resolving SIA target component identifier: '{component_identifier}'")

        agent_info = self.get_agent(component_identifier)
        if agent_info and agent_info.get("metadata") and agent_info["metadata"].get("script_path"):
            script_p_str = agent_info["metadata"]["script_path"]
            script_p = Path(script_p_str)
            if script_p.is_absolute() and script_p.exists() and script_p.is_file():
                logger.debug(f"Coordinator: Resolved '{component_identifier}' to registered agent script: {script_p}")
                return script_p.resolve()
            if not script_p.is_absolute():
                script_p_abs = (PROJECT_ROOT / script_p).resolve()
                if script_p_abs.exists() and script_p_abs.is_file():
                    logger.debug(f"Coordinator: Resolved '{component_identifier}' to registered agent script (relative): {script_p_abs}")
                    return script_p_abs

        if "." in component_identifier:
            parts = component_identifier.split('.')
            potential_path = (PROJECT_ROOT / Path(*parts)).with_suffix('.py')
            if potential_path.exists() and potential_path.is_file():
                logger.debug(f"Coordinator: Resolved module path '{component_identifier}' to: {potential_path}")
                return potential_path.resolve()
        
        path_from_project_root = (PROJECT_ROOT / component_identifier).resolve()
        if path_from_project_root.exists() and path_from_project_root.is_file():
            logger.debug(f"Coordinator: Resolved '{component_identifier}' as direct file path relative to PROJECT_ROOT: {path_from_project_root}")
            return path_from_project_root

        logger.warning(f"Coordinator: Could not resolve component identifier '{component_identifier}' to a valid file path. PROJECT_ROOT: {PROJECT_ROOT}")
        return None


    async def _process_component_improvement_cli(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        if not self.self_improve_agent_script_path:
            logger.error("Coordinator: SelfImprovementAgent script path not configured. Cannot process improvement.")
            return {"status": "FAILURE", "message": "SelfImprovementAgent script path not configured."}

        metadata = interaction.metadata
        target_component_id = metadata.get("target_component")
        analysis_context = metadata.get("analysis_context", f"General improvement request for {target_component_id} from Coordinator.")
        priority_for_sia = metadata.get("priority_for_sia")
        backlog_item_id = metadata.get("backlog_item_id")

        if not target_component_id:
            return {"status": "FAILURE", "message": "COMPONENT_IMPROVEMENT requires 'target_component' in metadata."}

        target_path = await self._resolve_component_path_for_sia(target_component_id)
        if not target_path:
            return {"status": "FAILURE", "message": f"Could not resolve target component '{target_component_id}' to a file path."}

        interaction.add_to_history("coordinator", f"Preparing SIA CLI call for target: {target_path}. Context (start): {(analysis_context or '')[:100]}...")

        command = [
            str(sys.executable),
            str(self.self_improve_agent_script_path),
            str(target_path),
            "--output-json"
        ]

        tmp_ctx_file: Optional[Path] = None
        context_file_threshold_bytes = self.config.get("coordinator.sia_context_file_threshold_bytes", 2048)
        
        # CORRECTED CONTEXT HANDLING LOGIC
        if analysis_context and len(analysis_context.encode('utf-8')) > context_file_threshold_bytes:
            try:
                temp_dir_for_context = PROJECT_ROOT / "data" / "temp_sia_contexts"
                temp_dir_for_context.mkdir(parents=True, exist_ok=True)
                fd, temp_file_path_str = tempfile.mkstemp(suffix=".txt", prefix="sia_ctx_", dir=str(temp_dir_for_context), text=True)
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                    tmp_f.write(analysis_context)
                tmp_ctx_file = Path(temp_file_path_str)
                command.extend(["--context-file", str(tmp_ctx_file)])
                interaction.add_to_history("coordinator", f"Large context for SIA written to temp file: {tmp_ctx_file.name}")
                logger.debug(f"Coordinator: Large context for SIA ({len(analysis_context)} chars) written to {tmp_ctx_file}")
            except Exception as e_ctx_file: # pragma: no cover
                logger.error(f"Coordinator: Failed to write large context to temp file: {e_ctx_file}. Passing context directly (may be truncated).", exc_info=True)
                if analysis_context: # Fallback only if context still exists
                    command.extend(["--context", analysis_context])
        elif analysis_context: # Context is not empty and not large enough for a file (or file write failed)
            command.extend(["--context", analysis_context])
        # If analysis_context is None or empty, no context argument is added.

        if priority_for_sia is not None:
            command.extend(["--priority", str(priority_for_sia)])
        if metadata.get("max_iterations_sia"): command.extend(["--max-iterations", str(metadata["max_iterations_sia"])])
        if metadata.get("target_function_sia"): command.extend(["--target-function", metadata["target_function_sia"]])
        if metadata.get("test_command_sia"): command.extend(["--test-command", metadata["test_command_sia"]])
        if metadata.get("auto_apply_sia") is False: command.append("--no-auto-apply")

        logger.info(f"Coordinator: Executing SIA CLI: {' '.join(command)}")
        process_cwd = PROJECT_ROOT
        sia_cli_result_json: Dict[str, Any] = {"status": "FAILURE", "message": "SIA call did not complete."}
        timeout_seconds = self.config.get("coordinator.sia_cli_timeout_seconds", 300.0)

        try:
            async with self.sia_concurrency_limit:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=process_cwd
                )
                interaction.add_to_history("coordinator", f"SIA process started (PID: {process.pid}). Timeout: {timeout_seconds}s.")
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)

            stdout_str = stdout.decode(errors='ignore').strip()
            stderr_str = stderr.decode(errors='ignore').strip()

            if stderr_str:
                interaction.add_to_history("sia_stderr", stderr_str[:1000])
                logger.warning(f"Coordinator: SIA stderr for target '{target_component_id}':\n{stderr_str}")

            if stdout_str:
                try:
                    sia_cli_result_json = json.loads(stdout_str)
                    interaction.add_to_history("sia_stdout_json", "SIA produced JSON output.", sia_cli_result_json)
                except json.JSONDecodeError:
                    logger.error(f"Coordinator: SIA STDOUT for '{target_component_id}' was not valid JSON:\n{stdout_str}")
                    interaction.add_to_history("sia_stdout_raw_error", "SIA STDOUT not JSON.", {"raw_stdout": stdout_str[:1000]})
                    sia_cli_result_json = {"status": "FAILURE", "message": "SIA STDOUT not JSON.", "raw_stdout": stdout_str}
            elif process.returncode != 0:
                logger.error(f"Coordinator: SIA process for '{target_component_id}' failed with code {process.returncode}. stderr: {stderr_str}")
                sia_cli_result_json = {"status": "FAILURE", "message": f"SIA process failed (Code: {process.returncode})", "stderr": stderr_str}
            else:
                logger.warning(f"Coordinator: SIA process for '{target_component_id}' exited 0 but STDOUT was empty. stderr: {stderr_str}")
                sia_cli_result_json = {"status": "SUCCESS", "message": "SIA process exited 0, STDOUT empty.", "data": {"final_status": "SUCCESS_NO_OUTPUT"}}

            campaign_entry = {
                "campaign_id": interaction.interaction_id, "timestamp": time.time(),
                "target_component": target_component_id, "target_path": str(target_path),
                "context_used": bool(analysis_context),
                "sia_result_status": sia_cli_result_json.get("status", "UNKNOWN"),
                "sia_final_op_status": sia_cli_result_json.get("data", {}).get("final_status", "UNKNOWN"),
                "message": sia_cli_result_json.get("message", ""),
                "modified_files": sia_cli_result_json.get("data", {}).get("modified_files", []),
                "applied_successfully": sia_cli_result_json.get("data", {}).get("applied_successfully", False),
                "backlog_item_id": backlog_item_id
            }
            self.improvement_campaign_history.append(campaign_entry)
            self._save_campaign_history()

            if sia_cli_result_json.get("status") == "SUCCESS" and \
               sia_cli_result_json.get("data", {}).get("final_status", "").startswith("SUCCESS_NEEDS_APPROVAL"):
                is_critical = any(crit_stem in str(target_path) for crit_stem in self.critical_components_for_approval)
                if is_critical and self.require_human_approval_for_critical:
                    logger.warning(f"Coordinator: SIA modified CRITICAL component '{target_component_id}' and it requires approval. Original interaction ID: {interaction.interaction_id}")
                    if backlog_item_id:
                        for item in self.improvement_backlog:
                            if item.get("id") == backlog_item_id:
                                item["status"] = InteractionStatus.PENDING_APPROVAL.value
                                item["approval_details"] = {
                                    "message": "SIA generated changes requiring approval.",
                                    "diff_path": sia_cli_result_json.get("data", {}).get("diff_file_path"),
                                    "new_version_path": sia_cli_result_json.get("data", {}).get("new_version_path")
                                }
                                self._save_backlog()
                                break
                elif is_critical and not self.require_human_approval_for_critical:
                    logger.info(f"Coordinator: SIA modified CRITICAL component '{target_component_id}', but auto-approval is enabled for criticals.")

        except asyncio.TimeoutError: # pragma: no cover
            logger.error(f"Coordinator: SIA CLI call for '{target_component_id}' timed out after {timeout_seconds}s.")
            interaction.add_to_history("sia_timeout", f"SIA call timed out after {timeout_seconds}s.")
            sia_cli_result_json = {"status": "FAILURE", "message": f"SIA call timed out after {timeout_seconds}s."}
            if process and process.returncode is None:
                try: process.terminate(); await asyncio.wait_for(process.wait(), timeout=5.0)
                except: process.kill()
        except Exception as e_proc: # pragma: no cover
            logger.error(f"Coordinator: Exception during SIA CLI call for '{target_component_id}': {e_proc}", exc_info=True)
            interaction.add_to_history("sia_exception", f"Exception: {type(e_proc).__name__}: {e_proc}")
            sia_cli_result_json = {"status": "FAILURE", "message": f"SIA call exception: {type(e_proc).__name__}: {e_proc}"}
        finally:
            if tmp_ctx_file and tmp_ctx_file.exists():
                try: tmp_ctx_file.unlink()
                except OSError as e_unlink: logger.warning(f"Coordinator: Could not delete temp context file {tmp_ctx_file}: {e_unlink}")

        return sia_cli_result_json

    async def _process_component_rollback_cli(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        if not self.self_improve_agent_script_path:
            return {"status": "FAILURE", "message": "SelfImprovementAgent script path not configured."}

        metadata = interaction.metadata
        target_component_id = metadata.get("target_component_for_rollback")
        rollback_version_n = metadata.get("rollback_version_n", 1)

        if not target_component_id:
            return {"status": "FAILURE", "message": "ROLLBACK_COMPONENT requires 'target_component_for_rollback'."}

        effective_rollback_target_for_cli = "self" # SIA CLI usually takes "self" for its own rollback

        interaction.add_to_history("coordinator", f"Preparing SIA CLI to rollback '{effective_rollback_target_for_cli}' to Nth={rollback_version_n} backup.")
        command = [
            str(sys.executable),
            str(self.self_improve_agent_script_path),
            effective_rollback_target_for_cli,
            "--rollback", str(rollback_version_n),
            "--output-json"
        ]

        logger.info(f"Coordinator: Executing SIA Rollback CLI: {' '.join(command)}")
        process_cwd = PROJECT_ROOT
        sia_cli_result_json: Dict[str, Any] = {"status": "FAILURE", "message": "SIA rollback call did not complete."}
        timeout_seconds = self.config.get("coordinator.sia_rollback_timeout_seconds", 60.0)

        try:
            async with self.sia_concurrency_limit:
                process = await asyncio.create_subprocess_exec(
                    *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=process_cwd)
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)

            stdout_str = stdout.decode(errors='ignore').strip()
            stderr_str = stderr.decode(errors='ignore').strip()
            if stderr_str: interaction.add_to_history("sia_stderr_rollback", stderr_str[:1000])

            if stdout_str:
                try: sia_cli_result_json = json.loads(stdout_str)
                except json.JSONDecodeError: sia_cli_result_json = {"status": "FAILURE", "message": "SIA rollback STDOUT not JSON", "raw_stdout": stdout_str}
            elif process.returncode != 0: sia_cli_result_json = {"status": "FAILURE", "message": f"SIA rollback process failed (Code: {process.returncode})", "stderr": stderr_str}
            else: sia_cli_result_json = {"status": "SUCCESS", "message": "SIA rollback process exited 0, STDOUT empty.", "data": {"final_status": "SUCCESS_NO_OUTPUT"}}

            if sia_cli_result_json.get("status") == "SUCCESS":
                interaction.add_to_history("sia_cli_rollback_success", sia_cli_result_json.get("message","SIA rollback successful."), sia_cli_result_json.get("data",{}))
                logger.warning(f"Coordinator: SIA reported successful rollback of '{target_component_id}'. System restart may be required if it was SIA itself.")
                await self.belief_system.add_belief(f"system.restart_required.reason", f"SIA rollback executed for {target_component_id}", 0.98, BeliefSource.SELF_ANALYSIS)
            else:
                logger.error(f"Coordinator: SIA rollback operation reported failure. SIA JSON: {json.dumps(sia_cli_result_json)}")
                interaction.add_to_history("sia_cli_rollback_failure", sia_cli_result_json.get("message", "SIA rollback failed."), sia_cli_result_json)
        except asyncio.TimeoutError: sia_cli_result_json = {"status": "FAILURE", "message": f"SIA rollback call timed out."}
        except Exception as e_proc: sia_cli_result_json = {"status": "FAILURE", "message": f"SIA rollback call exception: {str(e_proc)}"}

        return sia_cli_result_json

    async def _autonomous_improvement_worker(self, interval_seconds: float): # pragma: no cover
        logger.info(f"Autonomous worker starting. Interval: {interval_seconds}s. HITL Critical: {self.require_human_approval_for_critical}")
        cool_down_store: Dict[str, float] = {}; cool_down_secs = self.config.get("coordinator.autonomous_improvement.cooldown_seconds_after_failure", 1800.0)
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                logger.info("Autonomous worker: Cycle start.")
                
                # Ensure LLM is ready for analysis
                if self.llm_handler is None: await self._async_init_llm()
                if self.llm_handler is None: logger.error("Autonomous worker: LLM not available, cannot perform analysis."); continue

                analysis_interaction = await self.create_interaction(InteractionType.SYSTEM_ANALYSIS, "Periodic autonomous system analysis for improvements.", agent_id="autonomous_worker_mindx")
                await self.process_interaction(analysis_interaction) # This populates the backlog via _process_system_analysis

                if self.resource_monitor and self.resource_monitor.get_resource_usage().get("cpu_percent",0) > self.config.get("coordinator.autonomous_improvement.max_cpu_before_sia",90.0):
                    logger.warning("Autonomous: CPU usage high, deferring SIA task processing this cycle."); continue

                if not self.improvement_backlog: logger.info("Autonomous: Improvement backlog is empty."); continue

                next_item_to_process: Optional[Dict[str,Any]] = None
                for item in self.improvement_backlog:
                    if item.get("status") == InteractionStatus.PENDING.value:
                        last_failed_attempt_ts = cool_down_store.get(item["target_component_path"])
                        if last_failed_attempt_ts and (time.time() - last_failed_attempt_ts < cool_down_secs):
                            logger.info(f"Autonomous: Skipping '{item['target_component_path']}' (ID {item.get('id','N/A')[:8]}), in cool-down period after previous failure.")
                            continue
                        next_item_to_process = item; break
                
                if not next_item_to_process: logger.info("Autonomous: No actionable (PENDING and not in cool-down) items found in backlog."); continue

                target_path = next_item_to_process["target_component_path"]
                is_critical = next_item_to_process.get("is_critical_target",False) or any(crit_stem in target_path for crit_stem in self.critical_components_for_approval)

                if self.require_human_approval_for_critical and is_critical and next_item_to_process.get("approved_at") is None:
                    if next_item_to_process["status"] != InteractionStatus.PENDING_APPROVAL.value:
                        next_item_to_process["status"] = InteractionStatus.PENDING_APPROVAL.value; self._save_backlog()
                        logger.warning(f"Autonomous: CRITICAL improvement for '{target_path}' (ID {next_item_to_process.get('id')[:8]}) requires human approval. Suggestion: {next_item_to_process['suggestion']}")
                    else:
                        logger.info(f"Autonomous: Critical item '{target_path}' (ID {next_item_to_process.get('id')[:8]}) still pending approval.")
                    continue # Skip to next cycle, waiting for approval

                logger.info(f"Autonomous: Attempting improvement for: '{target_path}' (ID {next_item_to_process.get('id')[:8]}, Prio: {next_item_to_process.get('priority')})")
                next_item_to_process["status"] = InteractionStatus.IN_PROGRESS.value
                next_item_to_process["attempt_count"] = next_item_to_process.get("attempt_count",0)+1
                next_item_to_process["last_attempted_at"] = time.time()
                self._save_backlog()

                imp_meta = {
                    "target_component":target_path,
                    "analysis_context":next_item_to_process["suggestion"],
                    "source":"autonomous_worker_backlog",
                    "backlog_item_id":next_item_to_process.get("id"),
                    "priority_for_sia": next_item_to_process.get("priority") # Pass priority to SIA
                }
                imp_content = f"Autonomous attempt (priority {next_item_to_process.get('priority')}): {next_item_to_process['suggestion'][:100]}"
                imp_interaction = await self.create_interaction(InteractionType.COMPONENT_IMPROVEMENT, imp_content, agent_id="autonomous_worker_mindx", metadata=imp_meta)
                processed_imp_int = await self.process_interaction(imp_interaction)

                sia_resp_json = processed_imp_int.response
                if isinstance(sia_resp_json, dict):
                    if sia_resp_json.get("status") == "SUCCESS":
                        final_sia_op_status = sia_resp_json.get("data",{}).get("final_status", "UNKNOWN_SUCCESS")
                        next_item_to_process["status"] = InteractionStatus.COMPLETED.value if final_sia_op_status.startswith("SUCCESS") else InteractionStatus.FAILED.value
                        next_item_to_process["sia_final_op_status"] = final_sia_op_status
                        if not final_sia_op_status.startswith("SUCCESS"): cool_down_store[target_path] = time.time()
                    else: # SIA itself reported failure
                        next_item_to_process["status"] = InteractionStatus.FAILED.value
                        next_item_to_process["sia_error"] = sia_resp_json.get("message", "SIA reported failure status.")
                        cool_down_store[target_path] = time.time()
                else: # Unexpected response from SIA call
                    next_item_to_process["status"] = InteractionStatus.FAILED.value
                    next_item_to_process["sia_error"] = "Invalid or missing JSON response from SIA process."
                    cool_down_store[target_path] = time.time()
                
                next_item_to_process["last_completed_at"] = time.time() # Mark completion of this attempt
                self._save_backlog()
                logger.info(f"Autonomous: Improvement attempt for '{target_path}' (ID {next_item_to_process.get('id')[:8]}) finished. Backlog item status: {next_item_to_process['status']}")

            except asyncio.CancelledError: # pragma: no cover
                logger.info("Autonomous improvement worker stopping due to cancellation.")
                break
            except Exception as e: # pragma: no cover
                logger.error(f"Autonomous improvement worker encountered an unhandled error: {e}", exc_info=True)
                await asyncio.sleep(max(interval_seconds // 2, 1800)) # Longer sleep on major error
        logger.info("Autonomous improvement worker has stopped.")

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]: # pragma: no cover
        return self.agent_registry.get(agent_id)

    async def handle_user_input( self, content: str, user_id: Optional[str] = None, interaction_type: Optional[Union[InteractionType, str]] = None, metadata: Optional[Dict[str, Any]] = None ) -> Dict[str, Any]: # pragma: no cover
        logger.info(f"Coordinator: User input from '{user_id}': '{content[:100]}...' Meta: {metadata}")
        metadata = metadata or {}
        parsed_content = content
        interaction_type_enum: Optional[InteractionType] = None

        if isinstance(interaction_type, InteractionType):
            interaction_type_enum = interaction_type
        elif isinstance(interaction_type, str):
            try: interaction_type_enum = InteractionType(interaction_type.lower())
            except ValueError:
                logger.error(f"Invalid interaction_type string provided: {interaction_type}")
                pass

        if not interaction_type_enum:
            parts = content.strip().split(" ", 1); cmd_verb = parts[0].lower(); args_str = parts[1] if len(parts) > 1 else ""
            if cmd_verb == "analyze_system": interaction_type_enum = InteractionType.SYSTEM_ANALYSIS; parsed_content = "System analysis"; metadata["analysis_context"] = args_str
            elif cmd_verb == "improve":
                interaction_type_enum = InteractionType.COMPONENT_IMPROVEMENT
                improve_args = args_str.split(" ", 1)
                if not improve_args or not improve_args[0]: return {"error": "Improve command requires target_component.", "status": InteractionStatus.FAILED.value}
                metadata["target_component"] = improve_args[0]
                if len(improve_args) > 1: metadata["analysis_context"] = improve_args[1]
                else: metadata["analysis_context"] = f"Improve component {metadata['target_component']}"
                parsed_content = f"Request to improve {metadata['target_component']}"
            elif cmd_verb == "approve":
                interaction_type_enum = InteractionType.APPROVE_IMPROVEMENT
                metadata["backlog_item_id"] = args_str.strip()
                parsed_content = f"Approve backlog item {args_str.strip()}"
            elif cmd_verb == "reject":
                interaction_type_enum = InteractionType.REJECT_IMPROVEMENT
                metadata["backlog_item_id"] = args_str.strip()
                parsed_content = f"Reject backlog item {args_str.strip()}"
            elif cmd_verb == "rollback":
                interaction_type_enum = InteractionType.ROLLBACK_COMPONENT
                rollback_args = args_str.split(" ",1)
                if not rollback_args or not rollback_args[0]: return {"error": "Rollback command requires target_component.", "status": InteractionStatus.FAILED.value}
                metadata["target_component_for_rollback"] = rollback_args[0]
                metadata["rollback_version_n"] = int(rollback_args[1]) if len(rollback_args)>1 and rollback_args[1].isdigit() else 1
                parsed_content = f"Rollback {metadata['target_component_for_rollback']} to Nth={metadata['rollback_version_n']}"
            else:
                interaction_type_enum = InteractionType.QUERY
                parsed_content = content
        
        try:
            interaction = await self.create_interaction( interaction_type_enum, parsed_content, user_id, metadata=metadata )
            processed_interaction = await self.process_interaction(interaction)
            return processed_interaction.to_dict()
        except Exception as e:
            logger.error(f"Error in handle_user_input (pre-processing): {e}", exc_info=True)
            return {"error": str(e), "status": InteractionStatus.FAILED.value}


    def start_autonomous_improvement_loop(self, interval_seconds: Optional[float] = None): # pragma: no cover
        if self.autonomous_improvement_task and not self.autonomous_improvement_task.done():
            logger.warning("Coordinator: Autonomous improvement loop already running.")
            return
        loop_interval = interval_seconds or self.config.get("coordinator.autonomous_improvement.interval_seconds", 3600.0)
        if loop_interval <= 0 :
             logger.error(f"Coordinator: Invalid autonomous interval: {loop_interval}. Not starting loop.")
             return
        self.autonomous_improvement_task = asyncio.create_task(self._autonomous_improvement_worker(loop_interval))
        logger.info(f"Coordinator: Autonomous improvement loop started. Interval: {loop_interval}s.")

    def stop_autonomous_improvement_loop(self): # pragma: no cover
        if self.autonomous_improvement_task and not self.autonomous_improvement_task.done():
            self.autonomous_improvement_task.cancel()
            logger.info("Coordinator: Autonomous improvement loop stopping request sent...")
        else:
            logger.info("Coordinator: Autonomous improvement loop not running or already stopped.")


    async def shutdown(self): # pragma: no cover
        logger.info(f"CoordinatorAgent mindX ({self.config.get('version', 'unknown')}) shutting down...")
        self.stop_autonomous_improvement_loop()
        if self.autonomous_improvement_task:
            try:
                await asyncio.wait_for(self.autonomous_improvement_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.warning("Coordinator: Autonomous improvement task did not shut down cleanly within timeout.")
            except Exception as e_shutdown_task: # pragma: no cover
                 logger.error(f"Coordinator: Exception during autonomous task shutdown: {e_shutdown_task}")


        if self.resource_monitor and hasattr(self.resource_monitor, 'stop_monitoring') and callable(self.resource_monitor.stop_monitoring):
            self.resource_monitor.stop_monitoring() # This should be synchronous
        if self.performance_monitor and hasattr(self.performance_monitor, 'shutdown') and asyncio.iscoroutinefunction(self.performance_monitor.shutdown):
            await self.performance_monitor.shutdown()
        
        self._save_backlog()
        self._save_campaign_history()
        logger.info(f"CoordinatorAgent mindX ({self.config.get('version', 'unknown')}) shutdown complete.")

    @classmethod
    async def reset_instance_async(cls): # pragma: no cover
        async with cls._lock:
            if cls._instance:
                logger.debug(f"CoordinatorAgent: Resetting existing instance.")
                await cls._instance.shutdown()
                cls._instance._initialized = False
                cls._instance = None
        logger.debug("CoordinatorAgent instance reset asynchronously.")

async def get_coordinator_agent_mindx_async(config_override: Optional[Config] = None, test_mode: bool = False) -> CoordinatorAgent: # pragma: no cover
    async with CoordinatorAgent._lock:
        should_create_new = CoordinatorAgent._instance is None or test_mode
        
        if should_create_new:
            if test_mode and CoordinatorAgent._instance is not None:
                logger.debug("CoordinatorAgent Factory: Test mode, shutting down existing instance before creating new.")
                await CoordinatorAgent._instance.shutdown()
                CoordinatorAgent._instance = None
            
            logger.info("CoordinatorAgent Factory: Creating new instance.")
            effective_config = config_override or Config(test_mode=test_mode)
            
            belief_system = BeliefSystem(
                persistence_file_path=PROJECT_ROOT / "data" / "coordinator_beliefs.json",
                test_mode=test_mode
            )
            
            resource_monitor_instance = await get_resource_monitor_async(config_override=effective_config, test_mode=test_mode)
            performance_monitor_instance = await get_performance_monitor_async(config_override=effective_config, test_mode=test_mode)
            
            new_instance = CoordinatorAgent(
                belief_system=belief_system,
                resource_monitor=resource_monitor_instance,
                performance_monitor=performance_monitor_instance,
                config_override=effective_config,
                test_mode=test_mode
            )
            await new_instance._async_init_llm()
            CoordinatorAgent._instance = new_instance
            logger.info("CoordinatorAgent Factory: New instance created and LLM initialized.")
        else: # pragma: no cover
            logger.debug("CoordinatorAgent Factory: Returning existing instance.")
            if CoordinatorAgent._instance.llm_handler is None:
                 await CoordinatorAgent._instance._async_init_llm()
        return CoordinatorAgent._instance

def get_coordinator_agent_mindx(config_override: Optional[Config] = None, test_mode:bool = False) -> CoordinatorAgent: # pragma: no cover
    # This sync getter is complex to make truly safe with async components.
    # Prefer async_get_coordinator_agent_mindx_async where possible.
    if CoordinatorAgent._instance and not test_mode:
        if CoordinatorAgent._instance.llm_handler is None:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running(): # pragma: no cover
                    loop.run_until_complete(CoordinatorAgent._instance._async_init_llm())
                else: # pragma: no cover
                    logger.warning("CoordinatorAgent sync getter: LLM not initialized and loop is running. LLM will init on first async op.")
            except RuntimeError: # No loop
                 asyncio.run(CoordinatorAgent._instance._async_init_llm())
        return CoordinatorAgent._instance

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        instance = loop.run_until_complete(get_coordinator_agent_mindx_async(config_override, test_mode))
        return instance
    else:
        if not loop.is_running(): # pragma: no cover
             return loop.run_until_complete(get_coordinator_agent_mindx_async(config_override, test_mode))
        else: # pragma: no cover
             logger.error("Sync get_coordinator_agent_mindx called improperly with a running asyncio loop without nest_asyncio.")
             raise RuntimeError("Sync get_coordinator_agent_mindx called improperly with a running asyncio loop.")
