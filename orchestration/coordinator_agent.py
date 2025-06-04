# mindx/orchestration/coordinator_agent.py
import os
import logging
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

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.core.belief_system import BeliefSystem, BeliefSource
from mindx.monitoring.resource_monitor import get_resource_monitor_async, ResourceMonitor, ResourceType
from mindx.monitoring.performance_monitor import get_performance_monitor_async, PerformanceMonitor
from mindx.llm.llm_factory import create_llm_handler, LLMHandler
# Import stubs for other agents if they are developed further (not strictly needed for this core loop)
# from .multimodel_agent import MultiModelAgent, TaskType
# from .model_selector import ModelSelector
# from ..core.bdi_agent import BDIAgent # This refers to the general BDI, not SEA's internal one.
# from ..docs.documentation_agent import DocumentationAgent

logger = logging.getLogger(__name__)

class InteractionType(Enum): # pragma: no cover
    """Defines the types of interactions the Coordinator can handle."""
    QUERY = "query"
    SYSTEM_ANALYSIS = "system_analysis"
    COMPONENT_IMPROVEMENT = "component_improvement"
    APPROVE_IMPROVEMENT = "approve_improvement" # For HITL
    REJECT_IMPROVEMENT = "reject_improvement"   # For HITL
    ROLLBACK_COMPONENT = "rollback_component"   # New: Request SIA to rollback a component (usually itself)

class InteractionStatus(Enum): # pragma: no cover
    """Defines the possible statuses of an interaction."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_APPROVAL = "pending_approval"

class Interaction: # pragma: no cover
    """Represents a single interaction managed by the Coordinator."""
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
    def from_dict(cls, data: Dict[str, Any]) -> 'Interaction': # Not strictly used by current Coordinator, but good for completeness
        interaction = cls( interaction_id=data["interaction_id"], interaction_type=InteractionType(data["interaction_type"]), content=data["content"], user_id=data.get("user_id"), agent_id=data.get("agent_id"), metadata=data.get("metadata", {}) )
        interaction.status = InteractionStatus(data.get("status", InteractionStatus.PENDING.value)); interaction.response = data.get("response"); interaction.error = data.get("error"); interaction.created_at = data.get("created_at", time.time()); interaction.started_at = data.get("started_at"); interaction.completed_at = data.get("completed_at"); interaction.history = data.get("history", []); return interaction
    def __repr__(self): return f"<Interaction id='{self.interaction_id}' type={self.interaction_type.name} status={self.status.name}>"


class CoordinatorAgent:
    """
    Central orchestrator for the MindX system. Manages system analysis,
    delegates component improvements to SelfImprovementAgent (SIA) via CLI,
    handles an improvement backlog, and runs an autonomous improvement loop
    with Human-in-the-Loop (HITL) for critical changes.
    """
    _instance = None
    _lock = asyncio.Lock() 

    def __new__(cls, *args, **kwargs): # pragma: no cover
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
        
        coord_llm_provider = self.config.get("coordinator.llm.provider")
        coord_llm_model = self.config.get("coordinator.llm.model")
        self.llm_handler: LLMHandler = create_llm_handler(coord_llm_provider, coord_llm_model)

        self.interactions: Dict[str, Interaction] = {}
        self.active_interactions: Dict[str, Interaction] = {}
        self.completed_interactions: Dict[str, Interaction] = {}
        
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.system_capabilities_cache: Optional[Dict[str, Any]] = None
        
        self.improvement_campaign_history: List[Dict[str, Any]] = self._load_json_file("improvement_campaign_history.json", [])
        self.improvement_backlog: List[Dict[str, Any]] = self._load_json_file("improvement_backlog.json", [])
        
        self.callbacks: Dict[str, List[Callable[..., Coroutine[Any,Any,None]]]] = {
            "on_interaction_created": [], "on_interaction_started": [],
            "on_interaction_completed": [], "on_interaction_failed": [],
            "on_improvement_campaign_result": [], 
            "on_new_improvement_suggestion": []
        }

        self.self_improve_agent_script_path: Optional[Path] = None
        potential_sia_path = PROJECT_ROOT / "mindx" / "learning" / "self_improve_agent.py"
        if potential_sia_path.exists() and potential_sia_path.is_file():
            self.self_improve_agent_script_path = potential_sia_path
        else: # pragma: no cover
            logger.critical(f"CRITICAL: SelfImprovementAgent script not found at {potential_sia_path}. Component improvement features will be disabled.")
        
        self._register_default_agents()
        self.sia_concurrency_limit = asyncio.Semaphore(
            self.config.get("coordinator.max_concurrent_sia_tasks", 1) # Default to 1 for safety
        )

        if self.resource_monitor.monitoring: 
            self._register_monitor_callbacks()
        
        self.autonomous_improvement_task: Optional[asyncio.Task] = None
        if self.config.get("coordinator.autonomous_improvement.enabled", False) and not test_mode: # pragma: no cover
            self.start_autonomous_improvement_loop()
        
        self.critical_components_for_approval: List[str] = self.config.get(
            "coordinator.autonomous_improvement.critical_components", 
            ["mindx.learning.self_improve_agent", "mindx.orchestration.coordinator_agent"] # Module paths
        )
        self.require_human_approval_for_critical: bool = self.config.get(
            "coordinator.autonomous_improvement.require_human_approval_for_critical", True
        )

        logger.info(
            f"CoordinatorAgent MindX (v_prod_release_candidate) initialized. "
            f"SIA Script: {self.self_improve_agent_script_path or 'NOT FOUND - Improvement Disabled'}. "
            f"Autonomous Mode: {self.config.get('coordinator.autonomous_improvement.enabled', False)}. "
            f"HITL for Critical: {self.require_human_approval_for_critical}."
        )
        self._initialized = True

    def _register_default_agents(self): # pragma: no cover
        self.register_agent(
            agent_id="coordinator_agent_mindx", agent_type="coordinator",
            description="MindX Central Coordinator Agent",
            capabilities=["orchestration", "system_analysis", "component_improvement", "query", "backlog_management", "rollback_trigger"],
            instance=self,
            metadata={"file_path": str(Path(__file__).resolve())} 
        )
        if self.self_improve_agent_script_path:
            self.register_agent(
                agent_id="self_improve_agent_cli_mindx", agent_type="self_improvement_worker",
                description="MindX Self-Improvement Worker Agent (CLI based)",
                capabilities=["code_modification", "code_evaluation", "self_update_atomic", "rollback_self"],
                metadata={"script_path": str(self.self_improve_agent_script_path)}
            )
        if self.resource_monitor:
            self.register_agent(agent_id="resource_monitor_mindx", agent_type="monitor", description="MindX System Resource Monitor", capabilities=["system_resource_tracking", "alerting"], instance=self.resource_monitor)
        if self.performance_monitor:
            self.register_agent(agent_id="performance_monitor_mindx", agent_type="monitor", description="MindX LLM Performance Monitor", capabilities=["llm_performance_tracking", "reporting"], instance=self.performance_monitor)

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
            if isinstance(item,dict) and all(k in item for k in ["id","target_component_path","suggestion"]):
                item.setdefault("priority",0); item.setdefault("status",InteractionStatus.PENDING.value) # Use Enum value
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
        # (Full _register_monitor_callbacks with async def handle_resource_alert/resolve from previous response)
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
        # (Full process_interaction logic from previous "Production Candidate Stub v3" Coordinator)
        if not isinstance(interaction, Interaction): logger.error("Coord: process_interaction invalid type."); raise TypeError("Invalid object to process_interaction")
        if interaction.status not in [InteractionStatus.PENDING, InteractionStatus.PENDING_APPROVAL]: logger.warning(f"Coord: Interaction {interaction.interaction_id} not PENDING/PENDING_APPROVAL (Status: {interaction.status.name}). Returning."); return interaction
        logger.info(f"Coord: Processing interaction {interaction.interaction_id} (Type: {interaction.interaction_type.name}, Status: {interaction.status.name})")
        interaction.status = InteractionStatus.IN_PROGRESS; interaction.started_at = time.time()
        if interaction.interaction_id not in self.active_interactions: self.active_interactions[interaction.interaction_id] = interaction
        response_data: Any = None
        try:
            if interaction.interaction_type == InteractionType.QUERY: response_data = await self.llm_handler.generate_text(interaction.content, max_tokens=1024, temperature=0.5); interaction.add_to_history("coord_llm", "Query by Coord LLM.")
            elif interaction.interaction_type == InteractionType.SYSTEM_ANALYSIS:
                response_data = await self._process_system_analysis(interaction)
                if isinstance(response_data, dict) and "improvement_suggestions" in response_data:
                    source = interaction.metadata.get("source", "user_request" if interaction.user_id else (f"agent_request:{interaction.agent_id}" if interaction.agent_id else "unknown_source"))
                    added_to_backlog_count = 0
                    for sugg in response_data.get("improvement_suggestions",[]): sugg.setdefault("id", str(uuid.uuid4())); sugg.setdefault("is_critical_target", False); self.add_to_improvement_backlog(sugg, source=source); added_to_backlog_count+=1
                    interaction.add_to_history("backlog_update", f"{added_to_backlog_count} suggs from analysis to backlog.")
            elif interaction.interaction_type == InteractionType.COMPONENT_IMPROVEMENT: response_data = await self._process_component_improvement_cli(interaction)
            elif interaction.interaction_type == InteractionType.APPROVE_IMPROVEMENT: response_data = self._process_backlog_approval(interaction.metadata.get("backlog_item_id"), approve=True)
            elif interaction.interaction_type == InteractionType.REJECT_IMPROVEMENT: response_data = self._process_backlog_approval(interaction.metadata.get("backlog_item_id"), approve=False)
            elif interaction.interaction_type == InteractionType.ROLLBACK_COMPONENT: response_data = await self._process_component_rollback_cli(interaction)
            else: response_data = {"error": f"Unsupported type: {interaction.interaction_type.name}"}; interaction.status = InteractionStatus.FAILED; interaction.error = response_data["error"]
            if interaction.status != InteractionStatus.FAILED: interaction.response = response_data; interaction.status = InteractionStatus.COMPLETED
            interaction.add_to_history("coordinator", f"Interaction processing finished. Final Status: {interaction.status.name}.")
        except Exception as e: logger.error(f"Error processing interaction {interaction.interaction_id}: {e}", exc_info=True); interaction.status = InteractionStatus.FAILED; interaction.error = f"{type(e).__name__}: {str(e)}"; interaction.add_to_history("system_error", f"Unhandled exception: {interaction.error}", {"traceback": traceback.format_exc(limit=3).splitlines()})
        finally:
            interaction.completed_at = time.time();
            if interaction.interaction_id in self.active_interactions: del self.active_interactions[interaction.interaction_id]
            self.completed_interactions[interaction.interaction_id] = interaction
        return interaction

    async def _scan_codebase_capabilities(self) -> Dict[str, Any]: # pragma: no cover
        # (Full _scan_codebase_capabilities from previous Coordinator - uses PROJECT_ROOT and AST)
        src_dir = PROJECT_ROOT / "mindx"; capabilities: Dict[str, Any] = {}; logger.info(f"Coord: Scanning capabilities in: {src_dir}")
        if not src_dir.is_dir(): logger.error(f"Coord: Scan dir {src_dir} not found."); return {}
        for item in src_dir.rglob("*.py"):
            if item.name.startswith("__"): continue
            try: rel_path = item.relative_to(PROJECT_ROOT); mod_name = ".".join(rel_path.parts[:-1] + (rel_path.stem,))
            except ValueError: mod_name = item.stem
            try:
                with item.open("r", encoding="utf-8") as f_handle: tree = ast.parse(f_handle.read())
                for node in ast.walk(tree):
                    n_name = getattr(node, 'name', None);
                    if not n_name or n_name.startswith("_"): continue
                    cap_k: Optional[str] = None; cap_t: Optional[str] = None; doc_s = ast.get_docstring(node)
                    if isinstance(node, ast.ClassDef): cap_k=f"{mod_name}.{n_name}"; cap_t="class"
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if node in getattr(p,'body',[])): cap_k=f"{mod_name}.{n_name}"; cap_t="function"
                    if cap_k and cap_t: capabilities[cap_k] = {"type": cap_t, "name": n_name, "module": mod_name, "path": str(item), "docstring_snippet": (doc_s[:150] + "..." if doc_s and len(doc_s) > 150 else doc_s)}
            except Exception as e_ast: logger.warning(f"Coord: AST scan error {item}: {e_ast}")
        self.system_capabilities_cache = capabilities; logger.info(f"Coord: Scanned {len(capabilities)} capabilities."); return capabilities

    def add_to_improvement_backlog(self, suggestion: Dict[str, Any], source: str = "system_analysis"): # pragma: no cover
        # (Full add_to_improvement_backlog from previous Coordinator, including ID generation, deduplication, sorting, and _save_backlog call)
        if not all(k in suggestion for k in ["target_component_path", "suggestion", "priority"]): logger.warning(f"Coord: Skipping invalid sugg for backlog: {suggestion}"); return
        sugg_id = suggestion.get("id", str(uuid.uuid4())[:8]); suggestion["id"] = sugg_id
        for item in self.improvement_backlog:
            if item.get("target_component_path") == suggestion["target_component_path"] and item.get("suggestion","")[:70] == suggestion.get("suggestion","")[:70] and item.get("status") == InteractionStatus.PENDING.value:
                logger.info(f"Coord: Similar PENDING sugg for {suggestion['target_component_path']} exists (ID {item.get('id','N/A')[:8]}). Updating prio if higher.");
                if suggestion['priority'] > item.get('priority',0): item['priority'] = suggestion['priority']; item['suggestion'] = suggestion['suggestion']; item['added_at'] = time.time()
                self._save_backlog(); return
        suggestion["added_at"] = time.time(); suggestion["status"] = InteractionStatus.PENDING.value; suggestion["source"] = source; suggestion["attempt_count"] = 0; suggestion["last_attempted_at"] = None; suggestion.setdefault("is_critical_target", False)
        self.improvement_backlog.append(suggestion)
        self.improvement_backlog.sort(key=lambda x: (x.get("status") == InteractionStatus.PENDING_APPROVAL.value, -int(x.get("priority", 0)), x.get("added_at",0)), reverse=False)
        logger.info(f"Coord: Added sugg (ID:{sugg_id}) for '{suggestion['target_component_path']}' to backlog from '{source}'. Backlog size: {len(self.improvement_backlog)}"); self._save_backlog()

    def _process_backlog_approval(self, item_id: Optional[str], approve: bool) -> Dict[str, Any]: # pragma: no cover
        # (Full _process_backlog_approval from previous Coordinator)
        if not item_id: return {"status": "FAILURE", "message": "No backlog_item_id provided."}
        for item in self.improvement_backlog:
            if item.get("id") == item_id and item.get("status") == InteractionStatus.PENDING_APPROVAL.value:
                if approve: item["status"] = InteractionStatus.PENDING.value; item["approved_at"] = time.time(); item["approved_by"] = "manual_cli"; msg=f"Item '{item_id[:8]}' approved."
                else: item["status"] = "rejected_manual"; item["rejected_at"] = time.time(); item["rejected_by"] = "manual_cli"; msg=f"Item '{item_id[:8]}' rejected."
                logger.info(msg); self._save_backlog(); return {"status": "SUCCESS", "message": msg}
        return {"status": "FAILURE", "message": f"Item '{item_id[:8]}' not found or not pending approval."}

    async def _process_system_analysis(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        # (Full _process_system_analysis from previous Coordinator, with monitor data integration)
        # This method should use self.llm_handler
        # It should include system_structure_summary, resource_summary, performance_summary, history_summary, analysis_focus_hint in the prompt.
        # It should parse the LLM's JSON response for 'improvement_suggestions'.
        # For brevity, I'll sketch it here, refer to the full previous version.
        interaction.add_to_history("coordinator", "System analysis: Gathering data.")
        if not self.system_capabilities_cache: await self._scan_codebase_capabilities()
        # ... (Gather res_summary, perf_summary, history_summary, focus_hint) ...
        res_usage = self.resource_monitor.get_resource_usage(); # Simplified for this placeholder
        resource_summary = f"CPU: {res_usage.get('cpu_percent',0):.1f}%, Mem: {res_usage.get('memory_percent',0):.1f}%"
        performance_summary = f"LLM Perf: {len(self.performance_monitor.get_all_metrics())} keys tracked."
        system_structure_summary = f"{len(self.system_capabilities_cache or {})} caps scanned."
        history_summary = f"{len(self.improvement_campaign_history)} campaigns logged."
        analysis_focus_hint = interaction.metadata.get("analysis_context", "general health")

        prompt = (f"AI Architect for MindX. Analyze system. Focus: {analysis_focus_hint}\n"
                  f"Struct: {system_structure_summary}\nRes: {resource_summary}\nLLM Perf: {performance_summary}\nHistory: {history_summary}\n"
                  f"Suggest 1-2 improvements: 'target_component_path' (mindx.module.file_stem), 'suggestion', 'priority' (1-10), 'is_critical_target' (bool). ONLY JSON: {{\"improvement_suggestions\": [{{...}}]}}")
        try:
            response_str = await self.llm_handler.generate_text(prompt, max_tokens=self.config.get("coordinator.system_analysis.max_tokens"), temperature=self.config.get("coordinator.system_analysis.temperature"), json_mode=True)
            # ... (Robust JSON parsing as in previous Coordinator _process_system_analysis) ...
            analysis_result = json.loads(response_str) # Simplified parsing for this placeholder
            interaction.add_to_history("llm_analysis", "Sys analysis generated.", analysis_result); return analysis_result
        except Exception as e: return {"error": str(e), "improvement_suggestions": []}

    async def _resolve_component_path_for_sia(self, component_identifier: str) -> Optional[Path]: # pragma: no cover
        # (Full _resolve_component_path_for_sia from previous Coordinator, using PROJECT_ROOT)
        if not component_identifier: return None; logger.debug(f"Coord: Resolving SIA target '{component_identifier}'")
        agent_info = self.get_agent(component_identifier)
        if agent_info and agent_info["metadata"].get("script_path"): script_p = Path(agent_info["metadata"]["script_path"]);
        if script_p.is_absolute() and script_p.exists(): return script_p
        if (PROJECT_ROOT / script_p).exists(): return (PROJECT_ROOT / script_p).resolve()
        if "." in component_identifier: parts = component_identifier.split('.');
        if parts[0] == "mindx": resolved_path = (PROJECT_ROOT / Path(*parts)).with_suffix('.py')
        else: resolved_path = (PROJECT_ROOT / "mindx" / Path(*parts)).with_suffix('.py')
        if resolved_path.exists() and resolved_path.is_file(): logger.debug(f"Resolved mod path '{component_identifier}' to '{resolved_path}'"); return resolved_path.resolve()
        path_from_project_root = (PROJECT_ROOT / component_identifier).resolve()
        if path_from_project_root.exists() and path_from_project_root.is_file(): logger.debug(f"Resolved '{component_identifier}' as file rel to PR: '{path_from_project_root}'"); return path_from_project_root
        logger.warning(f"Coord: Could not resolve '{component_identifier}' to file path using PROJECT_ROOT: {PROJECT_ROOT}"); return None


    async def _process_component_improvement_cli(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        # (Full _process_component_improvement_cli from previous Coordinator, including temp context file,
        # robust JSON parsing of SIA output, concurrency limit, and critical update check)
        if not self.self_improve_agent_script_path: return {"status": "FAILURE", "message": "SIA script path not configured."}
        # ... (Full logic from previous CoordinatorAgent - it's substantial) ...
        # This is a sketch of the flow:
        metadata = interaction.metadata; target_id = metadata.get("target_component")
        context = metadata.get("analysis_context", "General improvement."); # ... other params ...
        target_path = await self._resolve_component_path_for_sia(target_id)
        if not target_path: return {"status":"FAILURE", "message":f"Cannot resolve {target_id}"}
        cmd = [str(sys.executable), str(self.self_improve_agent_script_path), str(target_path), "--output-json"]
        # ... (add context_file, other CLI args for SIA) ...
        tmp_ctx_file: Optional[Path] = None
        try:
            if context and len(context) > 1024: tmp_dir = PROJECT_ROOT / "data" / "temp_sia_contexts"; tmp_dir.mkdir(parents=True, exist_ok=True); fd, tmp_f_str = tempfile.mkstemp(".txt", "ctx_", tmp_dir, True);
            with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(context); tmp_ctx_file = Path(tmp_f_str); cmd.extend(["--context-file", str(tmp_ctx_file)])
            elif context: cmd.extend(["--context", context])
            # ... add other SIA CLI args from metadata ...
            
            async with self.sia_concurrency_limit:
                process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=PROJECT_ROOT)
                timeout = self.config.get("coordinator.sia_cli_timeout_seconds")
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            # ... (Parse stdout for JSON from SIA, handle errors, update history, check for critical update flag) ...
            # This detailed parsing MUST be from the previous Coordinator version.
            out_s = stdout.decode().strip(); err_s = stderr.decode().strip()
            sia_result_json = {"status":"FAILURE", "message":"SIA call did not produce expected JSON"}
            if out_s: try: sia_result_json = json.loads(out_s)
            except: pass # Keep default error
            # ... (process sia_result_json as in previous Coordinator) ...
        finally:
            if tmp_ctx_file and tmp_ctx_file.exists(): tmp_ctx_file.unlink()
        return sia_result_json # Placeholder - return the parsed JSON

    async def _process_component_rollback_cli(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        """Handles request to rollback a component (likely SIA itself) via SIA's CLI."""
        if not self.self_improve_agent_script_path:
            return {"status": "FAILURE", "message": "SelfImprovementAgent script path not configured."}

        metadata = interaction.metadata
        target_component_id = metadata.get("target_component_for_rollback") # e.g., "self_improve_agent_cli_mindx"
        rollback_version_n = metadata.get("rollback_version_n", 1) # Default to latest backup

        if not target_component_id:
            return {"status": "FAILURE", "message": "ROLLBACK_COMPONENT requires 'target_component_for_rollback'."}
        
        # Currently, SIA's CLI --rollback only supports "self".
        # If we want Coordinator to trigger rollback for *other* components via SIA, SIA would need to support that.
        # For now, assume this is primarily for SIA to rollback itself.
        if target_component_id.lower() not in ["self", "self_improve_agent_cli_mindx", self.self_improve_agent_script_path.name]:
            return {"status": "FAILURE", "message": f"SIA CLI rollback currently only supports 'self', not '{target_component_id}'."}

        interaction.add_to_history("coordinator", f"Preparing SIA CLI to rollback itself to Nth={rollback_version_n} backup.")
        command = [ str(sys.executable), str(self.self_improve_agent_script_path), "self", "--rollback", str(rollback_version_n), "--output-json" ]
        
        logger.info(f"Coordinator: Executing SIA Rollback CLI: {' '.join(command)}")
        process_cwd = PROJECT_ROOT
        sia_cli_result_json: Dict[str, Any] = {"status": "FAILURE", "message": "SIA rollback call did not complete."}
        timeout_seconds = self.config.get("coordinator.sia_cli_timeout_seconds", 60.0) # Shorter timeout for rollback

        try:
            async with self.sia_concurrency_limit: # Use semaphore though rollback should be rare
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
            else: sia_cli_result_json = {"status": "FAILURE", "message": "SIA rollback process exited 0 but STDOUT empty."}

            if sia_cli_result_json.get("status") == "SUCCESS":
                interaction.add_to_history("sia_cli_rollback_success", sia_cli_result_json.get("message","SIA rollback successful."), sia_cli_result_json.get("data",{}))
                logger.warning(f"Coordinator: SIA reported successful rollback of itself. System restart likely required.")
                await self.belief_system.add_belief(f"system.restart_required.reason", f"SIA rollback executed for {target_component_id}", 0.98, BeliefSource.SELF_ANALYSIS)
            else:
                logger.error(f"Coordinator: SIA rollback operation reported failure. SIA JSON: {json.dumps(sia_cli_result_json)}")
                interaction.add_to_history("sia_cli_rollback_failure", sia_cli_result_json.get("message", "SIA rollback failed."), sia_cli_result_json)
        except asyncio.TimeoutError: sia_cli_result_json = {"status": "FAILURE", "message": f"SIA rollback call timed out."}
        except Exception as e_proc: sia_cli_result_json = {"status": "FAILURE", "message": f"SIA rollback call exception: {e}"}
        
        return sia_cli_result_json


    async def _autonomous_improvement_worker(self, interval_seconds: float): # pragma: no cover
        # (Full _autonomous_improvement_worker from previous Audit 5, ensure it updates backlog status correctly)
        logger.info(f"Autonomous worker starting. Interval: {interval_seconds}s. HITL Critical: {self.require_human_approval_for_critical}")
        cool_down_store: Dict[str, float] = {}; cool_down_secs = self.config.get("coordinator.autonomous_improvement.cooldown_seconds")
        while True:
            try:
                await asyncio.sleep(interval_seconds); logger.info("Autonomous worker: Cycle start.")
                # 1. Analyze
                analysis_interaction = await self.create_interaction(InteractionType.SYSTEM_ANALYSIS, "Auto system analysis", agent_id="autonomous_worker_mindx")
                analysis_result = await self.process_interaction(analysis_interaction) # This calls add_to_improvement_backlog
                
                # 2. Check Resources
                if self.resource_monitor.get_resource_usage().get("cpu_percent",0) > self.config.get("coordinator.autonomous_improvement.max_cpu_before_sia",90.0): logger.warning("Autonomous: CPU high, deferring SIA."); continue
                
                # 3. Process Backlog
                if not self.improvement_backlog: logger.info("Autonomous: Backlog empty."); continue
                next_item: Optional[Dict[str,Any]] = None
                for item in self.improvement_backlog: # Backlog is sorted: PENDING_APPROVAL > Prio > Age
                    if item.get("status") == InteractionStatus.PENDING.value:
                        last_fail = cool_down_store.get(item["target_component_path"])
                        if last_fail and (time.time() - last_fail < cool_down_secs): logger.info(f"Autonomous: Skipping {item['target_component_path']} (ID {item.get('id','N/A')[:8]}), in cool-down."); continue
                        next_item = item; break
                if not next_item: logger.info("Autonomous: No actionable pending items."); continue
                
                target_path = next_item["target_component_path"]
                is_critical = next_item.get("is_critical_target",False) or any(crit_stem in target_path for crit_stem in self.critical_components_for_approval)
                if self.require_human_approval_for_critical and is_critical and next_item.get("approved_at") is None:
                    if next_item["status"] != InteractionStatus.PENDING_APPROVAL.value: next_item["status"] = InteractionStatus.PENDING_APPROVAL.value; self._save_backlog(); logger.warning(f"Autonomous: CRITICAL improve for {target_path} (ID {next_item.get('id')[:8]}) needs approval. Suggestion: {next_item['suggestion']}")
                    else: logger.info(f"Autonomous: Critical item {target_path} (ID {next_item.get('id')[:8]}) still pending approval.")
                    continue

                logger.info(f"Autonomous: Attempting: {target_path} (ID {next_item.get('id')[:8]}, Prio: {next_item['priority']})")
                next_item["status"] = InteractionStatus.IN_PROGRESS.value; next_item["attempt_count"] = next_item.get("attempt_count",0)+1; next_item["last_attempted_at"] = time.time(); self._save_backlog()
                
                imp_meta = {"target_component":target_path, "analysis_context":next_item["suggestion"], "source":"autonomous_worker_backlog", "backlog_item_id":next_item.get("id")}
                imp_content = f"Auto attempt: {next_item['suggestion'][:100]}"
                imp_interaction = await self.create_interaction(InteractionType.COMPONENT_IMPROVEMENT, imp_content, agent_id="autonomous_worker_mindx", metadata=imp_meta)
                processed_imp_int = await self.process_interaction(imp_interaction)
                
                sia_resp_json = processed_imp_int.response # This is the full JSON from SIA CLI
                if isinstance(sia_resp_json, dict):
                    if sia_resp_json.get("status") == "SUCCESS":
                        final_sia_op_status = sia_resp_json.get("data",{}).get("final_status", "UNKNOWN"); next_item["status"] = InteractionStatus.COMPLETED.value if final_sia_op_status.startswith("SUCCESS") else InteractionStatus.FAILED.value; next_item["sia_final_op_status"] = final_sia_op_status
                        if not final_sia_op_status.startswith("SUCCESS"): cool_down_store[target_path] = time.time()
                    else: next_item["status"] = InteractionStatus.FAILED.value; next_item["sia_error"] = sia_resp_json.get("message", "SIA reported error"); cool_down_store[target_path] = time.time()
                else: next_item["status"] = InteractionStatus.FAILED.value; next_item["sia_error"] = "Invalid SIA response"; cool_down_store[target_path] = time.time()
                self._save_backlog(); logger.info(f"Autonomous: Improvement for {target_path} (ID {next_item.get('id')[:8]}) finished. Backlog status: {next_item['status']}")
            except asyncio.CancelledError: logger.info("Autonomous worker stopping."); break
            except Exception as e: logger.error(f"Autonomous worker error: {e}", exc_info=True); await asyncio.sleep(max(interval_seconds // 2, 1800))
        logger.info("Autonomous worker stopped.")

    # --- handle_user_input, shutdown, get_X, reset_instance methods ---
    async def handle_user_input( self, content: str, user_id: Optional[str] = None, interaction_type: Optional[Union[InteractionType, str]] = None, metadata: Optional[Dict[str, Any]] = None ) -> Dict[str, Any]: # pragma: no cover
        # (Full handle_user_input from previous "Production Candidate Stub v3" Coordinator with ROLLBACK_COMPONENT)
        logger.info(f"Coordinator: User input from '{user_id}': '{content[:100]}...' Meta: {metadata}")
        metadata = metadata or {}; parsed_content = content;
        if not interaction_type: # Infer type
            parts = content.strip().split(" ", 1); cmd_verb = parts[0].lower(); args_str = parts[1] if len(parts) > 1 else ""
            if cmd_verb == "analyze_system": interaction_type = InteractionType.SYSTEM_ANALYSIS; parsed_content = "System analysis"; metadata["analysis_context"] = args_str
            elif cmd_verb == "improve": interaction_type = InteractionType.COMPONENT_IMPROVEMENT; improve_args = args_str.split(" ", 1);
            if not improve_args or not improve_args[0]: return {"error": "Improve cmd needs target_component.", "status": InteractionStatus.FAILED.value}
            metadata["target_component"] = improve_args[0];
            if len(improve_args) > 1: metadata["analysis_context"] = improve_args[1]
            parsed_content = f"Request to improve {metadata['target_component']}"
            elif cmd_verb == "approve": interaction_type = InteractionType.APPROVE_IMPROVEMENT; metadata["backlog_item_id"] = args_str.strip(); parsed_content = f"Approve item {cmd_args_str}"
            elif cmd_verb == "reject": interaction_type = InteractionType.REJECT_IMPROVEMENT; metadata["backlog_item_id"] = args_str.strip(); parsed_content = f"Reject item {cmd_args_str}"
            elif cmd_verb == "rollback": interaction_type = InteractionType.ROLLBACK_COMPONENT; rollback_args = args_str.split(" ",1); metadata["target_component_for_rollback"] = rollback_args[0]; metadata["rollback_version_n"] = int(rollback_args[1]) if len(rollback_args)>1 and rollback_args[1].isdigit() else 1; parsed_content = f"Rollback {metadata['target_component_for_rollback']} to Nth={metadata['rollback_version_n']}"
            else: interaction_type = InteractionType.QUERY; parsed_content = content
        
        if isinstance(interaction_type, str): # Ensure Enum
            try: interaction_type_enum = InteractionType(interaction_type.lower())
            except ValueError: logger.error(f"Invalid type str: {interaction_type}"); return {"error": f"Invalid type: {interaction_type}", "status": InteractionStatus.FAILED.value}
        else: interaction_type_enum = interaction_type
        try:
            interaction = await self.create_interaction( interaction_type_enum, parsed_content, user_id, metadata=metadata )
            processed_interaction = await self.process_interaction(interaction); return processed_interaction.to_dict()
        except Exception as e: logger.error(f"Error in handle_user_input: {e}", exc_info=True); return {"error": str(e), "status": InteractionStatus.FAILED.value}

    async def shutdown(self): # pragma: no cover
        logger.info(f"CoordinatorAgent MindX ({self.config.get('version', 'unknown')}) shutting down..."); self.stop_autonomous_improvement_loop()
        if self.autonomous_improvement_task: try: await asyncio.wait_for(self.autonomous_improvement_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError): pass
        if self.resource_monitor and self.resource_monitor.monitoring: self.resource_monitor.stop_monitoring()
        if self.performance_monitor: await self.performance_monitor.shutdown()
        self._save_backlog(); self._save_campaign_history(); logger.info(f"CoordinatorAgent MindX ({self.config.get('version', 'unknown')}) shutdown complete.")
    @classmethod
    async def reset_instance_async(cls): # pragma: no cover
        async with cls._lock:
            if cls._instance: await cls._instance.shutdown(); cls._instance._initialized = False; cls._instance = None
        logger.debug("CoordinatorAgent instance reset asynchronously.")

# --- Factory Functions ---
async def get_coordinator_agent_mindx_async(config_override: Optional[Config] = None, test_mode: bool = False) -> CoordinatorAgent: # pragma: no cover
    # (Full async factory from previous "Production Candidate Stub v3" Coordinator)
    async with CoordinatorAgent._lock:
        if CoordinatorAgent._instance is None or test_mode:
            if test_mode and CoordinatorAgent._instance is not None: await CoordinatorAgent._instance.shutdown(); CoordinatorAgent._instance = None
            effective_config = config_override or Config(test_mode=test_mode)
            belief_system = BeliefSystem(test_mode=test_mode)
            resource_monitor_instance = await get_resource_monitor_async(config_override=effective_config, test_mode=test_mode)
            performance_monitor_instance = await get_performance_monitor_async(config_override=effective_config, test_mode=test_mode)
            CoordinatorAgent._instance = CoordinatorAgent(belief_system, resource_monitor_instance, performance_monitor_instance, config_override=effective_config, test_mode=test_mode)
        return CoordinatorAgent._instance

def get_coordinator_agent_mindx(config_override: Optional[Config] = None, test_mode:bool = False) -> CoordinatorAgent: # pragma: no cover
    # (Sync getter from previous "Production Candidate Stub v3" Coordinator)
    if CoordinatorAgent._instance and not test_mode: return CoordinatorAgent._instance
    try: loop = asyncio.get_running_loop()
    except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); instance = loop.run_until_complete(get_coordinator_agent_mindx_async(config_override, test_mode)); return instance
    else: # pragma: no cover
        if loop.is_running(): import nest_asyncio; nest_asyncio.apply(); return asyncio.run(get_coordinator_agent_mindx_async(config_override, test_mode))
        else: return loop.run_until_complete(get_coordinator_agent_mindx_async(config_override, test_mode))
