# mindx/orchestration/coordinator_agent.py
import os
import logging # Standard logging, configured by logging_config
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

# Use canonical PROJECT_ROOT from config
from mindx.utils.config import Config, PROJECT_ROOT
from mindx.core.belief_system import BeliefSystem, BeliefSource # BeliefSource used for adding beliefs
from mindx.monitoring.resource_monitor import get_resource_monitor_async, ResourceMonitor, ResourceType # Use async getter
from mindx.monitoring.performance_monitor import get_performance_monitor_async, PerformanceMonitor # Use async getter
from mindx.llm.llm_factory import create_llm_handler, LLMHandler
# Import stubs for other agents to allow type hinting and basic registration
# from .multimodel_agent import MultiModelAgent # Example, if used
# from .model_selector import ModelSelector   # Example
# from ..core.bdi_agent import BDIAgent      # Example
# from ..docs.documentation_agent import DocumentationAgent # Example

logger = logging.getLogger(__name__) # Use MindX standard logger

class InteractionType(Enum): # pragma: no cover
    """Defines the types of interactions the Coordinator can handle."""
    QUERY = "query"                             # General purpose query, likely to an LLM
    SYSTEM_ANALYSIS = "system_analysis"         # Request for a high-level analysis of the MindX system
    COMPONENT_IMPROVEMENT = "component_improvement" # Request to improve a specific code component
    APPROVE_IMPROVEMENT = "approve_improvement" # Human/Agent approval for a pending critical improvement
    REJECT_IMPROVEMENT = "reject_improvement"   # Human/Agent rejection for a pending critical improvement

class InteractionStatus(Enum): # pragma: no cover
    """Defines the possible statuses of an interaction."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled" 
    PENDING_APPROVAL = "pending_approval" 

class Interaction: # pragma: no cover
    """
    Represents a single interaction (task, query, request) managed by the Coordinator.
    Includes history tracking for the interaction's lifecycle.
    """
    def __init__( self, 
                  interaction_id: str, 
                  interaction_type: InteractionType, 
                  content: str, 
                  user_id: Optional[str] = None, 
                  agent_id: Optional[str] = None, 
                  metadata: Optional[Dict[str, Any]] = None ):
        self.interaction_id = interaction_id
        self.interaction_type = interaction_type
        self.content = content 
        self.user_id = user_id
        self.agent_id = agent_id 
        self.metadata = metadata or {}
        self.status = InteractionStatus.PENDING
        self.response: Any = None 
        self.error: Optional[str] = None
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.history: List[Dict[str, Any]] = []

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
        interaction = cls(
            interaction_id=data["interaction_id"], interaction_type=InteractionType(data["interaction_type"]),
            content=data["content"], user_id=data.get("user_id"), agent_id=data.get("agent_id"),
            metadata=data.get("metadata", {}) )
        interaction.status = InteractionStatus(data.get("status", InteractionStatus.PENDING.value))
        interaction.response = data.get("response"); interaction.error = data.get("error")
        interaction.created_at = data.get("created_at", time.time()); interaction.started_at = data.get("started_at")
        interaction.completed_at = data.get("completed_at"); interaction.history = data.get("history", [])
        return interaction

    def __repr__(self):
        return f"<Interaction id='{self.interaction_id}' type={self.interaction_type.name} status={self.status.name}>"


class CoordinatorAgent:
    """
    Central orchestrator for the MindX system (Augmentic Project).
    Manages system-level analysis, delegates component improvements to the
    SelfImprovementAgent (SIA) via its CLI, handles an improvement backlog,
    and can run an autonomous improvement loop with optional human-in-the-loop.
    """
    _instance = None
    _lock = asyncio.Lock() # Class-level lock for singleton creation

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
            logger.critical(
                f"SelfImprovementAgent script not found at expected location: {potential_sia_path}. "
                "Component improvement features will be disabled."
            )
        
        self._register_default_agents()
        self.sia_concurrency_limit = asyncio.Semaphore(
            self.config.get("coordinator.max_concurrent_sia_tasks", 1)
        )

        # Resource monitoring is started by the factory that creates this Coordinator
        if self.resource_monitor.monitoring: # Check if it was started by factory
            self._register_monitor_callbacks()
        
        self.autonomous_improvement_task: Optional[asyncio.Task] = None
        if self.config.get("coordinator.autonomous_improvement.enabled", False) and not test_mode: # pragma: no cover
            self.start_autonomous_improvement_loop()
        
        self.critical_components_for_approval: List[str] = self.config.get(
            "coordinator.autonomous_improvement.critical_components", 
            ["mindx.learning.self_improve_agent", "mindx.orchestration.coordinator_agent"] # Default criticals
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
        """Registers this coordinator and its knowledge of other key system parts."""
        self.register_agent(
            agent_id="coordinator_agent_mindx", agent_type="coordinator",
            description="MindX Central Coordinator Agent",
            capabilities=["orchestration", "system_analysis", "component_improvement", "query", "backlog_management"],
            instance=self,
            metadata={"file_path": str(Path(__file__).resolve())} # Its own file path
        )
        if self.self_improve_agent_script_path:
            self.register_agent(
                agent_id="self_improve_agent_cli_mindx", agent_type="self_improvement_worker",
                description="MindX Self-Improvement Worker Agent (CLI based)",
                capabilities=["code_modification", "code_evaluation", "self_update_atomic"],
                metadata={"script_path": str(self.self_improve_agent_script_path)}
            )
        if self.resource_monitor:
            self.register_agent(
                agent_id="resource_monitor_mindx", agent_type="monitor",
                description="MindX System Resource Monitor",
                capabilities=["system_resource_tracking", "alerting"], instance=self.resource_monitor
            )
        if self.performance_monitor:
            self.register_agent(
                agent_id="performance_monitor_mindx", agent_type="monitor",
                description="MindX LLM Performance Monitor",
                capabilities=["llm_performance_tracking", "reporting"], instance=self.performance_monitor
            )

    def _load_json_file(self, file_name: str, default_value: Union[List, Dict]) -> Union[List, Dict]: # pragma: no cover
        """Loads a JSON file from the PROJECT_ROOT/data directory."""
        file_path = PROJECT_ROOT / "data" / file_name
        if file_path.exists():
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Coordinator: Error loading {file_name} from {file_path}: {e}")
        return default_value

    def _save_json_file(self, file_name: str, data: Union[List, Dict]): # pragma: no cover
        """Saves data to a JSON file in the PROJECT_ROOT/data directory."""
        file_path = PROJECT_ROOT / "data" / file_name
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Coordinator: Saved data to {file_path}")
        except Exception as e:
            logger.error(f"Coordinator: Error saving {file_name} to {file_path}: {e}")

    def _load_backlog(self) -> List[Dict[str, Any]]: # pragma: no cover
        loaded_items = self._load_json_file("improvement_backlog.json", [])
        valid_items = []
        for item in loaded_items:
            if isinstance(item, dict) and all(k in item for k in ["id", "target_component_path", "suggestion"]):
                item.setdefault("priority", 0); item.setdefault("status", "pending")
                item.setdefault("added_at", time.time()); item.setdefault("attempt_count",0)
                item.setdefault("is_critical_target", item.get("is_critical_target", False)) # Preserve if loaded
                valid_items.append(item)
            else: logger.warning(f"Coordinator: Discarding malformed backlog item: {str(item)[:100]}")
        # Initial sort on load
        valid_items.sort(key=lambda x: (x.get("status") == InteractionStatus.PENDING_APPROVAL.value, -int(x.get("priority", 0)), x.get("added_at",0)), reverse=False)
        return valid_items
    
    def _save_backlog(self): # pragma: no cover
        self._save_json_file("improvement_backlog.json", self.improvement_backlog)

    def _load_campaign_history(self) -> List[Dict[str, Any]]: # pragma: no cover
        return self._load_json_file("improvement_campaign_history.json", [])

    def _save_campaign_history(self): # pragma: no cover
        self._save_json_file("improvement_campaign_history.json", self.improvement_campaign_history)

    def _register_monitor_callbacks(self): # pragma: no cover
        """Registers internal handlers for alerts from the ResourceMonitor."""
        async def handle_resource_alert(monitor_instance: ResourceMonitor, 
                                         resource_type: ResourceType, 
                                         current_value: float, 
                                         path_if_disk: Optional[str] = None):
            alert_key_base = f"system_health.{resource_type.value}.alert_active" # Use a consistent key for active alerts
            alert_key = f"{alert_key_base}.{Path(path_if_disk).name.replace('.', '_')}" if path_if_disk else alert_key_base
            
            logger.warning(f"Coordinator CB: HIGH RESOURCE USAGE DETECTED - {resource_type.name} at {current_value:.1f}%" + (f" for path '{path_if_disk}'" if path_if_disk else ""))
            # Add a belief that this resource is currently in an alert state
            await self.belief_system.add_belief(
                alert_key,
                {"percent": current_value, "path": path_if_disk, "timestamp": time.time(), 
                 "threshold": monitor_instance.get_resource_limits().get(f"max_{resource_type.value}_percent", "N/A") if resource_type != ResourceType.DISK else monitor_instance.get_resource_limits().get("disk_threshold_map",{}).get(path_if_disk, "N/A") },
                0.9, BeliefSource.PERCEPTION,
                ttl_seconds=self.config.get("coordinator.autonomous_improvement.cooldown_seconds", 3600.0) * 2 # Alert belief persists longer
            )

        async def handle_resource_resolve(monitor_instance: ResourceMonitor, resource_type: ResourceType, current_value: float, path_if_disk: Optional[str] = None):
            alert_key_base = f"system_health.{resource_type.value}.alert_active"
            alert_key_to_clear = f"{alert_key_base}.{Path(path_if_disk).name.replace('.', '_')}" if path_if_disk else alert_key_base

            logger.info(f"Coordinator CB: RESOURCE USAGE RESOLVED - {resource_type.name} now at {current_value:.1f}%" + (f" for path '{path_if_disk}'" if path_if_disk else ""))
            # Remove the corresponding active alert belief
            await self.belief_system.remove_belief(alert_key_to_clear)
            # Optionally, add a short-lived "resolved" event belief
            await self.belief_system.add_belief(
                alert_key_to_clear.replace("alert_active", "resolved_event"),
                {"percent": current_value, "path": path_if_disk, "timestamp": time.time()},
                0.8, BeliefSource.PERCEPTION, ttl_seconds=600 # Keep resolved event for 10 mins
            )

        self.resource_monitor.register_alert_callback(handle_resource_alert)
        self.resource_monitor.register_resolve_callback(handle_resource_resolve)
        logger.info("Coordinator: Registered internal alert/resolve callbacks for resource monitor events.")
    
    def register_agent( self, agent_id: str, agent_type: str, description: str, capabilities: List[str], metadata: Optional[Dict[str, Any]] = None, instance: Any = None ): # pragma: no cover
        self.agent_registry[agent_id] = { "agent_id": agent_id, "agent_type": agent_type, "description": description, "capabilities": capabilities, "metadata": metadata or {}, "status": "available", "registered_at": time.time(), "instance": instance }; logger.info(f"Coordinator: Registered agent {agent_id} (Type: {agent_type})")
        # Update coordinator's own metadata if it's already registered
        if "coordinator_agent_mindx" in self.agent_registry:
            self.agent_registry["coordinator_agent_mindx"]["metadata"]["managed_agents"] = list(self.agent_registry.keys())

    async def create_interaction( self, interaction_type: Union[InteractionType, str], content: str, user_id: Optional[str] = None, agent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None ) -> Interaction: # pragma: no cover
        if isinstance(interaction_type, str):
             try: interaction_type_enum = InteractionType(interaction_type.lower())
             except ValueError as e: logger.error(f"Coordinator: Invalid interaction_type string '{interaction_type}' provided for content '{content[:50]}...'."); raise ValueError(f"Invalid interaction_type: {interaction_type}") from e
        elif isinstance(interaction_type, InteractionType): interaction_type_enum = interaction_type
        else: raise TypeError(f"interaction_type must be InteractionType enum or string, not {type(interaction_type)}")
        
        interaction_id = str(uuid.uuid4()); interaction = Interaction( interaction_id, interaction_type_enum, content, user_id, agent_id, metadata )
        initiator_role = "user" if user_id else ("agent" if agent_id else "system_internal")
        interaction.add_to_history(initiator_role, f"Interaction created. Type: {interaction_type_enum.name}. Content (start): {content[:100]}...")
        self.interactions[interaction_id] = interaction
        # self._trigger_callbacks("on_interaction_created", interaction)
        logger.info(f"Coordinator: Created interaction {interaction_id} type {interaction_type_enum.value}"); return interaction

    async def process_interaction(self, interaction: Interaction) -> Interaction: # pragma: no cover
        if not isinstance(interaction, Interaction):
            logger.error("Coordinator: process_interaction called with invalid type.")
            raise TypeError("Invalid object passed to process_interaction")
        if interaction.status not in [InteractionStatus.PENDING, InteractionStatus.PENDING_APPROVAL]:
            logger.warning(f"Coordinator: Interaction {interaction.interaction_id} not in PENDING or PENDING_APPROVAL status (Is: {interaction.status.name}). Returning current state.")
            return interaction

        logger.info(f"Coordinator: Processing interaction {interaction.interaction_id} (Type: {interaction.interaction_type.name}, Current Status: {interaction.status.name})")
        interaction.status = InteractionStatus.IN_PROGRESS; interaction.started_at = time.time()
        if interaction.interaction_id not in self.active_interactions:
            self.active_interactions[interaction.interaction_id] = interaction
        # self._trigger_callbacks("on_interaction_started", interaction)

        response_data: Any = None
        try:
            if interaction.interaction_type == InteractionType.QUERY:
                response_data = await self.llm_handler.generate_text(interaction.content, max_tokens=1024, temperature=0.5)
                interaction.add_to_history("coordinator_llm", "Query processed by Coordinator's LLM.")
            elif interaction.interaction_type == InteractionType.SYSTEM_ANALYSIS:
                response_data = await self._process_system_analysis(interaction)
                if isinstance(response_data, dict) and "improvement_suggestions" in response_data:
                    source = interaction.metadata.get("source", "user_request" if interaction.user_id else ("agent_request:"+interaction.agent_id if interaction.agent_id else "unknown_source"))
                    num_added_to_backlog = 0
                    for sugg in response_data.get("improvement_suggestions",[]):
                        # Add an ID if LLM didn't provide one (though it should not)
                        sugg.setdefault("id", str(uuid.uuid4()))
                        sugg.setdefault("is_critical_target", False) # Default if LLM misses it
                        self.add_to_improvement_backlog(sugg, source=source)
                        num_added_to_backlog +=1
                    interaction.add_to_history("backlog_update", f"{num_added_to_backlog} suggestions from analysis added/updated in backlog.")
            elif interaction.interaction_type == InteractionType.COMPONENT_IMPROVEMENT:
                response_data = await self._process_component_improvement_cli(interaction)
            elif interaction.interaction_type == InteractionType.APPROVE_IMPROVEMENT:
                response_data = self._process_backlog_approval(interaction.metadata.get("backlog_item_id"), approve=True)
            elif interaction.interaction_type == InteractionType.REJECT_IMPROVEMENT:
                response_data = self._process_backlog_approval(interaction.metadata.get("backlog_item_id"), approve=False)
            else:
                response_data = {"error": f"Unsupported interaction type: {interaction.interaction_type.name}"}
                interaction.status = InteractionStatus.FAILED; interaction.error = response_data["error"]
            
            if interaction.status != InteractionStatus.FAILED:
                interaction.response = response_data
                interaction.status = InteractionStatus.COMPLETED
            interaction.add_to_history("coordinator", f"Interaction processing finished. Final Status: {interaction.status.name}.")
        except Exception as e:
            logger.error(f"Error processing interaction {interaction.interaction_id}: {e}", exc_info=True)
            interaction.status = InteractionStatus.FAILED; interaction.error = f"{type(e).__name__}: {str(e)}"
            interaction.add_to_history("system_error", f"Unhandled exception: {interaction.error}", {"traceback_snippet": traceback.format_exc(limit=5).splitlines()})
        finally:
            interaction.completed_at = time.time()
            if interaction.interaction_id in self.active_interactions: del self.active_interactions[interaction.interaction_id]
            self.completed_interactions[interaction.interaction_id] = interaction
            # self._trigger_callbacks("on_interaction_completed" or "on_interaction_failed", etc.)
        return interaction

    async def _scan_codebase_capabilities(self) -> Dict[str, Any]: # pragma: no cover
        """Scans the PROJECT_ROOT/mindx directory for Python files and extracts capability info using AST."""
        src_dir = PROJECT_ROOT / "mindx" # Standardized package name
        logger.info(f"Coordinator: Scanning capabilities in directory: {src_dir}")
        capabilities: Dict[str, Any] = {}
        if not src_dir.is_dir():
            logger.error(f"Coordinator: Source directory for capability scan not found: {src_dir}")
            return capabilities

        for file_path_obj in src_dir.rglob("*.py"):
            if file_path_obj.name.startswith("__"): continue # Skip __init__.py, etc.
            
            try:
                # Module name relative to PROJECT_ROOT (e.g., mindx.core.belief_system)
                relative_path_to_project = file_path_obj.relative_to(PROJECT_ROOT)
                module_name = ".".join(relative_path_to_project.parts[:-1] + (relative_path_to_project.stem,))
            except ValueError: # pragma: no cover # Path not under PROJECT_ROOT
                 module_name = file_path_obj.stem 
                 logger.warning(f"Coordinator: Could not determine full module path for {file_path_obj}, using stem: {module_name}")

            try:
                with file_path_obj.open("r", encoding="utf-8") as f_handle:
                    file_content = f_handle.read()
                    tree = ast.parse(file_content)

                for node in ast.walk(tree):
                    node_name_attr = getattr(node, 'name', None)
                    if not node_name_attr: continue

                    cap_key: Optional[str] = None; cap_type: Optional[str] = None
                    docstring: Optional[str] = ast.get_docstring(node)

                    if isinstance(node, ast.ClassDef) and not node_name_attr.startswith("_"):
                        cap_key = f"{module_name}.{node_name_attr}"
                        cap_type = "class"
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                         not node_name_attr.startswith("_"):
                        # Check if it's a method by seeing if its parent is a ClassDef
                        is_method = False
                        # This simple parent check is not robust for deeply nested structures.
                        # A full parent walk or context stack during AST traversal is better.
                        # For this scan, we'll assume top-level functions or methods directly under classes.
                        # More accurate would be to build a full AST map.
                        # For now, we'll just list all functions and classes.
                        # If complex logic is needed to distinguish, it can be added.
                        # This simplified version might list methods as separate functions.
                        cap_key = f"{module_name}.{node_name_attr}"
                        cap_type = "function" # Includes async functions
                    
                    if cap_key and cap_type:
                        capabilities[cap_key] = {
                            "type": cap_type, "name": node_name_attr,
                            "module": module_name, "path": str(file_path_obj), 
                            "docstring_snippet": (docstring[:150] + "...") if docstring and len(docstring) > 150 else docstring
                        }
            except SyntaxError as e_ast_syn: # pragma: no cover
                 logger.warning(f"Coordinator: AST SyntaxError scanning file {file_path_obj}: {e_ast_syn}")
            except Exception as e_ast: # pragma: no cover
                logger.warning(f"Coordinator: Error AST scanning file {file_path_obj}: {e_ast}")
        
        self.system_capabilities_cache = capabilities
        logger.info(f"Coordinator: Scanned {len(capabilities)} capabilities from codebase.")
        return capabilities

    async def _process_system_analysis(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        """Performs a system-wide analysis using LLM, incorporating monitor data and capabilities."""
        interaction.add_to_history("coordinator", "Performing system analysis with monitor data integration.")
        if self.system_capabilities_cache is None: # Ensure cache is populated
            await self._scan_codebase_capabilities()
        
        if not self.system_capabilities_cache: # pragma: no cover
            logger.error("Coordinator: System capabilities cache is empty after scan. Cannot perform analysis.")
            return {"error": "Could not scan system capabilities for analysis.", "improvement_suggestions": []}

        # --- Gather data from monitors ---
        resource_usage_summary = "Resource usage appears nominal."
        if self.resource_monitor: # pragma: no cover
            usage = self.resource_monitor.get_resource_usage()
            limits = self.resource_monitor.get_resource_limits()
            alerts = []
            cpu_thresh = limits.get("max_cpu_percent", 101.0) # Default to unachievable if not set
            if usage.get("cpu_percent", 0.0) > cpu_thresh - 10: # Near or over threshold
                alerts.append(f"CPU high ({usage['cpu_percent']:.1f}%)")
            mem_thresh = limits.get("max_memory_percent", 101.0)
            if usage.get("memory_percent", 0.0) > mem_thresh - 10:
                alerts.append(f"Memory high ({usage['memory_percent']:.1f}%)")
            
            disk_usage_map = usage.get("disk_usage_map", {})
            disk_threshold_map = limits.get("disk_threshold_map", {})
            for p_disk, u_disk in disk_usage_map.items():
                if u_disk > disk_threshold_map.get(p_disk, 101.0) - 5 : # Near or over threshold
                    alerts.append(f"Disk {Path(p_disk).name} high ({u_disk:.1f}%)")
            if alerts: resource_usage_summary = f"Potential Resource Issues: {', '.join(alerts)}."
            else: resource_usage_summary = f"Resource Usage (CPU: {usage.get('cpu_percent',0.0):.1f}%, Mem: {usage.get('memory_percent',0.0):.1f}%)."

        performance_summary = "LLM performance metrics appear stable."
        if self.performance_monitor: # pragma: no cover
            all_perf_metrics = self.performance_monitor.get_all_metrics()
            if all_perf_metrics:
                problematic_llms = []
                for key_str, metric_data in all_perf_metrics.items():
                    if metric_data.get("requests",0) > self.config.get("coordinator.system_analysis.min_perf_requests_for_alert", 10) and \
                       metric_data.get("success_rate",1.0) < self.config.get("coordinator.system_analysis.min_perf_success_rate_for_alert", 0.85):
                        problematic_llms.append(f"{key_str} (SR: {metric_data['success_rate']:.2%})")
                if problematic_llms:
                    performance_summary = f"Potential LLM Performance Issues: {', '.join(problematic_llms)} show low success rate."
                else: # pragma: no cover
                    total_reqs = sum(m.get("requests",0) for m in all_perf_metrics.values())
                    performance_summary = f"LLM performance metrics generally stable across {total_reqs} total recorded requests."

        # --- Prepare system structure summary ---
        modules = sorted(list(set(cap.get("module", "unknown_module") for cap in self.system_capabilities_cache.values())))
        example_capabilities = list(self.system_capabilities_cache.keys())[:5] # Show a few examples
        system_structure_summary = (
            f"MindX System has approximately {len(modules)} modules and {len(self.system_capabilities_cache)} scanned capabilities. "
            f"Example capabilities include: {', '.join(example_capabilities)}."
        )
        
        history_summary = "No recent major improvement campaigns logged."
        if self.improvement_campaign_history: # pragma: no cover
            recent_campaigns = self.improvement_campaign_history[-3:] # Last 3 campaigns
            history_summary = "Recent Improvement Campaign Summaries:\n" + "\n".join([
                f"- Target: {att.get('target_component_id','N/A')}, SIA Status: {att.get('status_from_sia_json','N/A')}, Description: {att.get('summary_message','N/A')[:80]}..."
                for att in recent_campaigns
            ])

        analysis_focus_hint = interaction.metadata.get("analysis_context", "overall system health, identifying areas for refactoring, optimization, or new critical features")

        prompt = (
            f"You are an AI System Architect for the MindX system (developed by Augmentic).\n"
            f"Your task is to analyze the current state of MindX and provide actionable improvement suggestions.\n"
            f"Focus of this analysis: {analysis_focus_hint}\n\n"
            f"CONTEXTUAL DATA:\n"
            f"1. System Structure Summary: {system_structure_summary}\n"
            f"2. Current Resource Status: {resource_usage_summary}\n"
            f"3. Current LLM Performance Summary: {performance_summary}\n"
            f"4. Recent Improvement History: {history_summary}\n\n"
            f"INSTRUCTIONS:\n"
            f"Based on all the provided data, identify 1 to 3 high-impact improvement areas or specific components. "
            f"For each suggestion, you MUST provide:\n"
            f"  a. 'target_component_path': The full Python module path of the component to improve (e.g., 'mindx.learning.self_improve_agent' for self_improve_agent.py). If suggesting a new component, use a plausible new module path.\n"
            f"  b. 'suggestion': A concise description of WHAT specific improvement should be made and WHY it's beneficial (e.g., 'Refactor method X for clarity due to high complexity readings from logs', 'Optimize component Y as it's linked to high memory usage alerts', 'Add feature Z to module A to address missing core functionality').\n"
            f"  c. 'priority': An integer from 1 (lowest) to 10 (highest) indicating urgency and impact.\n"
            f"  d. 'is_critical_target': A boolean (true/false) indicating if the target component is core to system operation (e.g., the SelfImprovementAgent, CoordinatorAgent, or core Config/Utility modules).\n\n"
            f"Respond ONLY with a single, valid JSON object containing a single key 'improvement_suggestions', which is a list of these suggestion objects. Ensure your JSON is well-formed.\n"
            f"Example of a single suggestion object: {{\"target_component_path\": \"mindx.module.file_stem\", \"suggestion\": \"Detailed suggestion text.\", \"priority\": 7, \"is_critical_target\": false}}"
        )
        
        interaction.add_to_history("llm_prompt_summary", f"System analysis prompt generated. Focus: {analysis_focus_hint}. Structure/Resource/Performance/History data included.")
        try:
            response_str = await self.llm_handler.generate_text(
                prompt,
                max_tokens=self.config.get("coordinator.system_analysis.max_tokens", 2048), # Increased tokens
                temperature=self.config.get("coordinator.system_analysis.temperature", 0.1),
                json_mode=True # Strongly request JSON output
            )
            
            analysis_result = {}
            if response_str and not response_str.startswith("Error:"):
                try: analysis_result = json.loads(response_str) # Direct parse
                except json.JSONDecodeError: # pragma: no cover # Try to extract from markdown
                    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", response_str, re.DOTALL)
                    if match:
                        try: analysis_result = json.loads(match.group(1))
                        except json.JSONDecodeError as e_json_extract: logger.warning(f"Coordinator: Failed to parse extracted JSON from LLM analysis response: {e_json_extract}. Raw: {match.group(1)[:200]}"); raise ValueError("LLM response contained malformed JSON within markdown.") from e_json_extract
                    else: # Final attempt: find first '{' and last '}'
                        json_start = response_str.find('{'); json_end = response_str.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            try: analysis_result = json.loads(response_str[json_start:json_end])
                            except json.JSONDecodeError as e_json_slice: logger.warning(f"Coordinator: Failed to parse sliced JSON from LLM analysis response: {e_json_slice}. Raw slice: {response_str[json_start:json_end][:200]}"); raise ValueError("LLM response was not valid JSON after slicing.") from e_json_slice
                        else: # pragma: no cover
                            raise ValueError(f"No valid JSON object found in LLM response for system analysis. Raw response snippet: {response_str[:300]}")
            else: # pragma: no cover
                raise ValueError(f"LLM analysis returned an error or empty response: {response_str}")

            if "improvement_suggestions" not in analysis_result or not isinstance(analysis_result["improvement_suggestions"], list): # pragma: no cover
                logger.warning(f"Coordinator: LLM analysis result missing 'improvement_suggestions' list or it's not a list: {analysis_result}")
                analysis_result["improvement_suggestions"] = [] # Ensure it exists as a list
            
            # Validate structure of suggestions
            valid_suggestions = []
            for sugg_idx, sugg_item in enumerate(analysis_result.get("improvement_suggestions", [])):
                if isinstance(sugg_item, dict) and \
                   all(k in sugg_item for k in ["target_component_path", "suggestion", "priority", "is_critical_target"]) and \
                   isinstance(sugg_item["target_component_path"], str) and \
                   isinstance(sugg_item["suggestion"], str) and \
                   isinstance(sugg_item["priority"], int) and \
                   isinstance(sugg_item["is_critical_target"], bool):
                    valid_suggestions.append(sugg_item)
                else: # pragma: no cover
                    logger.warning(f"Coordinator: Discarding malformed suggestion #{sugg_idx+1} from LLM analysis: {str(sugg_item)[:200]}")
            analysis_result["improvement_suggestions"] = valid_suggestions


            interaction.add_to_history("llm_analysis_parsed", "System analysis parsed successfully.", {"num_suggestions": len(analysis_result["improvement_suggestions"]), "raw_response_snippet": response_str[:200]})
            return analysis_result
        except Exception as e: # pragma: no cover
            logger.error(f"Coordinator: System analysis LLM processing error: {e}", exc_info=True)
            interaction.add_to_history("system_error", f"LLM call or parsing failed during system analysis: {e}")
            return {"error": f"LLM call or parsing failed during system analysis: {type(e).__name__}: {e}", "improvement_suggestions": []}

    async def _resolve_component_path_for_sia(self, component_identifier: str) -> Optional[Path]: # pragma: no cover
        """Resolves a component identifier to an absolute file path for the SIA."""
        if not component_identifier: return None
        logger.debug(f"Coordinator: Resolving component ID '{component_identifier}' for SIA.")

        # Strategy 1: Check if it's a registered agent_id with a "script_path" or "file_path" in metadata
        # This is useful for targeting agents themselves, like "self_improve_agent_cli_mindx"
        agent_info = self.get_agent(component_identifier)
        if agent_info:
            path_meta_key = agent_info["metadata"].get("script_path", agent_info["metadata"].get("file_path"))
            if path_meta_key:
                potential_path = Path(path_meta_key)
                if potential_path.is_absolute() and potential_path.exists() and potential_path.is_file():
                    logger.debug(f"Resolved '{component_identifier}' via agent registry metadata to absolute path: {potential_path}")
                    return potential_path
                # If path in metadata is relative, assume relative to PROJECT_ROOT
                resolved_from_meta_relative = (PROJECT_ROOT / potential_path).resolve()
                if resolved_from_meta_relative.exists() and resolved_from_meta_relative.is_file():
                    logger.debug(f"Resolved '{component_identifier}' via agent registry metadata (relative to project root) to: {resolved_from_meta_relative}")
                    return resolved_from_meta_relative
                logger.warning(f"Agent '{component_identifier}' metadata path '{path_meta_key}' not found.")

        # Strategy 2: Assume it's a Python module path like "mindx.core.belief_system"
        # This should resolve to PROJECT_ROOT/mindx/core/belief_system.py
        if "." in component_identifier:
            parts = component_identifier.split('.')
            # If it starts with "mindx", assume it's a path within the mindx package under PROJECT_ROOT
            if parts[0] == "mindx":
                # e.g., mindx.core.belief_system -> PROJECT_ROOT/mindx/core/belief_system.py
                resolved_path = (PROJECT_ROOT / Path(*parts)).with_suffix('.py')
            else:
                # If it doesn't start with "mindx", assume it's a module path relative to PROJECT_ROOT/mindx
                # e.g., "utils.config" -> PROJECT_ROOT/mindx/utils/config.py
                resolved_path = (PROJECT_ROOT / "mindx" / Path(*parts)).with_suffix('.py')
            
            if resolved_path.exists() and resolved_path.is_file():
                logger.debug(f"Resolved module path '{component_identifier}' to: {resolved_path}")
                return resolved_path.resolve() # Ensure absolute

        # Strategy 3: Try as a direct file path relative to PROJECT_ROOT
        # e.g., "scripts/run_mindx_coordinator.py"
        path_from_project_root = (PROJECT_ROOT / component_identifier).resolve()
        if path_from_project_root.exists() and path_from_project_root.is_file():
            logger.debug(f"Resolved '{component_identifier}' as file relative to project root: '{path_from_project_root}'")
            return path_from_project_root
            
        logger.warning(f"Coordinator: Could not resolve component identifier '{component_identifier}' to a valid file path using PROJECT_ROOT: {PROJECT_ROOT}")
        return None

    async def _process_component_improvement_cli(self, interaction: Interaction) -> Dict[str, Any]: # pragma: no cover
        """Delegates a specific component improvement task to SelfImprovementAgent CLI."""
        if not self.self_improve_agent_script_path: # pragma: no cover
            return {"status": "FAILURE", "message": "SelfImprovementAgent script path not configured in Coordinator."}

        metadata = interaction.metadata
        target_component_id = metadata.get("target_component")
        improvement_context = metadata.get("analysis_context", "General improvement request based on prior analysis or direct request.")
        max_cycles = metadata.get("max_cycles", self.config.get("self_improvement_agent.default_max_cycles", 1))
        
        # Allow overriding SIA's LLM config and evaluation params from interaction metadata
        sia_llm_provider = metadata.get("sia_llm_provider") # Uses SIA's config if None
        sia_llm_model = metadata.get("sia_llm_model")
        sia_critique_threshold = metadata.get("sia_critique_threshold")
        sia_self_test_timeout = metadata.get("sia_self_test_timeout")

        if not target_component_id: # pragma: no cover
            return {"status": "FAILURE", "message": "COMPONENT_IMPROVEMENT interaction requires 'target_component' in metadata."}

        target_file_to_improve_abs = await self._resolve_component_path_for_sia(target_component_id)
        if not target_file_to_improve_abs: # pragma: no cover
            return {"status": "FAILURE", "message": f"Could not resolve target_component '{target_component_id}' to a valid file path."}

        interaction.add_to_history("coordinator", f"Preparing SelfImprovementAgent CLI call for target: {target_file_to_improve_abs.name}")

        command = [
            str(sys.executable), # Use current Python interpreter
            str(self.self_improve_agent_script_path),
            str(target_file_to_improve_abs), # SIA CLI expects absolute path to target
            "--cycles", str(max_cycles),
            "--output-json" # CRITICAL for Coordinator to parse SIA's output
        ]
        
        context_temp_file_path: Optional[Path] = None
        # Handle potentially large context string by writing to a temp file
        if improvement_context:
            # Arbitrary threshold for using a file vs direct arg
            if len(improvement_context) > self.config.get("coordinator.sia_context_file_threshold_chars", 1024): 
                try:
                    temp_dir = PROJECT_ROOT / "data" / "temp_sia_contexts"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    # Use a unique name for the temp file to avoid collisions
                    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", prefix="context_", dir=temp_dir, encoding="utf-8") as tmp_f:
                        tmp_f.write(improvement_context)
                        context_temp_file_path = Path(tmp_f.name)
                    command.extend(["--context-file", str(context_temp_file_path)])
                    logger.info(f"Coordinator: Large context for SIA written to temp file: {context_temp_file_path}")
                except Exception as e_ctx: # pragma: no cover
                    logger.error(f"Coordinator: Failed to write context to temp file: {e_ctx}. Passing context directly (might be truncated by OS).")
                    command.extend(["--context", improvement_context[:4096]]) # Fallback, truncate if very long
            else: # Small context, pass directly
                 command.extend(["--context", improvement_context])

        # Add optional overrides for SIA's internal config
        if sia_llm_provider: command.extend(["--llm-provider", sia_llm_provider])
        if sia_llm_model: command.extend(["--llm-model", sia_llm_model])
        if sia_critique_threshold is not None: command.extend(["--critique-threshold", str(sia_critique_threshold)])
        if sia_self_test_timeout is not None: command.extend(["--self-test-timeout", str(sia_self_test_timeout)])
        
        logger.info(f"Coordinator: Executing SIA CLI: {' '.join(command)}")
        # SIA script should ideally be runnable from PROJECT_ROOT if it uses relative paths for its own data
        process_cwd = PROJECT_ROOT
        
        sia_cli_result_json: Dict[str, Any] = {"status": "FAILURE", "message": "SIA CLI call did not complete as expected"} # Default
        timeout_seconds = self.config.get("coordinator.sia_cli_timeout_seconds")
        try:
            async with self.sia_concurrency_limit: # Limit concurrent SIA calls
                process = await asyncio.create_subprocess_exec(
                    *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=process_cwd )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
            
            stdout_str = stdout.decode(errors='ignore').strip()
            stderr_str = stderr.decode(errors='ignore').strip()

            if stderr_str: # Log any stderr from SIA, even on success
                interaction.add_to_history("sia_stderr", f"SIA STDERR Output (first 1000 chars): {stderr_str[:1000]}")

            # SIA CLI *must* output JSON on stdout with "status", "message", "data"
            if stdout_str:
                try:
                    sia_cli_result_json = json.loads(stdout_str)
                    if not isinstance(sia_cli_result_json, dict) or "status" not in sia_cli_result_json: # pragma: no cover
                        raise json.JSONDecodeError("SIA JSON missing 'status' key.", stdout_str, 0)
                except json.JSONDecodeError as e_json: # pragma: no cover
                    err_msg = f"SIA CLI STDOUT not valid JSON as expected. Error: {e_json}"
                    logger.error(f"{err_msg} Raw STDOUT: {stdout_str[:500]}")
                    interaction.add_to_history("system_error", err_msg, {"raw_stdout": stdout_str})
                    sia_cli_result_json = {"status": "FAILURE", "message": err_msg, "raw_stdout": stdout_str}
            elif process.returncode != 0 : # Process failed and no stdout
                 sia_cli_result_json = {"status": "FAILURE", "message": f"SIA process failed (Code: {process.returncode}) with no STDOUT.", "stderr": stderr_str}
            else: # Process success (rc=0) but no stdout - SIA CLI should always output JSON
                 sia_cli_result_json = {"status": "FAILURE", "message": f"SIA process exited 0 but STDOUT was empty."}


            # Process SIA's structured response
            if sia_cli_result_json.get("status") == "SUCCESS":
                interaction.add_to_history("sia_cli_success", sia_cli_result_json.get("message", "SIA operation successful."), {"sia_data": sia_cli_result_json.get("data", {})})
                sia_data = sia_cli_result_json.get("data", {}) # This is the detailed result from SIA's methods
                if sia_data.get("code_updated_requires_restart", False): # SIA signals critical update
                    affected_module_path_str = sia_data.get("target_script_path", str(target_file_to_improve_abs))
                    affected_module_name = Path(affected_module_path_str).name
                    logger.warning(f"Coordinator: SIA reported CRITICAL UPDATE to '{affected_module_name}'. System restart likely required for changes to take full effect.")
                    interaction.add_to_history("coordinator_alert", f"Critical module '{affected_module_name}' was updated by SIA. System restart may be advised.")
                    await self.belief_system.add_belief(
                        f"system.restart_advised.reason", 
                        f"Critical update to module: {affected_module_name}", 
                        0.95, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600*24 # Belief persists for a day
                    )
            else: # SIA reported failure in its JSON, or process itself failed and we constructed failure JSON
                logger.error(f"Coordinator: SIA operation reported failure. SIA JSON Response: {json.dumps(sia_cli_result_json, indent=2)}")
                interaction.add_to_history("sia_cli_failure", sia_cli_result_json.get("message", "SIA operation failed."), {"sia_response": sia_cli_result_json})

        except asyncio.TimeoutError: # pragma: no cover
            err_msg = f"SIA CLI call timed out after {timeout_seconds}s for target '{target_file_to_improve_abs.name}'."
            logger.error(err_msg)
            interaction.add_to_history("system_error", err_msg); sia_cli_result_json = {"status": "FAILURE", "message": err_msg, "reason": "TIMEOUT"}
        except Exception as e_proc: # pragma: no cover
            err_msg = f"Exception during SIA CLI call orchestration for '{target_file_to_improve_abs.name}': {type(e_proc).__name__}: {e_proc}"
            logger.error(err_msg, exc_info=True)
            interaction.add_to_history("system_error", err_msg); sia_cli_result_json = {"status": "FAILURE", "message": err_msg, "reason": "EXCEPTION"}
        finally:
            if context_temp_file_path and context_temp_file_path.exists():
                try: context_temp_file_path.unlink(); logger.debug(f"Coordinator: Deleted temp context file: {context_temp_file_path}")
                except Exception as e_del_ctx: logger.warning(f"Coordinator: Failed to delete temp context file {context_temp_file_path}: {e_del_ctx}")
        
        # Record in coordinator's high-level campaign history
        campaign_status_from_sia = sia_cli_result_json.get("status", "UNKNOWN_SIA_JSON_STATUS")
        # Use final_status from SIA's internal operation if available in 'data'
        final_sia_operation_status = sia_cli_result_json.get("data",{}).get("final_status", "NOT_APPLICABLE" if campaign_status_from_sia == "SUCCESS" else "UNKNOWN_SIA_OP_STATUS")

        campaign_entry = {
            "timestamp": time.time(), "interaction_id": interaction.interaction_id,
            "target_component_id": target_component_id, "resolved_path": str(target_file_to_improve_abs),
            "status_from_sia_cli": campaign_status_from_sia, # SUCCESS/FAILURE of the CLI call itself
            "final_sia_op_status": final_sia_operation_status, # SUCCESS_PROMOTED etc. from SIA's internal logic
            "summary_message": sia_cli_result_json.get("message", "N/A")
        }
        self.improvement_campaign_history.append(campaign_entry)
        self._save_campaign_history()
        # self._trigger_callbacks("on_improvement_campaign_result", campaign_entry)
        return sia_cli_result_json # Return the full JSON response from SIA CLI

    async def _autonomous_improvement_worker(self, interval_seconds: float): # pragma: no cover
        """Periodically analyzes system, manages backlog, and triggers improvements with HITL."""
        logger.info(f"Autonomous improvement worker started. Check interval: {interval_seconds}s.")
        # Load cool-down store from beliefs or a file if persistence is desired
        cool_down_store: Dict[str, float] = {} # target_component_path -> last_fail_timestamp for this session

        while True:
            try:
                await asyncio.sleep(interval_seconds) # Wait for the interval
                logger.info("Autonomous worker: Starting periodic self-improvement cycle.")

                # 1. Perform System Analysis (and add to backlog)
                logger.info("Autonomous worker: Performing system analysis...")
                analysis_interaction_meta = {"source": "autonomous_worker_analysis"}
                analysis_interaction = await self.create_interaction(
                    InteractionType.SYSTEM_ANALYSIS,
                    "Scheduled autonomous system analysis for MindX. Identify key improvement areas.",
                    agent_id="coordinator_agent_mindx", # This agent is initiating
                    metadata=analysis_interaction_meta
                )
                processed_analysis = await self.process_interaction(analysis_interaction)
                # add_to_improvement_backlog is called within _process_system_analysis if successful

                # 2. Check Resource Usage before attempting any SIA task
                current_cpu_usage = self.resource_monitor.get_resource_usage().get("cpu_percent", 0.0)
                max_cpu_for_sia = self.config.get("coordinator.autonomous_improvement.max_cpu_before_sia", 90.0)
                if current_cpu_usage > max_cpu_for_sia:
                    logger.warning(f"Autonomous worker: System CPU usage ({current_cpu_usage:.1f}%) is above threshold ({max_cpu_for_sia}%). Deferring SIA improvement tasks for this cycle.")
                    continue # Skip to next interval

                # 3. Process one item from the improvement backlog
                if not self.improvement_backlog:
                    logger.info("Autonomous worker: Improvement backlog is currently empty."); continue
                
                next_improvement_to_try: Optional[Dict[str, Any]] = None
                # Iterate to find a suitable item from the backlog (already sorted by add_to_improvement_backlog)
                for item_idx, backlog_item in enumerate(self.improvement_backlog):
                    if backlog_item.get("status") == InteractionStatus.PENDING.value: # Only process "pending"
                        target_path = backlog_item["target_component_path"]
                        last_fail_time = cool_down_store.get(target_path)
                        cool_down_duration = self.config.get("coordinator.autonomous_improvement.cooldown_seconds", 3 * 3600.0)
                        if last_fail_time and (time.time() - last_fail_time < cool_down_duration):
                            logger.info(f"Autonomous worker: Skipping '{target_path}' (ID {backlog_item.get('id','N/A')[:8]}), it's in cool-down period after a recent failure.")
                            continue
                        next_improvement_to_try = backlog_item
                        break 
                
                if not next_improvement_to_try:
                    logger.info("Autonomous worker: No actionable (pending and not in cool-down) items in backlog currently."); continue

                # Human-in-the-Loop (HITL) Check for critical components
                target_module_path = next_improvement_to_try["target_component_path"]
                # Check if any critical stem is a substring of the target module path
                is_critical_target = next_improvement_to_try.get("is_critical_target", False) or \
                                     any(crit_comp_stem in target_module_path for crit_comp_stem in self.critical_components_for_approval)

                if self.require_human_approval_for_critical and is_critical_target and \
                   next_improvement_to_try.get("approved_at") is None: # Not yet approved
                    if next_improvement_to_try["status"] != InteractionStatus.PENDING_APPROVAL.value: # Only mark if not already
                        next_improvement_to_try["status"] = InteractionStatus.PENDING_APPROVAL.value
                        self._save_backlog() # Persist status change
                        logger.warning(
                            f"Autonomous worker: CRITICAL improvement suggested for '{target_module_path}' "
                            f"(ID {next_improvement_to_try.get('id','N/A')[:8]}) requires human approval. "
                            f"Marked as '{InteractionStatus.PENDING_APPROVAL.value}'. "
                            f"Suggestion: {next_improvement_to_try['suggestion']}"
                        )
                        # In a real system, this might trigger a notification (email, Slack, etc.)
                    else: # Already pending approval
                        logger.info(f"Autonomous worker: Critical item '{target_module_path}' (ID {next_improvement_to_try.get('id','N/A')[:8]}) still pending human approval.")
                    continue # Skip processing this item by autonomous loop until approved

                # Proceed with improvement attempt
                logger.info(
                    f"Autonomous worker: Attempting improvement for: '{next_improvement_to_try['target_component_path']}' "
                    f"(ID {next_improvement_to_try.get('id','N/A')[:8]}, Priority: {next_improvement_to_try['priority']})"
                )
                next_improvement_to_try["status"] = InteractionStatus.IN_PROGRESS.value # Mark as "in_progress_sia" conceptually
                next_improvement_to_try["attempt_count"] = next_improvement_to_try.get("attempt_count", 0) + 1
                next_improvement_to_try["last_attempted_at"] = time.time()
                self._save_backlog() # Persist status update

                improvement_metadata = {
                    "target_component": next_improvement_to_try["target_component_path"],
                    "analysis_context": next_improvement_to_try["suggestion"], # The suggestion itself is the context/goal
                    "source": "autonomous_worker_backlog_processing",
                    "backlog_item_id": next_improvement_to_try.get("id"),
                    "original_priority": next_improvement_to_try["priority"]
                    # Pass SIA specific overrides if they were part of the suggestion or derived
                }
                improvement_content = f"Autonomous MindX System: Attempting improvement on '{next_improvement_to_try['target_component_path']}'. Goal: {next_improvement_to_try['suggestion'][:150]}..."
                
                improvement_interaction = await self.create_interaction(
                    InteractionType.COMPONENT_IMPROVEMENT,
                    improvement_content,
                    agent_id="coordinator_agent_mindx", # Coordinator is initiating this based on autonomous logic
                    metadata=improvement_metadata
                )
                processed_improvement_interaction = await self.process_interaction(improvement_interaction)
                
                # Update backlog item based on the SIA's CLI JSON response (which is in processed_improvement_interaction.response)
                sia_response_json = processed_improvement_interaction.response
                if isinstance(sia_response_json, dict):
                    if sia_response_json.get("status") == "SUCCESS":
                        # SIA CLI call itself was successful, now check SIA's internal operation status
                        sia_internal_data = sia_response_json.get("data", {})
                        final_sia_op_status = sia_internal_data.get("final_status", "UNKNOWN_SIA_OP_STATUS")
                        if final_sia_op_status.startswith("SUCCESS"): # e.g. SUCCESS_PROMOTED, SUCCESS_EVALUATED
                            next_improvement_to_try["status"] = InteractionStatus.COMPLETED.value # Mark as completed
                        else: # SIA ran but didn't achieve full success (e.g., FAILED_EVALUATION)
                            next_improvement_to_try["status"] = InteractionStatus.FAILED.value # Mark as failed for backlog
                            cool_down_store[next_improvement_to_try["target_component_path"]] = time.time() # Add to cool-down
                        next_improvement_to_try["sia_final_op_status"] = final_sia_op_status
                        next_improvement_to_try["sia_message"] = sia_response_json.get("message")
                    else: # SIA CLI call itself reported failure in its top-level JSON
                        next_improvement_to_try["status"] = InteractionStatus.FAILED.value
                        next_improvement_to_try["sia_error"] = sia_response_json.get("message", "SIA CLI reported failure.")
                        cool_down_store[next_improvement_to_try["target_component_path"]] = time.time()
                else: # Should not happen if SIA CLI guarantees JSON
                    next_improvement_to_try["status"] = InteractionStatus.FAILED.value
                    next_improvement_to_try["sia_error"] = "Invalid or no JSON response from SIA CLI."
                    cool_down_store[next_improvement_to_try["target_component_path"]] = time.time()
                
                self._save_backlog() # Persist updated backlog item
                logger.info(f"Autonomous worker: Improvement attempt for '{next_improvement_to_try['target_component_path']}' (ID {next_improvement_to_try.get('id','N/A')[:8]}) finished. Backlog status: {next_improvement_to_try['status']}")

            except asyncio.CancelledError: # pragma: no cover
                logger.info("Autonomous improvement worker stopping due to cancellation.")
                break
            except Exception as e: # pragma: no cover
                logger.error(f"Autonomous improvement worker encountered an unhandled error: {e}", exc_info=True)
                # Sleep longer on major error to avoid rapid failure loops and allow investigation
                await asyncio.sleep(max(interval_seconds * 2, 3600)) # Min 1hr sleep on such error
        logger.info("Autonomous improvement worker has stopped.") # pragma: no cover


    # --- Handle User/Agent Input ---
    async def handle_user_input( self, content: str, user_id: Optional[str] = None, interaction_type: Optional[Union[InteractionType, str]] = None, metadata: Optional[Dict[str, Any]] = None ) -> Dict[str, Any]: # pragma: no cover
        """Handles user input, creating and processing an interaction, returns result as dict."""
        logger.info(f"Coordinator: User input from '{user_id}': '{content[:100]}...' Metadata: {metadata}")
        metadata = metadata or {}
        parsed_content_for_interaction = content # Default
        
        # Infer interaction_type and parse relevant metadata if not explicitly provided by caller
        if not interaction_type:
            parts = content.strip().split(" ", 1)
            cmd_verb = parts[0].lower()
            cmd_args_str = parts[1] if len(parts) > 1 else ""

            if cmd_verb == "analyze_system":
                interaction_type = InteractionType.SYSTEM_ANALYSIS
                parsed_content_for_interaction = "System analysis request" # Standardized content for this type
                if cmd_args_str: metadata["analysis_context"] = cmd_args_str # User can provide focus
            elif cmd_verb == "improve":
                interaction_type = InteractionType.COMPONENT_IMPROVEMENT
                improve_args_parts = cmd_args_str.split(" ", 1)
                if not improve_args_parts or not improve_args_parts[0]:
                    return {"error": "'improve' command requires a <target_component_id>.", "status": InteractionStatus.FAILED.value}
                metadata["target_component"] = improve_args_parts[0]
                if len(improve_args_parts) > 1:
                    metadata["analysis_context"] = improve_args_parts[1] # This becomes goal/context for SIA
                parsed_content_for_interaction = f"Request to improve component: {metadata['target_component']}"
            elif cmd_verb == "approve":
                interaction_type = InteractionType.APPROVE_IMPROVEMENT
                metadata["backlog_item_id"] = cmd_args_str.strip()
                parsed_content_for_interaction = f"Request to approve backlog item: {metadata['backlog_item_id']}"
            elif cmd_verb == "reject":
                interaction_type = InteractionType.REJECT_IMPROVEMENT
                metadata["backlog_item_id"] = cmd_args_str.strip()
                parsed_content_for_interaction = f"Request to reject backlog item: {metadata['backlog_item_id']}"
            elif cmd_verb == "query": # Explicit query command
                interaction_type = InteractionType.QUERY
                parsed_content_for_interaction = cmd_args_str
            else: # Default to query if command is not recognized
                interaction_type = InteractionType.QUERY
                parsed_content_for_interaction = content # Use full original content for query
        
        # Ensure interaction_type is Enum by this point
        if isinstance(interaction_type, str):
            try: interaction_type_enum = InteractionType(interaction_type.lower())
            except ValueError: 
                logger.error(f"Coordinator: Invalid interaction type string provided: '{interaction_type}'")
                return {"error": f"Invalid interaction type: {interaction_type}", "status": InteractionStatus.FAILED.value}
        elif isinstance(interaction_type, InteractionType):
            interaction_type_enum = interaction_type
        else: # Should not be reached if logic above is correct
            logger.error(f"Coordinator: interaction_type is of unexpected type: {type(interaction_type)}")
            return {"error": "Internal error: Unexpected interaction_type format.", "status": InteractionStatus.FAILED.value}

        try:
            interaction = await self.create_interaction(
                interaction_type=interaction_type_enum,
                content=parsed_content_for_interaction,
                user_id=user_id,
                metadata=metadata
            )
            processed_interaction = await self.process_interaction(interaction)
            return processed_interaction.to_dict() # Return dict representation for external callers
        except Exception as e:
            logger.error(f"Coordinator: Unhandled error in handle_user_input after type processing: {e}", exc_info=True)
            return {"interaction_id": None, "status": InteractionStatus.FAILED.value, 
                    "response": None, "error": f"Unhandled error: {type(e).__name__}: {str(e)}"}

    # --- Shutdown and Singleton Reset ---
    async def shutdown(self): # pragma: no cover
        """Gracefully shuts down the CoordinatorAgent and its managed tasks."""
        logger.info("CoordinatorAgent MindX (v_prod_candidate_final) shutting down...")
        self.stop_autonomous_improvement_loop() # Signal autonomous loop to stop
        if self.autonomous_improvement_task:
            try:
                logger.debug("Waiting for autonomous improvement task to finish cancellation...")
                await asyncio.wait_for(self.autonomous_improvement_task, timeout=5.0)
            except asyncio.CancelledError: logger.info("Autonomous task was successfully cancelled.")
            except asyncio.TimeoutError: logger.warning("Autonomous task did not shut down cleanly in time during coordinator shutdown.")
            except Exception as e: logger.error(f"Error awaiting autonomous task during shutdown: {e}", exc_info=True)
        
        if self.resource_monitor and self.resource_monitor.monitoring:
            self.resource_monitor.stop_monitoring() # This cancels its task
            # Could await resource_monitor.monitoring_task if made accessible and important
        
        if self.performance_monitor:
            await self.performance_monitor.shutdown() # Performance monitor has async shutdown
        
        self._save_backlog() # Final save of backlog
        self._save_campaign_history() # Final save of campaign history
        logger.info("CoordinatorAgent MindX (v_prod_candidate_final) shutdown complete.")

    @classmethod
    async def reset_instance_async(cls): # For testing # pragma: no cover
        """Asynchronously resets the singleton instance. Ensures tasks are stopped."""
        async with cls._lock: # Ensure only one reset happens at a time
            if cls._instance:
                await cls._instance.shutdown() # Call proper shutdown
                cls._instance._initialized = False # Allow re-init for next test
                cls._instance = None
        logger.debug("CoordinatorAgent instance reset asynchronously.")

# --- Factory Functions ---
async def get_coordinator_agent_mindx_async(config_override: Optional[Config] = None, test_mode: bool = False) -> CoordinatorAgent: # pragma: no cover
    """Asynchronously gets or creates the CoordinatorAgent singleton, ensuring dependencies are also async-created."""
    # Use class lock for thread/task-safe singleton creation
    # This ensures that even if multiple coroutines call this simultaneously at startup,
    # only one instance is created.
    async with CoordinatorAgent._lock:
        if CoordinatorAgent._instance is None or test_mode:
            if test_mode and CoordinatorAgent._instance is not None: # If test_mode and instance exists, shut down old one
                logger.info("Test mode: Shutting down existing Coordinator for reset.")
                await CoordinatorAgent._instance.shutdown() 
                CoordinatorAgent._instance = None # Nullify to force re-creation

            effective_config = config_override or Config(test_mode=test_mode) # Ensure Config respects test_mode
            
            # Get dependencies, also respecting test_mode for their singletons if applicable
            belief_system = BeliefSystem(test_mode=test_mode) # Assuming BeliefSystem has test_mode handling
            resource_monitor_instance = await get_resource_monitor_async(config_override=effective_config, test_mode=test_mode)
            performance_monitor_instance = await get_performance_monitor_async(config_override=effective_config, test_mode=test_mode)
            
            # Create the new CoordinatorAgent instance
            CoordinatorAgent._instance = CoordinatorAgent(
                belief_system=belief_system,
                resource_monitor=resource_monitor_instance,
                performance_monitor=performance_monitor_instance,
                config_override=effective_config, # Pass the specific config instance
                test_mode=test_mode
            )
        return CoordinatorAgent._instance

def get_coordinator_agent_mindx(config_override: Optional[Config] = None, test_mode:bool = False) -> CoordinatorAgent: # pragma: no cover
    """
    Synchronously gets or creates the CoordinatorAgent singleton.
    Best used at the application's main entry point or in purely synchronous test setups.
    Be cautious if your application is heavily async.
    """
    if CoordinatorAgent._instance and not test_mode:
        return CoordinatorAgent._instance
    
    # This sync getter is complex for an async-heavy object.
    # The primary way to get the agent should be the async factory.
    try:
        loop = asyncio.get_running_loop() # Check if a loop is already running
        if loop.is_running(): # pragma: no cover 
            # If called from within an already running async context, this is tricky.
            # `asyncio.run` cannot be called if a loop is already running.
            # `nest_asyncio` can patch this for some environments (like Jupyter) but is a hack.
            # The safest thing if a loop is running is to tell the user to use the async factory.
            logger.warning(
                "get_coordinator_agent_mindx (sync) called while an event loop is running. "
                "This can lead to issues. Prefer using get_coordinator_agent_mindx_async() from an async context."
                "Attempting to use nest_asyncio as a fallback for this call."
            )
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(get_coordinator_agent_mindx_async(config_override, test_mode))
            
    except RuntimeError: # No event loop is running
        pass # Proceed to create a new loop for this sync call

    # Create a new loop specifically for this synchronous call to get the instance
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    try:
        instance = new_loop.run_until_complete(get_coordinator_agent_mindx_async(config_override, test_mode))
        return instance
    finally:
        # It's important not to close the loop if it might be used by other parts of a sync application
        # that got this loop set as current. For a one-off get, closing might be okay.
        # For robust library use, managing loops explicitly is better.
        # asyncio.set_event_loop(None) # Detach this loop as the current one.
        pass # Let the caller manage loop closure if it created it.
