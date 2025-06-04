# mindx/orchestration/multimodel_agent.py
import os
import logging
import asyncio
import json
import time
import yaml # For loading capability files
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Coroutine
from enum import Enum

from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from mindx.llm.model_registry import get_model_registry, ModelRegistry # Singleton accessor
from mindx.llm.llm_interface import LLMInterface # Abstract base class for handlers
from mindx.core.belief_system import BeliefSystem, BeliefSource
# Import ModelSelector which MMA will use
from .model_selector import ModelSelector # Relative import

logger = get_logger(__name__)

class TaskType(Enum): # pragma: no cover
    """Defines the types of tasks the MultiModelAgent can handle."""
    GENERATION = "generation"; REASONING = "reasoning"; SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"; CODE_REVIEW = "code_review"; PLANNING = "planning"
    RESEARCH = "research"; SELF_IMPROVEMENT_ANALYSIS = "self_improvement_analysis"
    DATA_EXTRACTION = "data_extraction"; TRANSLATION = "translation" # Added more examples

class TaskPriority(Enum): LOW = 1; MEDIUM = 2; HIGH = 3; CRITICAL = 4 # pragma: no cover
class TaskStatus(Enum): PENDING = "pending"; ASSIGNED = "assigned"; IN_PROGRESS = "in_progress"; COMPLETED = "completed"; FAILED = "failed"; CANCELLED = "cancelled" # pragma: no cover


class Task: # pragma: no cover
    """Represents a task to be processed by a model within the MultiModelAgent."""
    def __init__(
        self, task_id: str, task_type: TaskType, prompt: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        requirements: Optional[Dict[str, Any]] = None,
        max_attempts_override: Optional[int] = None, # Allow task-specific override
        callback: Optional[Callable[['Task'], Coroutine[Any,Any,None]]] = None # Async callback for completion
    ):
        self.task_id = task_id; self.task_type = task_type; self.prompt = prompt
        self.priority = priority; self.context = context or {}; self.requirements = requirements or {}
        self.status = TaskStatus.PENDING; self.assigned_model: Optional[str] = None
        self.result: Any = None; self.error: Optional[str] = None
        self.created_at = time.time(); self.started_at: Optional[float] = None; self.completed_at: Optional[float] = None
        self.attempts = 0
        # Max attempts from config, then task override
        default_max_attempts = Config().get("orchestration.multimodel_agent.task_max_attempts", 3)
        self.max_attempts = max_attempts_override if max_attempts_override is not None else default_max_attempts
        
        self.history: List[Dict[str, Any]] = []
        self.callback = callback # Store callback if provided

    def to_dict(self) -> Dict[str, Any]:
        return { k: (v.value if isinstance(v, Enum) else (str(v) if isinstance(v, Path) else v)) 
                 for k, v in self.__dict__.items() if k != "callback" } # Exclude non-serializable callback
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        task = cls( task_id=data["task_id"], task_type=TaskType(data["task_type"]), prompt=data["prompt"] )
        for key, value in data.items():
            if key == "priority": task.priority = TaskPriority(value)
            elif key == "status": task.status = TaskStatus(value)
            elif hasattr(task, key) and key not in ["task_type"]: setattr(task, key, value)
        return task
    
    def add_history_entry(self, event_type: str, message: str, data: Optional[Dict] = None):
        entry: Dict[str, Any] = {"timestamp": time.time(), "event_type": event_type, "message": message}
        if data: entry["data"] = {k: (str(v) if isinstance(v, Path) else v) for k,v in data.items()} # Ensure data is serializable
        self.history.append(entry)
    
    def __lt__(self, other: 'Task') -> bool: # For priority queue: higher prio value = more important
        if not isinstance(other, Task): return NotImplemented
        return (-self.priority.value, self.created_at) < (-other.priority.value, other.created_at)

    def __repr__(self): return f"<Task id={self.task_id} type={self.task_type.name} status={self.status.name}>"

class ModelCapability: # pragma: no cover
    """Represents a model's capabilities and dynamic runtime statistics."""
    def __init__(
        self, model_id: str, provider: str, model_name_for_api: str,
        capabilities: Dict[TaskType, float], 
        resource_usage: Dict[str, float], max_context_length: int,
        supports_streaming: bool = False, supports_function_calling: bool = False ):
        self.model_id = model_id; self.provider = provider; self.model_name_for_api = model_name_for_api
        self.capabilities = capabilities; self.resource_usage = resource_usage
        self.max_context_length = max_context_length; self.supports_streaming = supports_streaming
        self.supports_function_calling = supports_function_calling
        
        # Dynamic stats, initialized from config or to defaults if not found
        # This allows pre-setting some known stats or starting fresh.
        config = Config()
        stats_base_key = f"orchestration.multimodel_agent.model_initial_stats.{model_id.replace('/','_')}"
        self.availability: float = config.get(f"{stats_base_key}.availability", 1.0)
        self.success_rate: float = config.get(f"{stats_base_key}.success_rate", 0.95)
        self.average_latency_ms: float = config.get(f"{stats_base_key}.average_latency_ms", 1000.0)

    def get_capability_score(self, task_type: TaskType) -> float:
        return self.capabilities.get(task_type, 0.01) # Small default if not listed, to not be zero always

    def update_runtime_stats(self, success: bool, latency_seconds: Optional[float] = None, perf_monitor: Optional[PerformanceMonitor] = None, task_type_for_perf: Optional[TaskType]=None, initiating_agent_id: Optional[str]=None, input_tokens: int=0, output_tokens: int=0, cost_usd: float=0.0, error_type: Optional[str]=None):
        alpha = Config().get("orchestration.multimodel_agent.stats_smoothing_factor", 0.1)
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        if success and latency_seconds is not None and latency_seconds > 0:
            latency_ms = latency_seconds * 1000
            self.average_latency_ms = (1 - alpha) * self.average_latency_ms + alpha * latency_ms
        
        # Update availability based on certain errors (conceptual)
        # if not success and error_type in ["APIError", "ServiceUnavailable"]: self.availability *= 0.9 
        # else: self.availability = min(1.0, self.availability / 0.95) # Gradual recovery

        logger.debug(f"Updated runtime stats for {self.model_id}: SR={self.success_rate:.3f}, AvgLatMs={self.average_latency_ms:.0f}, Avail={self.availability:.2f}")
        
        # Record to global PerformanceMonitor if provided
        if perf_monitor: # pragma: no cover
            asyncio.create_task(perf_monitor.record_request(
                model_id=self.model_id, success=success, latency_seconds=latency_seconds or 0.0,
                input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost_usd,
                task_type=task_type_for_perf.value if task_type_for_perf else None, 
                initiating_agent_id=initiating_agent_id, error_type=error_type
            ))


    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id, "provider": self.provider, "model_name_for_api": self.model_name_for_api,
            "capabilities": {tt.value: s for tt,s in self.capabilities.items()},
            "resource_usage": self.resource_usage, "max_context_length": self.max_context_length,
            "supports_streaming": self.supports_streaming, "supports_function_calling": self.supports_function_calling,
            "availability": self.availability, "success_rate": self.success_rate, "average_latency_ms": self.average_latency_ms
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['ModelCapability']:
        if not isinstance(data, dict) or not all(k in data for k in ["model_id", "provider", "model_name_for_api"]):
            logger.warning(f"ModelCapability from_dict: Missing essential keys in data: {data}"); return None
        try:
            raw_caps = data.get("capabilities", {})
            parsed_caps = {}
            for tt_str, score_val in raw_caps.items():
                try: parsed_caps[TaskType(tt_str.lower())] = float(score_val)
                except ValueError: logger.warning(f"Invalid TaskType '{tt_str}' in capabilities for {data['model_id']}"); continue
            
            mc = cls( model_id=data["model_id"], provider=data["provider"], model_name_for_api=data["model_name_for_api"],
                      capabilities=parsed_caps, resource_usage=data.get("resource_usage",{}), 
                      max_context_length=int(data.get("max_context_length",4096)),
                      supports_streaming=bool(data.get("supports_streaming",False)), 
                      supports_function_calling=bool(data.get("supports_function_calling",False)) )
            # Allow loading of dynamic stats if they were persisted
            mc.availability = float(data.get("availability", 1.0))
            mc.success_rate = float(data.get("success_rate", 0.95))
            mc.average_latency_ms = float(data.get("average_latency_ms", 1000.0))
            return mc
        except Exception as e: logger.error(f"Error creating ModelCapability from dict for {data.get('model_id','UNKNOWN')}: {e}"); return None
    def __repr__(self): return f"<ModelCapability id={self.model_id} SR={self.success_rate:.2f} LatMs={self.average_latency_ms:.0f}>"


class MultiModelAgent: # pragma: no cover
    """
    Manages task execution using multiple LLMs.
    Loads capabilities from YAML, selects models using ModelSelector, and processes tasks.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance: cls._instance = super(MultiModelAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self, belief_system: Optional[BeliefSystem] = None, 
                 config_override: Optional[Config] = None, test_mode: bool = False):
        if hasattr(self, '_initialized') and self._initialized and not test_mode: return
        
        self.config = config_override or Config()
        self.belief_system = belief_system or BeliefSystem(test_mode=test_mode)
        self.model_registry: ModelRegistry = get_model_registry(config_override=self.config, test_mode=test_mode)
        self.model_selector: ModelSelector = ModelSelector(config=self.config) # MMA uses ModelSelector
        self.performance_monitor: PerformanceMonitor = get_performance_monitor(config_override=self.config, test_mode=test_mode) # For updating stats

        self.task_queue: asyncio.PriorityQueue[Task] = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        
        self.model_capabilities: Dict[str, ModelCapability] = {}
        self.model_handlers: Dict[str, LLMInterface] = {}

        models_dir_str = self.config.get("orchestration.multimodel_agent.models_config_dir", "data/mindx_models_config")
        self.models_config_dir = PROJECT_ROOT / models_dir_str
        if not self.models_config_dir.is_dir():
            logger.error(f"MMA: Models config directory NOT FOUND: {self.models_config_dir}. No model capabilities will be loaded.")

        self._is_shutting_down = False; self._worker_tasks: List[asyncio.Task] = []
        logger.info("MultiModelAgent (v_prod_stub) sync init complete. Call `await agent.initialize()`.")
        self._initialized = True

    async def initialize(self, num_workers: Optional[int] = None):
        """Performs asynchronous initialization."""
        logger.info("MMA: Async initialization started...")
        # ModelRegistry should be initialized by its getter if not already.
        # Handlers are obtained from registry after capabilities are loaded.
        await self._load_model_capabilities_from_files()
        self._initialize_model_handlers_from_registry()

        if not self.model_capabilities: logger.warning("MMA: No model capabilities loaded!")
        if not self.model_handlers: logger.warning("MMA: No model handlers initialized!")

        if num_workers is None: num_workers = self.config.get("orchestration.multimodel_agent.num_workers", max(1, (os.cpu_count() or 2) // 2 ))
        if num_workers > 0:
            for i in range(num_workers):
                task = asyncio.create_task(self._task_worker_loop(worker_id=i), name=f"MMA_Worker-{i}")
                self._worker_tasks.append(task)
            logger.info(f"MMA: Started {num_workers} task workers.")
        else: logger.info("MMA: No task workers started (num_workers <= 0). Tasks must be processed via direct process_task calls.")
        logger.info(f"MMA: Async init complete. {len(self.model_capabilities)} caps, {len(self.model_handlers)} handlers.")

    async def _load_model_capabilities_from_files(self):
        # (Full _load_model_capabilities_from_files from previous "COMPLETE AND FUNCTIONAL STUB v2" MMA)
        logger.info(f"MMA: Loading model capabilities from: {self.models_config_dir}")
        self.model_capabilities.clear(); loaded_count = 0
        if not self.models_config_dir.is_dir(): return
        for provider_file in self.models_config_dir.glob("*.yaml"):
            provider_name = provider_file.stem.lower()
            logger.debug(f"MMA: Processing capability file: {provider_file} for provider '{provider_name}'")
            try:
                with provider_file.open('r', encoding='utf-8') as f: provider_data = yaml.safe_load(f)
                if not isinstance(provider_data, dict) or "models" not in provider_data or not isinstance(provider_data["models"], dict): logger.warning(f"Skipping {provider_file}: Invalid format."); continue
                for model_api_name, caps_dict in provider_data["models"].items():
                    if not isinstance(caps_dict, dict): logger.warning(f"Skipping model '{model_api_name}' in {provider_file}: caps not dict."); continue
                    model_id = f"{provider_name}/{model_api_name}"
                    caps_dict_full = {"model_id": model_id, "provider": provider_name, "model_name_for_api": model_api_name, **caps_dict}
                    model_cap_obj = ModelCapability.from_dict(caps_dict_full)
                    if model_cap_obj: self.model_capabilities[model_id] = model_cap_obj; loaded_count += 1; logger.debug(f"MMA: Loaded capability for {model_id}")
                    else: logger.warning(f"MMA: Failed to parse/load capability for {model_id} from {provider_file}")
            except Exception as e: logger.error(f"MMA: Error processing capability file {provider_file}: {e}", exc_info=True)
        logger.info(f"MMA: Finished loading capabilities. Total models with capabilities: {loaded_count}")


    def _initialize_model_handlers_from_registry(self):
        # (Full _initialize_model_handlers_from_registry from previous "COMPLETE AND FUNCTIONAL STUB v2" MMA)
        logger.info("MMA: Initializing model handlers from ModelRegistry...")
        registry = get_model_registry(); self.model_handlers.clear()
        providers_needed = set(cap.provider for cap in self.model_capabilities.values())
        for provider_name in providers_needed:
            handler = registry.get_handler(provider_name) # Registry get_handler should return LLMInterface
            if handler: self.model_handlers[provider_name] = handler; logger.info(f"MMA: Acquired handler for provider '{provider_name}'.")
            else: logger.warning(f"MMA: No handler found in ModelRegistry for provider '{provider_name}'.")
        logger.info(f"MMA: Initialized {len(self.model_handlers)} provider handlers: {list(self.model_handlers.keys())}")


    async def create_task( self, task_type: Union[TaskType, str], prompt: str, priority: Union[TaskPriority, int] = TaskPriority.MEDIUM, context: Optional[Dict[str, Any]] = None, requirements: Optional[Dict[str, Any]] = None, task_id_prefix: str = "mma_task", callback: Optional[Callable[['Task'], Coroutine[Any,Any,None]]] = None ) -> Task: # pragma: no cover
        # (Full create_task from previous "COMPLETE AND FUNCTIONAL STUB v2" MMA, ensure TaskType/Priority conversion)
        try:
            tt_enum = TaskType(task_type.lower()) if isinstance(task_type, str) else task_type
            prio_enum = TaskPriority(priority) if isinstance(priority, int) else priority
            if not isinstance(tt_enum, TaskType) or not isinstance(prio_enum, TaskPriority): raise ValueError("Invalid type/priority enum.")
        except ValueError as e: logger.error(f"MMA: Failed to create task object: {e}"); raise
        task_id = f"{task_id_prefix}_{str(uuid.uuid4())[:8]}";
        task = Task(task_id, tt_enum, prompt, prio_enum, context, requirements, callback=callback)
        task.add_history_entry("creation", f"Task created with type {tt_enum.name}")
        logger.info(f"MMA: Created task {task!r}")
        return task

    async def submit_task(self, task: Task) -> str: # pragma: no cover
        """Adds a task to the priority queue for asynchronous processing by workers."""
        if not isinstance(task, Task): # pragma: no cover
             logger.error("MMA: Attempted to submit non-Task object to queue.")
             raise TypeError("Invalid object submitted to MMA queue, must be a Task.")
        await self.task_queue.put(task) # PriorityQueue expects item directly for sort with __lt__
        logger.info(f"MMA: Submitted task {task.task_id} to queue (Approx. size: {self.task_queue.qsize()})")
        return task.task_id

    async def _task_worker_loop(self, worker_id: int): # pragma: no cover
        """Worker loop that processes tasks from the priority queue."""
        logger.info(f"MMA Task Worker-{worker_id}: Started.")
        while not self._is_shutting_down:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0) # Get Task object
                if task:
                    logger.info(f"MMA Worker-{worker_id}: Picked up task {task.task_id} ('{task.prompt[:30]}...').")
                    processed_task = await self.process_task(task) # Process it
                    self.task_queue.task_done() # Signal completion
                    if processed_task.callback: # pragma: no cover
                        try: await processed_task.callback(processed_task)
                        except Exception as e_cb: logger.error(f"MMA Worker-{worker_id}: Error in task callback for {task.task_id}: {e_cb}")
            except asyncio.TimeoutError: continue # Normal, allows checking shutdown flag
            except asyncio.CancelledError: logger.info(f"MMA Task Worker-{worker_id}: Cancellation received. Exiting."); break
            except Exception as e: logger.error(f"MMA Task Worker-{worker_id}: Unhandled error: {e}", exc_info=True); await asyncio.sleep(5)
        logger.info(f"MMA Task Worker-{worker_id}: Stopped.")

    async def select_model_for_task(self, task: Task, excluded_models: Optional[Set[str]] = None) -> Optional[str]: # pragma: no cover
        """Selects model using the dedicated ModelSelector instance."""
        if not self.model_capabilities: logger.warning("MMA: No model capabilities loaded for selection"); return None
        
        selection_data = {
            "model_capabilities": self.model_capabilities,
            "task_type": task.task_type,
            "context": task.context, # Pass task context which might have selection hints
            "num_models": 1, # MMA usually selects one model per task attempt
            "excluded_models": excluded_models or set(),
            "debug_mode": self.config.get("orchestration.multimodel_agent.selection_debug_mode", False)
        }
        # Add task-specific requirements to selection_data.context.task_requirements
        selection_data["context"].setdefault("task_requirements", {}).update(task.requirements)

        selected_ids = self.model_selector.select_models(selection_data)
        if selected_ids:
            logger.info(f"MMA: ModelSelector chose '{selected_ids[0]}' for task {task.task_id}.")
            return selected_ids[0]
        else:
            logger.warning(f"MMA: ModelSelector found no suitable models for task {task.task_id}.")
            return None

    async def process_task(self, task: Task, _failed_models_session: Optional[Set[str]] = None) -> Task: # pragma: no cover
        """Processes a single task: model selection, execution, retries. _failed_models_session for internal retry tracking."""
        session_failed_models = _failed_models_session or set()
        # task.attempts is number of times this task processing has been started by a worker/direct call.
        # We need a local attempt counter for retries *within this specific process_task call*.
        local_attempts_this_session = 0
        max_retries_this_session = self.config.get("orchestration.multimodel_agent.max_retries_per_process_call", task.max_attempts)


        task.add_history_entry("processing_session_start", f"Starting processing session for task.", {"total_task_attempts_so_far": task.attempts})
        task.attempts +=1 # Increment overall task attempts

        while local_attempts_this_session < max_retries_this_session:
            local_attempts_this_session += 1
            logger.info(f"MMA: Processing task {task.task_id} (Overall Attempt: {task.attempts}, Session Attempt: {local_attempts_this_session}/{max_retries_this_session})")
            task.add_history_entry("attempt_start", f"Session attempt {local_attempts_this_session}", {"failed_in_session": list(session_failed_models)})

            model_id = await self.select_model_for_task(task, excluded_models=session_failed_models)
            if not model_id: task.error = f"No suitable models for session attempt {local_attempts_this_session}"; break
            
            capability = self.model_capabilities.get(model_id)
            if not capability: task.error = f"InternalError: Selected {model_id} lacks capability data."; session_failed_models.add(model_id); continue
            handler = self.model_handlers.get(capability.provider)
            if not handler: task.error = f"Handler for provider '{capability.provider}' not found."; session_failed_models.add(model_id); continue
            
            task.status = TaskStatus.IN_PROGRESS; task.assigned_model = model_id; task.started_at = time.time()
            if task.task_id not in self.active_tasks and task.task_id not in self.completed_tasks : self.active_tasks[task.task_id] = task

            latency_s: Optional[float] = None; success_flag = False; error_type_str: Optional[str] = None
            # Get token/cost info from PerformanceMonitor if it's part of handler's result
            # This is conceptual; LLMHandler would need to return this. For now, assume 0.
            tokens_in = 0; tokens_out = 0; cost = 0.0 
            try:
                logger.info(f"MMA: Executing task {task.task_id} with {model_id} (API: {capability.model_name_for_api})")
                gen_params = task.context.get("generation_params", {})
                if task.requirements.get("max_tokens"): gen_params["max_tokens"] = task.requirements["max_tokens"] # Task reqs can override context
                if task.requirements.get("temperature") is not None: gen_params["temperature"] = task.requirements["temperature"]
                if task.requirements.get("json_mode") is not None: gen_params["json_mode"] = task.requirements["json_mode"]
                
                # The handler's `generate` method is expected to be part of LLMInterface
                response_content_or_error = await handler.generate(prompt=task.prompt, model=capability.model_name_for_api, **gen_params)
                task.completed_at = time.time(); latency_s = task.completed_at - task.started_at
                
                if isinstance(response_content_or_error, str) and response_content_or_error.startswith("Error:"):
                    raise RuntimeError(f"LLM Handler Error: {response_content_or_error}")

                task.status = TaskStatus.COMPLETED; task.result = response_content_or_error; task.error = None; success_flag = True
                task.add_history_entry("attempt_success", f"Completed with {model_id} in {latency_s:.2f}s", {"model_used": model_id}); break
            except Exception as e:
                if task.started_at: task.completed_at = time.time(); latency_s = task.completed_at - task.started_at
                task.error = f"Attempt {local_attempts_this_session} with {model_id}: {type(e).__name__} - {str(e)[:200]}"; error_type_str = type(e).__name__
                logger.error(f"MMA: Error task {task.task_id} with {model_id} (Attempt {local_attempts_this_session}): {e}", exc_info=True)
                session_failed_models.add(model_id)
                task.add_history_entry("attempt_failure", f"Failed with {model_id}: {task.error}", {"model_used": model_id, "error_type": error_type_str})
                if local_attempts_this_session >= max_retries_this_session: logger.error(f"MMA: Task {task.task_id} failed max session retries ({max_retries_this_session}). Final error: {task.error}"); break
                else: # Prepare for next attempt in loop
                     retry_delay_s = self.config.get("orchestration.multimodel_agent.retry_delay_seconds", 1.0) * (2**(local_attempts_this_session-1))
                     logger.warning(f"MMA: Retrying task {task.task_id} (next session attempt {local_attempts_this_session+1}) in {retry_delay_s:.1f}s."); await asyncio.sleep(retry_delay_s)
            finally: # Update stats regardless of success/failure of this attempt
                if model_id and capability : capability.update_runtime_stats(success_flag, latency_s, self.performance_monitor, task.task_type, task.context.get("initiating_agent_id"), tokens_in, tokens_out, cost, error_type_str)
        
        if task.status != TaskStatus.COMPLETED: task.status = TaskStatus.FAILED;
        if not task.error and task.status == TaskStatus.FAILED: task.error = f"Task failed after {local_attempts_this_session} session attempts with no specific final error."
        
        if task.task_id in self.active_tasks: del self.active_tasks[task.task_id]
        self.completed_tasks[task.task_id] = task
        await self.belief_system.add_belief(f"mindx.mma.task.{task.task_id}.final_status", task.status.value, 0.9, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600*24)
        return task

    async def shutdown(self): # pragma: no cover
        logger.info("MMA: Shutting down..."); self._is_shutting_down = True
        # Cancel and await worker tasks
        if self._worker_tasks:
            for worker_task in self._worker_tasks:
                if not worker_task.done(): worker_task.cancel()
            results = await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, asyncio.CancelledError): logger.info(f"MMA Worker-{i} successfully cancelled.")
                elif isinstance(result, Exception): logger.error(f"MMA Worker-{i} error during shutdown: {result}", exc_info=result)
        # Persist any final state (e.g., updated model_capabilities if stats are saved there)
        # This might involve writing self.model_capabilities back to YAML or a cache.
        logger.info("MMA: Shutdown complete.")
    
    # --- Getters (Unchanged) ---
    def get_task(self, task_id: str) -> Optional[Task]: return self.active_tasks.get(task_id) or self.completed_tasks.get(task_id) or next((t for prio,ts,t in list(self.task_queue._queue) if t.task_id == task_id), None) # pragma: no cover
    def get_model_capability(self, model_id: str) -> Optional[ModelCapability]: return self.model_capabilities.get(model_id) # pragma: no cover
    def get_handler(self, provider_name: str) -> Optional[LLMInterface]: handler = self.model_handlers.get(provider_name.lower()); return handler # pragma: no cover
    def get_all_model_capabilities(self) -> Dict[str, ModelCapability]: return self.model_capabilities.copy() # pragma: no cover

    @classmethod
    async def reset_instance_async(cls): # For testing # pragma: no cover
        async with cls._lock:
            if cls._instance: await cls._instance.shutdown(); cls._instance._initialized = False; cls._instance = None
        logger.debug("MultiModelAgent instance reset asynchronously.")

_mma_factory_lock = asyncio.Lock()
async def get_multimodel_agent_async(belief_system: Optional[BeliefSystem] = None, config_override: Optional[Config] = None, test_mode: bool = False) -> MultiModelAgent: # pragma: no cover
    """Async factory for MultiModelAgent singleton. Ensures initialize() is called."""
    async with _mma_factory_lock: # Use a dedicated lock for factory to prevent re-entry during init
        if MultiModelAgent._instance is None or test_mode:
            if test_mode and MultiModelAgent._instance is not None:
                await MultiModelAgent._instance.shutdown() # Gracefully stop old test instance
                MultiModelAgent._instance = None # Force re-creation
            
            eff_belief_system = belief_system
            if eff_belief_system is None:
                logger.warning("MMA Factory: BeliefSystem not provided, creating default for MMA.")
                eff_belief_system = BeliefSystem(test_mode=test_mode)

            instance = MultiModelAgent(belief_system=eff_belief_system, config_override=config_override, test_mode=test_mode)
            await instance.initialize() # CRITICAL: Call async initialize
            MultiModelAgent._instance = instance
    return MultiModelAgent._instance
