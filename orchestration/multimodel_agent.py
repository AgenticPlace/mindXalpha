# mindx/orchestration/multimodel_agent.py
import os
import logging
import asyncio
import json
import time
import yaml # Needed for loading capability files
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from enum import Enum

# Use relative imports within the 'mindx' package structure
from mindx.utils.config import Config, PROJECT_ROOT # Use canonical PROJECT_ROOT
from mindx.utils.logging_config import get_logger
from mindx.llm.model_registry import get_model_registry, ModelRegistry
from mindx.llm.llm_interface import LLMInterface
from mindx.core.belief_system import BeliefSystem #, BeliefSource (if used by MMA)

logger = get_logger(__name__)

# --- Enums (TaskType, TaskPriority, TaskStatus) ---
class TaskType(Enum): # pragma: no cover
    """Defines the types of tasks the MultiModelAgent can handle."""
    GENERATION = "generation"; REASONING = "reasoning"; SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"; CODE_REVIEW = "code_review"; PLANNING = "planning"
    RESEARCH = "research"; SELF_IMPROVEMENT_ANALYSIS = "self_improvement_analysis" # Specific for SIA's needs via Coord

class TaskPriority(Enum): LOW = 1; MEDIUM = 2; HIGH = 3; CRITICAL = 4 # pragma: no cover
class TaskStatus(Enum): PENDING = "pending"; ASSIGNED = "assigned"; IN_PROGRESS = "in_progress"; COMPLETED = "completed"; FAILED = "failed"; CANCELLED = "cancelled" # pragma: no cover


# --- Data Classes (Task, ModelCapability) ---
class Task: # pragma: no cover
    """Represents a task to be processed by a model within the MultiModelAgent."""
    def __init__(
        self, task_id: str, task_type: TaskType, prompt: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        requirements: Optional[Dict[str, Any]] = None, # e.g., {"min_context_length": 8000, "target_model_id": "ollama/specific_one"}
        max_attempts: Optional[int] = None
    ):
        self.task_id = task_id; self.task_type = task_type; self.prompt = prompt
        self.priority = priority; self.context = context or {}; self.requirements = requirements or {}
        self.status = TaskStatus.PENDING; self.assigned_model: Optional[str] = None # Full model_id like "provider/model_api_name"
        self.result: Any = None; self.error: Optional[str] = None
        self.created_at = time.time(); self.started_at: Optional[float] = None; self.completed_at: Optional[float] = None
        self.attempts = 0
        self.max_attempts = max_attempts if max_attempts is not None else Config().get("orchestration.multimodel_agent.task_max_attempts", 3)
        self.history: List[Dict[str, Any]] = [] # For logging attempts and model choices within this task

    def to_dict(self) -> Dict[str, Any]:
        return { k: (v.value if isinstance(v, Enum) else v) for k, v in self.__dict__.items() }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        # Simplified from_dict for this stub version
        task = cls( task_id=data["task_id"], task_type=TaskType(data["task_type"]), prompt=data["prompt"] )
        for key, value in data.items(): # Apply other fields
            if hasattr(task, key) and key not in ["task_type", "priority", "status"]: setattr(task, key, value)
        if "priority" in data: task.priority = TaskPriority(data["priority"])
        if "status" in data: task.status = TaskStatus(data["status"])
        return task
    def add_history_entry(self, event_type: str, message: str, data: Optional[Dict] = None):
        entry = {"timestamp": time.time(), "event_type": event_type, "message": message}
        if data: entry.update(data)
        self.history.append(entry)
    def __repr__(self): return f"<Task id={self.task_id} type={self.task_type.name} status={self.status.name}>"

class ModelCapability: # pragma: no cover
    """Represents a model's capabilities and dynamic runtime statistics."""
    def __init__(
        self, model_id: str, provider: str, model_name_for_api: str, # The name/tag for the provider's API
        capabilities: Dict[TaskType, float], # TaskType Enum -> Score (0.0-1.0)
        resource_usage: Dict[str, float], # e.g., {"cost_per_input_token": 0.0001, "cost_per_output_token": 0.0002}
        max_context_length: int,
        supports_streaming: bool = False, supports_function_calling: bool = False ):
        self.model_id = model_id # Unique system-wide ID, e.g., "ollama/nous-hermes2:latest"
        self.provider = provider
        self.model_name_for_api = model_name_for_api
        self.capabilities = capabilities # {TaskType.REASONING: 0.8, ...}
        self.resource_usage = resource_usage
        self.max_context_length = max_context_length
        self.supports_streaming = supports_streaming
        self.supports_function_calling = supports_function_calling
        # Dynamic stats, initialized or loaded from persistence
        self.availability: float = Config().get(f"model_stats.{model_id}.availability", 1.0)
        self.success_rate: float = Config().get(f"model_stats.{model_id}.success_rate", 0.95) # Default optimistic
        self.average_latency_ms: float = Config().get(f"model_stats.{model_id}.average_latency_ms", 1000.0) # Default 1s

    def get_capability_score(self, task_type: TaskType) -> float:
        return self.capabilities.get(task_type, 0.0) # Default to 0 if task_type not explicitly listed

    def update_runtime_stats(self, success: bool, latency_seconds: Optional[float] = None):
        alpha = Config().get("orchestration.multimodel_agent.stats_smoothing_factor", 0.1) # For EMA
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        if success and latency_seconds is not None and latency_seconds > 0:
            latency_ms = latency_seconds * 1000
            self.average_latency_ms = (1 - alpha) * self.average_latency_ms + alpha * latency_ms
        # Could also update availability based on certain error types
        # Persisting these updated stats is a TODO for a full implementation
        logger.debug(f"Updated runtime stats for {self.model_id}: SR={self.success_rate:.3f}, AvgLatMs={self.average_latency_ms:.0f}")

    def to_dict(self) -> Dict[str, Any]: # For saving capabilities/stats if needed
        return {
            "model_id": self.model_id, "provider": self.provider, "model_name_for_api": self.model_name_for_api,
            "capabilities": {tt.value: s for tt,s in self.capabilities.items()}, # Store enum values
            "resource_usage": self.resource_usage, "max_context_length": self.max_context_length,
            "supports_streaming": self.supports_streaming, "supports_function_calling": self.supports_function_calling,
            "availability": self.availability, "success_rate": self.success_rate, "average_latency_ms": self.average_latency_ms
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['ModelCapability']: # Used by _load_model_capabilities_from_files
        if not isinstance(data, dict) or not data.get("model_id"): return None
        try:
            raw_caps = data.get("capabilities", {})
            parsed_caps = {TaskType(k.lower()): float(v) for k,v in raw_caps.items() if k.lower() in TaskType._value2member_map_}
            
            mc = cls( model_id=data["model_id"], provider=data["provider"], model_name_for_api=data["model_name_for_api"],
                      capabilities=parsed_caps, resource_usage=data.get("resource_usage",{}), max_context_length=int(data.get("max_context_length",4096)),
                      supports_streaming=bool(data.get("supports_streaming",False)), supports_function_calling=bool(data.get("supports_function_calling",False)) )
            # Load dynamic stats if present in data (e.g., from persisted cache)
            mc.availability = float(data.get("availability", 1.0)); mc.success_rate = float(data.get("success_rate", 0.95)); mc.average_latency_ms = float(data.get("average_latency_ms", 1000.0))
            return mc
        except Exception as e: logger.error(f"Error creating ModelCapability from dict for {data.get('model_id','UNKNOWN')}: {e}"); return None
    def __repr__(self): return f"<ModelCapability id={self.model_id} SR={self.success_rate:.2f} LatMs={self.average_latency_ms:.0f}>"


class MultiModelAgent: # pragma: no cover
    """
    Handles execution of tasks against LLMs, model selection, and basic task management.
    Loads model capabilities from YAML files.
    """
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MultiModelAgent, cls).__new__(cls)
        return cls._instance

    def __init__(self, belief_system: Optional[BeliefSystem] = None, config_override: Optional[Config] = None, test_mode: bool = False):
        if hasattr(self, '_initialized') and self._initialized and not test_mode: return
        
        self.config = config_override or Config()
        self.belief_system = belief_system or BeliefSystem(test_mode=test_mode) # Get/create singleton
        self.model_registry: ModelRegistry = get_model_registry(config_override=self.config, test_mode=test_mode) # Get singleton

        self.task_queue: asyncio.PriorityQueue[Tuple[int, Task]] = asyncio.PriorityQueue() # (priority_val, Task)
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {} # Could be LRU cache or persisted
        
        self.model_capabilities: Dict[str, ModelCapability] = {} # Populated by initialize: model_id -> ModelCapability
        self.model_handlers: Dict[str, LLMInterface] = {} # Populated by initialize: provider_name -> LLMHandler from llm_factory

        models_dir_str = self.config.get("orchestration.multimodel_agent.models_config_dir", "data/mindx_models_config")
        self.models_config_dir = PROJECT_ROOT / models_dir_str
        if not self.models_config_dir.is_dir():
            logger.warning(f"MMA: Models config directory NOT FOUND: {self.models_config_dir}. MMA may not load any capabilities.")

        # Model selection weights and parameters
        self.selection_weights = self.config.get("orchestration.multimodel_agent.selection_weights", 
            {"capability": 1.0, "success_rate": 0.8, "latency": 0.5, "cost": 0.3, "requirements_match": 2.0, "provider_preference": 0.2}
        )
        self.provider_preferences = self.config.get("orchestration.multimodel_agent.provider_preferences", 
            {"gemini": 1.0, "openai": 0.9, "anthropic": 0.8, "ollama": 0.5} # Example preferences
        )
        self.min_availability_threshold = self.config.get("orchestration.multimodel_agent.min_availability_threshold", 0.1)

        self._is_shutting_down = False
        self._worker_tasks: List[asyncio.Task] = []

        logger.info("MultiModelAgent synchronous initialization complete. Call `await agent.initialize()`.")
        self._initialized = True

    async def initialize(self, num_workers: Optional[int] = None):
        """Performs asynchronous initialization: loads capabilities, initializes handlers, starts workers."""
        logger.info("MMA: Starting asynchronous initialization...")
        await self._load_model_capabilities_from_files() # Load from YAML
        self._initialize_model_handlers_from_registry()  # Get handlers from ModelRegistry

        if not self.model_capabilities: logger.warning("MMA: No model capabilities loaded after initialization!")
        if not self.model_handlers: logger.warning("MMA: No model handlers initialized from registry!")

        # Start worker tasks to process the queue
        if num_workers is None: num_workers = self.config.get("orchestration.multimodel_agent.num_workers", 2)
        for i in range(num_workers):
            task = asyncio.create_task(self._task_worker_loop(worker_id=i))
            self._worker_tasks.append(task)
        logger.info(f"MMA: Asynchronous initialization complete. Started {num_workers} task workers. Ready with {len(self.model_capabilities)} models and {len(self.model_handlers)} provider handlers.")

    async def _load_model_capabilities_from_files(self):
        """Loads model capabilities from provider-specific YAML files in models_config_dir."""
        logger.info(f"MMA: Loading model capabilities from: {self.models_config_dir}")
        self.model_capabilities.clear()
        loaded_count = 0
        if not self.models_config_dir.is_dir(): return

        for provider_file in self.models_config_dir.glob("*.yaml"): # e.g., ollama.yaml, gemini.yaml
            provider_name = provider_file.stem.lower()
            logger.debug(f"MMA: Processing capability file: {provider_file} for provider '{provider_name}'")
            try:
                with provider_file.open('r', encoding='utf-8') as f: provider_data = yaml.safe_load(f)
                if not isinstance(provider_data, dict) or "models" not in provider_data or not isinstance(provider_data["models"], dict):
                    logger.warning(f"Skipping {provider_file}: Invalid format. Expected top-level 'models' dictionary."); continue
                
                for model_api_name, caps_dict in provider_data["models"].items():
                    if not isinstance(caps_dict, dict): logger.warning(f"Skipping model '{model_api_name}' in {provider_file}: capabilities not a dict."); continue
                    
                    # model_id is system-wide unique, model_name_for_api is what handler uses
                    model_id = f"{provider_name}/{model_api_name}" # e.g., ollama/llama3:8b-instruct
                    
                    # Add provider and model_name_for_api to caps_dict for ModelCapability.from_dict
                    caps_dict_full = {"model_id": model_id, "provider": provider_name, "model_name_for_api": model_api_name, **caps_dict}
                    
                    model_cap_obj = ModelCapability.from_dict(caps_dict_full)
                    if model_cap_obj:
                        self.model_capabilities[model_id] = model_cap_obj; loaded_count += 1
                        logger.debug(f"MMA: Loaded capability for {model_id}")
                    else: logger.warning(f"MMA: Failed to load/parse capability for {model_id} from {provider_file}")
            except Exception as e: logger.error(f"MMA: Error processing capability file {provider_file}: {e}", exc_info=True)
        logger.info(f"MMA: Finished loading capabilities. Total models with capabilities: {loaded_count}")

    def _initialize_model_handlers_from_registry(self):
        """Initializes model handlers by getting them from the ModelRegistry."""
        logger.info("MMA: Initializing model handlers from ModelRegistry...")
        registry = get_model_registry() # This ensures registry is initialized
        
        # We need handlers for each provider that we have capabilities for
        providers_needed = set(cap.provider for cap in self.model_capabilities.values())
        for provider_name in providers_needed:
            handler = registry.get_handler(provider_name)
            if handler:
                self.model_handlers[provider_name] = handler
                logger.info(f"MMA: Acquired handler for provider '{provider_name}'.")
            else: # pragma: no cover
                logger.warning(f"MMA: No handler found in ModelRegistry for provider '{provider_name}'. Models from this provider will not be usable.")
        logger.info(f"MMA: Initialized {len(self.model_handlers)} provider handlers: {list(self.model_handlers.keys())}")

    async def create_task( self, task_type: Union[TaskType, str], prompt: str, priority: Union[TaskPriority, int] = TaskPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None, requirements: Optional[Dict[str, Any]] = None,
        task_id_prefix: str = "mma_task" ) -> Task: # pragma: no cover
        # (Full create_task from previous MMA - this is mostly fine)
        try:
            if isinstance(task_type, str): task_type_enum = TaskType(task_type.lower())
            elif isinstance(task_type, TaskType): task_type_enum = task_type
            else: raise ValueError(f"Invalid task_type: {task_type}")
            if isinstance(priority, int): priority_enum = TaskPriority(priority)
            elif isinstance(priority, TaskPriority): priority_enum = priority
            else: raise ValueError(f"Invalid priority: {priority}")
        except ValueError as e: logger.error(f"MMA: Failed to create task: {e}"); raise
        task_id = f"{task_id_prefix}_{str(uuid.uuid4())[:8]}";
        task = Task(task_id, task_type_enum, prompt, priority_enum, context, requirements)
        task.add_history_entry("creation", f"Task created with type {task_type_enum.name}")
        logger.info(f"MMA: Created task {task!r}")
        return task

    async def add_task_to_queue(self, task: Task): # pragma: no cover
         if not isinstance(task, Task): logger.error("MMA: Attempted to add non-Task object to queue"); return
         # PriorityQueue uses (-priority, time_added, task) for min-heap behavior
         # Lower numeric priority value means higher logical priority.
         await self.task_queue.put((-task.priority.value, time.time(), task))
         logger.info(f"MMA: Added task {task.task_id} to queue (Approx. size: {self.task_queue.qsize()})")

    async def _task_worker_loop(self, worker_id: int): # pragma: no cover
        """Worker loop that processes tasks from the queue."""
        logger.info(f"MMA Task Worker-{worker_id}: Started.")
        while not self._is_shutting_down:
            try:
                # Wait for a task with a timeout to allow checking shutdown flag
                _, _, task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                if task:
                    logger.info(f"MMA Worker-{worker_id}: Picked up task {task.task_id}.")
                    await self.process_task(task) # This is the core processing logic
                    self.task_queue.task_done() # Signal completion for queue management
                # If timeout, loop continues and checks _is_shutting_down
            except asyncio.TimeoutError: # pragma: no cover
                continue # Just loop again to check shutdown flag
            except asyncio.CancelledError: # pragma: no cover
                logger.info(f"MMA Task Worker-{worker_id}: Cancellation received. Exiting."); break
            except Exception as e: # pragma: no cover
                logger.error(f"MMA Task Worker-{worker_id}: Unhandled error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(5) # Sleep a bit before retrying to avoid fast error loops
        logger.info(f"MMA Task Worker-{worker_id}: Stopped.")


    async def select_model_for_task(self, task: Task, excluded_models: Optional[Set[str]] = None) -> Optional[str]: # pragma: no cover
        # (Full select_model_for_task logic from previous MMA, ensure it uses self.selection_weights and self.provider_preferences)
        if not self.model_capabilities: logger.warning("MMA: No model capabilities loaded for selection"); return None
        excluded = excluded_models or set(); task_reqs = task.requirements or {}
        logger.debug(f"MMA: Selecting model for task {task.task_id} (Type: {task.task_type.name}), excluding: {excluded}")
        
        candidate_scores: Dict[str, float] = {}
        for model_id, cap in self.model_capabilities.items():
            if model_id in excluded or cap.availability < self.min_availability_threshold: continue

            score = cap.get_capability_score(task.task_type) * self.selection_weights.get("capability", 1.0)
            score *= (cap.success_rate * self.selection_weights.get("success_rate", 1.0) + (1.0 - self.selection_weights.get("success_rate", 1.0))) # Blend with default if weight < 1
            latency_factor = 1.0 / max(0.01, cap.average_latency_ms / 1000.0) # Normalize latency (seconds)
            score *= (latency_factor * self.selection_weights.get("latency", 0.2) + (1.0 - self.selection_weights.get("latency", 0.2)))
            
            # Cost factor (lower cost is better)
            # Assuming cost_per_token is for total tokens (input+output average). More complex cost models could be used.
            cost_per_k_tokens = cap.resource_usage.get("cost_per_kilo_total_tokens", cap.resource_usage.get("cost_per_token", 0.0) * 1000)
            cost_factor = 1.0 / max(0.01, 1.0 + cost_per_k_tokens) # Normalize cost, add 1 to avoid division by zero for free models
            score *= (cost_factor * self.selection_weights.get("cost", 0.1) + (1.0 - self.selection_weights.get("cost", 0.1)))
            
            # Provider preference
            score *= self.provider_preferences.get(cap.provider, 0.1) # Default low preference if provider not listed

            # Requirements matching
            req_match_score = 1.0
            if task_reqs.get("min_context_length") and cap.max_context_length < task_reqs["min_context_length"]: req_match_score *= 0.1 # Heavy penalty
            if task_reqs.get("supports_streaming") and not cap.supports_streaming: req_match_score *= 0.5
            if task_reqs.get("supports_function_calling") and not cap.supports_function_calling: req_match_score *= 0.5
            if task_reqs.get("target_model_id") and model_id != task_reqs["target_model_id"]: req_match_score = 0.0 # Must match if specified
            if task_reqs.get("target_provider") and cap.provider != task_reqs["target_provider"]: req_match_score = 0.0

            score *= (req_match_score * self.selection_weights.get("requirements_match", 2.0) + (1.0-self.selection_weights.get("requirements_match", 2.0)*(1-req_match_score))) # Complex weighting for match

            if score > 0.01: candidate_scores[model_id] = score # Threshold to be considered
            logger.debug(f"MMA Select: Candidate {model_id}, CapScore={cap.get_capability_score(task.task_type):.2f}, SR={cap.success_rate:.2f}, LatF={latency_factor:.2f}, CostF={cost_factor:.2f}, ProvP={self.provider_preferences.get(cap.provider,0.1):.2f}, ReqM={req_match_score:.2f} -> Final Score={score:.3f}")
        
        if not candidate_scores: logger.warning(f"MMA: No suitable models found for task {task.task_id}"); return None
        # Select best model (highest score, then tie-break)
        # Simple max score selection for now. Tie-breaking can be added if needed.
        best_model_id = max(candidate_scores, key=candidate_scores.get)
        logger.info(f"MMA: Selected model '{best_model_id}' for task {task.task_id} (Score: {candidate_scores[best_model_id]:.3f})")
        return best_model_id


    async def process_task(self, task: Task, _failed_models_session: Optional[Set[str]] = None) -> Task: # _failed_models for internal retry
        """Processes a single task, including model selection, execution, and retries."""
        # This method is now the core of what a worker would call, or what Coordinator calls directly.
        # The _failed_models_session is to track failures *within this specific process_task call's retries*.
        
        session_failed_models = _failed_models_session or set() # Models that failed during this current processing attempt (incl. retries)
        initial_task_attempts = task.attempts # Preserve original attempts before this session

        while task.attempts < task.max_attempts:
            task.attempts += 1
            current_attempt_total = task.attempts 
            logger.info(f"MMA: Processing task {task.task_id} (Attempt {current_attempt_total}/{task.max_attempts})")
            task.add_history_entry("attempt_start", f"Starting attempt {current_attempt_total}", {"failed_in_session_so_far": list(session_failed_models)})

            model_id_for_attempt = await self.select_model_for_task(task, excluded_models=session_failed_models)

            if not model_id_for_attempt:
                 task.error = f"No suitable models found for attempt {current_attempt_total} (excluded: {session_failed_models})"
                 logger.warning(f"MMA: Task {task.task_id} attempt {current_attempt_total} failed: {task.error}")
                 break # No model, cannot proceed with this task processing session

            capability = self.model_capabilities.get(model_id_for_attempt)
            if not capability: # Should not happen if select_model_for_task works
                 task.error = f"InternalError: Selected model {model_id_for_attempt} lacks capability data."; logger.error(task.error); session_failed_models.add(model_id_for_attempt); continue

            handler = self.model_handlers.get(capability.provider)
            if not handler:
                 task.error = f"Handler for provider '{capability.provider}' not found for model {model_id_for_attempt}."; logger.error(task.error); session_failed_models.add(model_id_for_attempt); continue
            
            task.status = TaskStatus.IN_PROGRESS; task.assigned_model = model_id_for_attempt
            task.started_at = time.time()
            if task.task_id not in self.active_tasks and task.task_id not in self.completed_tasks : self.active_tasks[task.task_id] = task # Add to active if first attempt in session

            latency_sec: Optional[float] = None
            try:
                logger.info(f"MMA: Executing task {task.task_id} with {model_id_for_attempt} (API model: {capability.model_name_for_api})")
                # Generation parameters from task context or requirements
                gen_params = task.context.get("generation_params", {}) 
                # Allow task requirements to override context gen_params
                if task.requirements.get("max_tokens"): gen_params["max_tokens"] = task.requirements["max_tokens"]
                if task.requirements.get("temperature"): gen_params["temperature"] = task.requirements["temperature"]
                if task.requirements.get("json_mode"): gen_params["json_mode"] = task.requirements["json_mode"]
                
                # Call the LLMInterface's generate method
                response_content = await handler.generate( prompt=task.prompt, model=capability.model_name_for_api, **gen_params )

                task.completed_at = time.time()
                latency_sec = task.completed_at - task.started_at
                
                # Check for error markers in response (some handlers might return error strings)
                if isinstance(response_content, str) and response_content.startswith("Error:"): # pragma: no cover
                    raise RuntimeError(f"LLM Handler returned an error: {response_content}")

                task.status = TaskStatus.COMPLETED; task.result = response_content; task.error = None
                capability.update_runtime_stats(success=True, latency_seconds=latency_sec)
                # Performance Monitor call should be done by the LLMHandler itself after an API call
                # self.performance_monitor.record_request(model_id_for_attempt, True, latency_sec, ...)
                task.add_history_entry("attempt_success", f"Completed with {model_id_for_attempt} in {latency_sec:.2f}s", {"model_used": model_id_for_attempt})
                logger.info(f"MMA: Completed task {task.task_id} with {model_id_for_attempt} in {latency_sec:.2f}s.")
                break # Successful attempt, exit retry loop

            except Exception as e:
                logger.error(f"MMA: Error processing task {task.task_id} with {model_id_for_attempt} (Attempt {current_attempt_total}): {e}", exc_info=True)
                if task.started_at: task.completed_at = time.time(); latency_sec = task.completed_at - task.started_at
                task.error = f"Attempt {current_attempt_total} with {model_id_for_attempt}: {type(e).__name__} - {str(e)[:200]}"
                capability.update_runtime_stats(success=False, latency_seconds=latency_sec)
                # self.performance_monitor.record_request(model_id_for_attempt, False, latency_sec, ..., error_type=type(e).__name__)
                session_failed_models.add(model_id_for_attempt)
                task.add_history_entry("attempt_failure", f"Failed with {model_id_for_attempt}: {task.error}", {"model_used": model_id_for_attempt, "error_type": type(e).__name__})

                if task.attempts >= task.max_attempts:
                     logger.error(f"MMA: Task {task.task_id} failed after max attempts ({task.max_attempts}). Final error: {task.error}")
                     break # Max attempts reached
                else: # Prepare for next attempt in loop
                     retry_delay_seconds = self.config.get("orchestration.multimodel_agent.retry_delay_seconds", 1.0) * (2**(task.attempts-1)) # Exponential backoff
                     logger.warning(f"MMA: Retrying task {task.task_id} (next attempt {task.attempts+1}/{task.max_attempts}) in {retry_delay_seconds:.1f}s.")
                     await asyncio.sleep(retry_delay_seconds)
                     # Continue to the next iteration of the while loop for retry
        
        # Finalize task state after loop
        if task.status != TaskStatus.COMPLETED:
            task.status = TaskStatus.FAILED
            if not task.error: task.error = f"Task failed after {task.attempts} attempts with no specific final error." # Should have error from last attempt
        
        if task.task_id in self.active_tasks: del self.active_tasks[task.task_id]
        self.completed_tasks[task.task_id] = task
        await self.belief_system.add_belief(f"mindx.mma.task.{task.task_id}.status", task.status.value, 0.9, BeliefSource.SELF_ANALYSIS, ttl_seconds=3600*24)

        return task

    async def shutdown(self): # pragma: no cover
        """Gracefully shuts down the MMA, cancelling worker tasks."""
        logger.info("MMA: Shutting down...")
        self._is_shutting_down = True
        # Cancel and await worker tasks
        for worker_task in self._worker_tasks:
            if not worker_task.done():
                worker_task.cancel()
        results = await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError): logger.info(f"MMA Worker-{i} successfully cancelled.")
            elif isinstance(result, Exception): logger.error(f"MMA Worker-{i} error during shutdown: {result}", exc_info=result)
        logger.info("MMA: All worker tasks processed for shutdown.")
        # Persist any final state if needed (e.g. model capabilities with updated runtime stats)
        # self._save_model_capabilities_to_files() # Example if dynamic stats were persisted

    # --- Getters ---
    def get_task(self, task_id: str) -> Optional[Task]: return self.active_tasks.get(task_id) or self.completed_tasks.get(task_id) or next((t for _,_,t in list(self.task_queue._queue) if t.task_id == task_id), None) # pragma: no cover
    def get_model_capability(self, model_id: str) -> Optional[ModelCapability]: return self.model_capabilities.get(model_id) # pragma: no cover
    def get_handler(self, provider_name: str) -> Optional[LLMInterface]: handler = self.model_handlers.get(provider_name.lower()); return handler # pragma: no cover
    def get_all_model_capabilities(self) -> Dict[str, ModelCapability]: return self.model_capabilities.copy() # pragma: no cover
    
    @classmethod
    async def reset_instance_async(cls): # For testing # pragma: no cover
        async with cls._lock:
            if cls._instance: await cls._instance.shutdown(); cls._instance._initialized = False; cls._instance = None
        logger.debug("MultiModelAgent instance reset asynchronously.")

_mma_instance_lock = asyncio.Lock()
async def get_multimodel_agent_async(belief_system: Optional[BeliefSystem] = None, config_override: Optional[Config] = None, test_mode: bool = False) -> MultiModelAgent: # pragma: no cover
    """Async factory for MultiModelAgent singleton."""
    if not MultiModelAgent._instance or test_mode:
        async with _mma_instance_lock:
            if MultiModelAgent._instance is None or test_mode:
                if test_mode and MultiModelAgent._instance is not None:
                    await MultiModelAgent._instance.shutdown()
                    MultiModelAgent._instance = None
                
                # Resolve belief_system if not provided
                eff_belief_system = belief_system
                if eff_belief_system is None: # pragma: no cover # Should be provided by Coordinator's factory
                    # In a real app, this might try to get/create a global BeliefSystem instance
                    # For now, assume it's passed or create a default one for MMA.
                    logger.warning("MMA Factory: BeliefSystem not provided, creating a default one for MMA.")
                    eff_belief_system = BeliefSystem(test_mode=test_mode) # Ensures it respects test_mode

                instance = MultiModelAgent(belief_system=eff_belief_system, config_override=config_override, test_mode=test_mode)
                await instance.initialize() # CRITICAL: Call async initialize
                MultiModelAgent._instance = instance
    return MultiModelAgent._instance
