# monitoring/performance_monitor.py
import time
import json
import asyncio
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Any, Optional, List, Deque # Added Deque
import os # For os.getenv

# Corrected imports
from utils.config import Config, PROJECT_ROOT
from utils.logging_config import get_logger

logger = get_logger(__name__)

class PerformanceMonitor:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs): # pragma: no cover
        if not cls._instance:
            cls._instance = super(PerformanceMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_override: Optional[Config] = None, test_mode: bool = False): # pragma: no cover
        if hasattr(self, '_initialized') and self._initialized and not test_mode:
            return

        self.config = config_override or Config()
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_latency_ms": 0,
            "latencies_ms": deque(maxlen=self.config.get("monitoring.performance.latency_window_size", 100)), # Store last N latencies
            "error_types": defaultdict(int),
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0, # In USD, for example
            "last_called_at": 0.0,
            "first_called_at": 0.0
        })
        self.metrics_file_path_str = self.config.get(
            "monitoring.performance.metrics_file",
            "data/performance_metrics.json"
        )
        self.metrics_file_path = PROJECT_ROOT / self.metrics_file_path_str
        self._load_metrics_sync() # Load synchronously during init

        self.save_periodically: bool = self.config.get("monitoring.performance.save_periodically", True)
        self.periodic_save_interval_seconds: float = self.config.get("monitoring.performance.periodic_save_interval_seconds", 300.0)
        self.save_on_request_count: int = self.config.get("monitoring.performance.save_on_request_count", 20) # Save every N requests if periodic is off
        self._request_counter_since_last_save: int = 0
        self._save_lock = asyncio.Lock()
        self._periodic_save_task: Optional[asyncio.Task] = None

        if self.save_periodically:
            self.start_periodic_save()

        logger.info(
            f"PerformanceMonitor initialized. Metrics File: {self.metrics_file_path}. "
            f"Save every: {self.save_on_request_count} reqs (if periodic save off). "
            f"Periodic Save: {self.save_periodically}, interval {self.periodic_save_interval_seconds}s."
        )
        self._initialized = True

    def _load_metrics_sync(self): # pragma: no cover
        if self.metrics_file_path.exists():
            try:
                with self.metrics_file_path.open("r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                # Reconstruct deque and defaultdicts
                for key, data in loaded_data.items():
                    self.metrics[key]["total_calls"] = data.get("total_calls", 0)
                    self.metrics[key]["successful_calls"] = data.get("successful_calls", 0)
                    self.metrics[key]["failed_calls"] = data.get("failed_calls", 0)
                    self.metrics[key]["total_latency_ms"] = data.get("total_latency_ms", 0)
                    # For deque, ensure maxlen is consistent with current config
                    deque_maxlen = self.config.get("monitoring.performance.latency_window_size", 100)
                    self.metrics[key]["latencies_ms"] = deque(data.get("latencies_ms", []), maxlen=deque_maxlen)
                    self.metrics[key]["error_types"] = defaultdict(int, data.get("error_types", {}))
                    self.metrics[key]["total_prompt_tokens"] = data.get("total_prompt_tokens", 0)
                    self.metrics[key]["total_completion_tokens"] = data.get("total_completion_tokens", 0)
                    self.metrics[key]["total_cost"] = data.get("total_cost", 0.0)
                    self.metrics[key]["last_called_at"] = data.get("last_called_at", 0.0)
                    self.metrics[key]["first_called_at"] = data.get("first_called_at", 0.0)

                logger.info(f"Performance metrics loaded from {self.metrics_file_path}")
            except Exception as e:
                logger.error(f"Error loading performance metrics from {self.metrics_file_path}: {e}. Starting fresh.", exc_info=True)
                self.metrics.clear() # Start fresh if loading fails
        else:
            logger.info(f"Performance metrics file not found at {self.metrics_file_path}. Starting with fresh metrics.")

    async def _save_metrics_async(self): # pragma: no cover
        async with self._save_lock:
            try:
                self.metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
                # Convert deques to lists for JSON serialization
                data_to_save = {}
                for key, metrics_data in self.metrics.items():
                    data_to_save[key] = metrics_data.copy() # Make a copy to modify
                    data_to_save[key]["latencies_ms"] = list(metrics_data["latencies_ms"])
                    data_to_save[key]["error_types"] = dict(metrics_data["error_types"]) # Convert defaultdict

                with self.metrics_file_path.open("w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, indent=2)
                logger.info(f"Performance metrics saved to {self.metrics_file_path}")
            except Exception as e:
                logger.error(f"Error saving performance metrics to {self.metrics_file_path}: {e}", exc_info=True)

    def log_llm_call(self,
                     model_name: str,
                     task_type: str, # e.g., "planning", "code_generation", "analysis"
                     initiating_agent_id: str,
                     latency_ms: float,
                     success: bool,
                     prompt_tokens: Optional[int] = None,
                     completion_tokens: Optional[int] = None,
                     cost: Optional[float] = None,
                     error_type: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None): # pragma: no cover
        
        metric_key = f"{model_name}|{task_type}|{initiating_agent_id}"
        now = time.time()

        entry = self.metrics[metric_key] # defaultdict creates if not exists
        if entry["total_calls"] == 0: # First call for this key
            entry["first_called_at"] = now

        entry["total_calls"] += 1
        entry["total_latency_ms"] += latency_ms
        entry["latencies_ms"].append(latency_ms)
        entry["last_called_at"] = now

        if success:
            entry["successful_calls"] += 1
        else:
            entry["failed_calls"] += 1
            if error_type:
                entry["error_types"][error_type] += 1
        
        if prompt_tokens is not None: entry["total_prompt_tokens"] += prompt_tokens
        if completion_tokens is not None: entry["total_completion_tokens"] += completion_tokens
        if cost is not None: entry["total_cost"] += cost
        
        self._request_counter_since_last_save += 1
        if not self.save_periodically and self._request_counter_since_last_save >= self.save_on_request_count:
            asyncio.create_task(self._save_metrics_async()) # Fire and forget save
            self._request_counter_since_last_save = 0

    def get_metrics_for_key(self, metric_key: str) -> Optional[Dict[str, Any]]: # pragma: no cover
        data = self.metrics.get(metric_key)
        if not data: return None
        
        summary = data.copy()
        summary["latencies_ms"] = list(data["latencies_ms"]) # Convert deque for easier use
        summary["error_types"] = dict(data["error_types"])
        if data["total_calls"] > 0:
            summary["avg_latency_ms"] = data["total_latency_ms"] / data["total_calls"]
            summary["success_rate"] = data["successful_calls"] / data["total_calls"]
        else:
            summary["avg_latency_ms"] = 0
            summary["success_rate"] = 1.0 # Or None
        return summary

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]: # pragma: no cover
        """Returns a summary of all tracked metrics."""
        all_summaries = {}
        for key in list(self.metrics.keys()): # Iterate over copy of keys
            summary = self.get_metrics_for_key(key)
            if summary:
                all_summaries[key] = summary
        return all_summaries

    # ADDED THIS METHOD
    def get_summary_metrics(self) -> Dict[str, Any]: # pragma: no cover
        """Calculates overall summary statistics across all tracked keys."""
        total_calls_overall = 0
        total_successful_calls_overall = 0
        total_failed_calls_overall = 0
        total_latency_ms_overall = 0
        # More complex aggregations like avg_prompt_tokens would need to sum tokens / sum calls
        # For now, just high-level call stats

        for key_metrics in self.metrics.values():
            total_calls_overall += key_metrics.get("total_calls", 0)
            total_successful_calls_overall += key_metrics.get("successful_calls", 0)
            total_failed_calls_overall += key_metrics.get("failed_calls", 0)
            total_latency_ms_overall += key_metrics.get("total_latency_ms", 0)
        
        avg_latency_overall_ms = (total_latency_ms_overall / total_calls_overall) if total_calls_overall > 0 else 0
        success_rate_overall = (total_successful_calls_overall / total_calls_overall) if total_calls_overall > 0 else 1.0

        return {
            "total_distinct_metric_keys": len(self.metrics),
            "total_calls": total_calls_overall,
            "total_successful_calls": total_successful_calls_overall,
            "total_failed_calls": total_failed_calls_overall,
            "overall_success_rate": success_rate_overall,
            "overall_avg_latency_ms": avg_latency_overall_ms
        }

    def start_periodic_save(self): # pragma: no cover
        if self._periodic_save_task and not self._periodic_save_task.done():
            logger.info("Periodic metrics saver already running.")
            return
        if self.periodic_save_interval_seconds > 0:
            self.save_periodically = True # Ensure flag is set
            self._periodic_save_task = asyncio.create_task(self._periodic_save_worker())
        else:
            logger.warning("Periodic save interval is not positive. Not starting periodic saver.")


    def stop_periodic_save(self): # pragma: no cover
        self.save_periodically = False # Signal to stop, though task cancellation is more direct
        if self._periodic_save_task and not self._periodic_save_task.done():
            self._periodic_save_task.cancel()
            logger.info("Periodic metrics saver cancellation requested.")
        self._periodic_save_task = None

    async def _periodic_save_worker(self): # pragma: no cover
        logger.info(f"Periodic performance metrics saver started. Interval: {self.periodic_save_interval_seconds}s")
        while self.save_periodically: # Check flag in loop
            try:
                await asyncio.sleep(self.periodic_save_interval_seconds)
                if not self.save_periodically: break # Check flag again after sleep
                await self._save_metrics_async()
                self._request_counter_since_last_save = 0 # Reset counter after periodic save
            except asyncio.CancelledError:
                logger.info("Periodic metrics saver task cancelled.")
                break
            except Exception as e: # pragma: no cover
                logger.error(f"Error in periodic metrics saver: {e}", exc_info=True)
                # Potentially increase sleep interval on error to avoid rapid error logging
                await asyncio.sleep(self.periodic_save_interval_seconds * 2) 
        logger.info("Periodic metrics saver task stopped.")


    async def shutdown(self): # pragma: no cover
        logger.info("PerformanceMonitor shutting down...")
        self.stop_periodic_save() # Request stop
        if self._periodic_save_task:
            try:
                await asyncio.wait_for(self._periodic_save_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError): # pragma: no cover
                logger.warning("Periodic save task did not shut down cleanly within timeout.")
            except Exception as e_shutdown: # pragma: no cover
                 logger.error(f"Exception during periodic save task shutdown: {e_shutdown}")

        # Final save before exiting
        if self._request_counter_since_last_save > 0 or not self.metrics_file_path.exists(): # Save if there are unsaved metrics or if file never created
            logger.info("Performing final save of performance metrics...")
            await self._save_metrics_async()
        logger.info("PerformanceMonitor shutdown complete.")

    @classmethod
    async def reset_instance_async(cls): # pragma: no cover
        async with cls._lock:
            if cls._instance:
                await cls._instance.shutdown()
                cls._instance._initialized = False
                cls._instance = None
        logger.debug("PerformanceMonitor instance reset asynchronously.")


async def get_performance_monitor_async(config_override: Optional[Config] = None, test_mode: bool = False) -> PerformanceMonitor: # pragma: no cover
    if not PerformanceMonitor._instance or test_mode:
        async with PerformanceMonitor._lock:
            if PerformanceMonitor._instance is None or test_mode:
                if test_mode and PerformanceMonitor._instance is not None:
                     await PerformanceMonitor._instance.shutdown() # Ensure old tasks are cleaned up
                     PerformanceMonitor._instance = None
                PerformanceMonitor._instance = PerformanceMonitor(config_override=config_override, test_mode=test_mode)
    return PerformanceMonitor._instance

def get_performance_monitor(config_override: Optional[Config] = None, test_mode: bool = False) -> PerformanceMonitor: # pragma: no cover
    if PerformanceMonitor._instance is None or test_mode:
        if test_mode and PerformanceMonitor._instance is not None:
             # Sync shutdown of async tasks is tricky; best effort
            logger.warning("Test mode reset for PerformanceMonitor in sync getter: async task shutdown not guaranteed.")
            if PerformanceMonitor._instance._periodic_save_task and not PerformanceMonitor._instance._periodic_save_task.done():
                 PerformanceMonitor._instance._periodic_save_task.cancel()
            PerformanceMonitor._instance = None
        PerformanceMonitor._instance = PerformanceMonitor(config_override=config_override, test_mode=test_mode)
    return PerformanceMonitor._instance
