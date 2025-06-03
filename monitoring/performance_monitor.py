# mindx/monitoring/performance_monitor.py
import os
import time
import logging # Standard logging
import asyncio
import json
from typing import Dict, Any, Optional, List, Tuple, Union, Coroutine
from collections import defaultdict
from pathlib import Path
import heapq # For efficient top N latencies / percentile approximations (if using that method)
import statistics # For more accurate median, percentiles if full sample is kept

# Use canonical PROJECT_ROOT from config
from mindx.utils.config import Config, PROJECT_ROOT
from mindx.utils.logging_config import get_logger

logger = get_logger(__name__)

# Metric Key: Can be just model_id (str) or a tuple (model_id, task_type_str, ?initiating_agent_id_str)
# For simplicity in JSON serialization, we'll serialize tuple keys to "val1|val2|val3"
METRIC_KEY_INTERNAL_TYPE = Union[str, Tuple[str, ...]]
METRIC_KEY_SERIALIZED_TYPE = str


class PerformanceMonitor:
    """
    Monitors LLM performance metrics such as latency, token counts, cost, success rates,
    and error types. Supports contextual tracking (e.g., per model, per task type).
    Metrics are persisted to a JSON file. Implemented as a singleton.
    """
    _instance = None
    _lock = asyncio.Lock() # Class-level lock for singleton creation and async file operations

    def __new__(cls, *args, **kwargs): # pragma: no cover
        if not cls._instance:
            cls._instance = super(PerformanceMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_override: Optional[Config] = None, test_mode: bool = False): # pragma: no cover
        if hasattr(self, '_initialized') and self._initialized and not test_mode:
            return
        
        self.config = config_override or Config()
        
        # Metrics structure: key -> {counts, totals, recent_latencies_ms_samples}
        # Key can be model_id string, or tuple like (model_id, task_type)
        self.metrics: Dict[METRIC_KEY_INTERNAL_TYPE, Dict[str, Any]] = defaultdict(
            lambda: { # Default factory for a new metric key
                "requests": 0, 
                "successes": 0, 
                "failures": 0,
                "total_latency_ms": 0,    # Store in ms for precision
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,    # In USD
                "error_types": defaultdict(int), # Counts specific error types
                # Stores the N most recent latencies for approximate percentiles
                "recent_latencies_ms_samples": [], 
                "requests_since_last_save": 0 # For batched saving by count
            }
        )
        
        self.metrics_file_path: Path = Path(self.config.get(
            "monitoring.performance.metrics_file_path",
            str(PROJECT_ROOT / "data" / "performance_metrics.json") # Default path
        ))
        self.save_every_n_requests: int = self.config.get("monitoring.performance.save_every_n_requests", 20)
        self.max_recent_latencies_samples: int = self.config.get("monitoring.performance.max_recent_latencies_samples", 100)
        self.persistence_enabled: bool = self.config.get("monitoring.performance.persistence_enabled", True)

        if self.persistence_enabled:
            self._load_metrics_sync() # Load synchronously during init for simplicity
        
        self.periodic_save_task: Optional[asyncio.Task] = None
        if self.persistence_enabled and self.config.get("monitoring.performance.enable_periodic_save", True) and not test_mode: # pragma: no cover
            interval = self.config.get("monitoring.performance.periodic_save_interval_seconds", 300.0) # Default 5 mins
            if interval > 0:
                self.periodic_save_task = asyncio.create_task(self._periodic_save_worker(interval))
            else: # pragma: no cover
                logger.warning("PerformanceMonitor: Periodic save is enabled but interval is <= 0. Disabling periodic save.")


        logger.info(
            f"PerformanceMonitor initialized. Metrics File: {self.metrics_file_path if self.persistence_enabled else 'Persistence Disabled'}. "
            f"Save every: {self.save_every_n_requests} reqs (if periodic save off). "
            f"Periodic Save: {'Enabled, interval ' + str(interval) + 's' if self.periodic_save_task else 'Disabled'}."
        )
        self._initialized = True
    
    def _serialize_metric_key(self, key: METRIC_KEY_INTERNAL_TYPE) -> METRIC_KEY_SERIALIZED_TYPE: # pragma: no cover
        """Converts internal metric key (str or tuple) to a string for JSON keys."""
        if isinstance(key, tuple):
            # Ensure all parts of the tuple are strings before joining
            return "|".join(str(k_part) for k_part in key)
        return str(key) # Ensure even single keys are strings

    def _deserialize_metric_key(self, key_str: METRIC_KEY_SERIALIZED_TYPE) -> METRIC_KEY_INTERNAL_TYPE: # pragma: no cover
        """Converts a string key from JSON back to internal tuple or str."""
        parts = key_str.split("|")
        # Attempt to convert parts back to original types if possible (e.g. int for counts if that was part of key)
        # For now, assume all parts of a key were strings or can be treated as strings.
        return tuple(parts) if len(parts) > 1 else parts[0]

    def _load_metrics_sync(self): # pragma: no cover
        """Synchronously loads metrics from file. Called during __init__."""
        if not self.metrics_file_path.exists():
            logger.info(f"Performance metrics file not found at {self.metrics_file_path}. Starting with fresh metrics.")
            return
        try:
            with self.metrics_file_path.open("r", encoding="utf-8") as f:
                loaded_data = json.load(f)
            
            temp_metrics_load: Dict[METRIC_KEY_INTERNAL_TYPE, Dict[str, Any]] = {}
            for key_str, metrics_data_loaded in loaded_data.items():
                internal_key = self._deserialize_metric_key(key_str)
                # Initialize with defaultdict factory structure then update
                # This ensures that if new fields are added to the default factory,
                # loaded data still gets them with default values if missing.
                default_entry = self.metrics.default_factory() # Get a new default entry
                
                error_types_data = metrics_data_loaded.get("error_types", {})
                default_entry.update(metrics_data_loaded) # Overwrite with loaded simple values
                default_entry["error_types"] = defaultdict(int, error_types_data) # Ensure it's defaultdict
                default_entry["recent_latencies_ms_samples"] = metrics_data_loaded.get("recent_latencies_ms_samples", [])
                default_entry["requests_since_last_save"] = metrics_data_loaded.get("requests_since_last_save", 0)
                
                temp_metrics_load[internal_key] = default_entry
            
            self.metrics.update(temp_metrics_load) # Update the main defaultdict
            logger.info(f"Loaded performance metrics for {len(self.metrics)} keys from {self.metrics_file_path}")

        except json.JSONDecodeError as e: # pragma: no cover
            logger.error(f"Error decoding JSON from performance metrics file {self.metrics_file_path}: {e}. Starting with fresh metrics.")
            self.metrics.clear()
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error loading performance metrics: {e}", exc_info=True)
            self.metrics.clear()

    async def _save_metrics_async(self): # pragma: no cover
        """Asynchronously saves current metrics to the configured file."""
        if not self.persistence_enabled: return

        async with self._lock: # Ensure only one save operation at a time
            try:
                self.metrics_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                data_to_save = {}
                # Iterate over a copy of items in case metrics are modified by another task (though lock should prevent)
                for key, value_dict in list(self.metrics.items()):
                    serialized_key = self._serialize_metric_key(key)
                    # Ensure error_types is a plain dict for JSON serialization
                    # and other potentially complex types are converted if necessary
                    saveable_value_dict = {
                        k: (dict(v) if isinstance(v, defaultdict) else v)
                        for k, v in value_dict.items()
                    }
                    data_to_save[serialized_key] = saveable_value_dict

                content_to_save = json.dumps(data_to_save, indent=2)
                
                loop = asyncio.get_running_loop()
                # Use run_in_executor for synchronous file I/O to avoid blocking event loop
                await loop.run_in_executor(None, self.metrics_file_path.write_text, content_to_save, "utf-8")
                
                logger.debug(f"Saved performance metrics for {len(self.metrics)} keys to {self.metrics_file_path}")
                # Reset requests_since_last_save for all entries after successful save
                for key_metrics in self.metrics.values():
                    key_metrics["requests_since_last_save"] = 0
            except Exception as e:
                logger.error(f"Error saving performance metrics to {self.metrics_file_path}: {e}", exc_info=True)

    async def _periodic_save_worker(self, interval_seconds: float): # pragma: no cover
        """Periodically saves metrics if persistence is enabled."""
        logger.info(f"Periodic performance metrics saver started. Interval: {interval_seconds}s")
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                if self.metrics: 
                    logger.info("Periodic saver: Saving all performance metrics...")
                    await self._save_metrics_async()
            except asyncio.CancelledError:
                logger.info("Periodic metrics saver task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in periodic metrics saver: {e}", exc_info=True)
                # Avoid fast loop on persistent error, but continue trying
                await asyncio.sleep(max(interval_seconds, 60)) # Min 1 min sleep on error
        logger.info("Periodic metrics saver task stopped.")


    async def record_request(self, model_id: str, success: bool, latency_seconds: float,
                             input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
                             task_type: Optional[str] = None, 
                             initiating_agent_id: Optional[str] = None,
                             error_type: Optional[str] = None): # pragma: no cover
        """
        Records a performance metric for an LLM request. Async due to potential save.
        """
        key_parts = [model_id]
        if task_type: key_parts.append(task_type.lower().strip().replace(" ", "_")) # Normalize
        if initiating_agent_id: key_parts.append(initiating_agent_id)
        
        internal_key: METRIC_KEY_INTERNAL_TYPE = tuple(key_parts) if len(key_parts) > 1 else model_id
        latency_ms = int(latency_seconds * 1000)

        should_save_by_count_flag = False
        async with self._lock: # Protect metrics dictionary modification
            entry = self.metrics[internal_key] # defaultdict creates if not exists
            entry["requests"] += 1
            if success: entry["successes"] += 1
            else:
                entry["failures"] += 1
                if error_type: entry["error_types"][error_type] = entry["error_types"].get(error_type, 0) + 1
            
            entry["total_latency_ms"] += latency_ms
            entry["total_input_tokens"] += input_tokens
            entry["total_output_tokens"] += output_tokens
            # Ensure total_cost_usd exists before incrementing
            entry["total_cost_usd"] = entry.get("total_cost_usd", 0.0) + cost_usd
            
            # Maintain N most recent latencies
            entry["recent_latencies_ms_samples"].append(latency_ms)
            if len(entry["recent_latencies_ms_samples"]) > self.max_recent_latencies_samples:
                entry["recent_latencies_ms_samples"].pop(0) # Remove oldest to keep N samples
            
            entry["requests_since_last_save"] = entry.get("requests_since_last_save", 0) + 1
            # Only trigger save-by-count if periodic saving is disabled
            if self.persistence_enabled and \
               not self.config.get("monitoring.performance.enable_periodic_save", True) and \
               entry["requests_since_last_save"] >= self.save_every_n_requests:
                should_save_by_count_flag = True
        
        if should_save_by_count_flag: # pragma: no cover
            await self._save_metrics_async() # This will re-acquire lock internally

        key_str = self._serialize_metric_key(internal_key)
        logger.debug(
            f"PerfRec: Key='{key_str}', Success={success}, Latency={latency_seconds:.3f}s ({latency_ms}ms), "
            f"InTok={input_tokens}, OutTok={output_tokens}, Cost=${cost_usd:.6f}, ErrType={error_type or 'N/A'}"
        )

    def get_metrics_for_key(self, key: METRIC_KEY_INTERNAL_TYPE) -> Dict[str, Any]: # pragma: no cover
        """Calculates and returns derived metrics for a given internal key. This method is synchronous."""
        # Reading self.metrics should be safe without lock if writes are protected,
        # but for full safety during complex dict operations, a read lock could be used if concerned.
        # For now, assuming Python's dict operations are mostly atomic for reads.
        raw_metrics = self.metrics.get(key) # Use .get() for safety if key might not exist (though defaultdict handles)
        if not raw_metrics or raw_metrics.get("requests", 0) == 0:
            # Return a structure consistent with a populated entry for easier consumption
            return {
                "key_str": self._serialize_metric_key(key), "requests": 0, "successes": 0, "failures": 0,
                "success_rate": 0.0, "avg_latency_ms": 0.0, 
                "p50_latency_ms":0.0, "p90_latency_ms":0.0, "p99_latency_ms":0.0,
                "avg_input_tokens":0.0, "avg_output_tokens":0.0, "avg_total_tokens":0.0, 
                "avg_cost_usd":0.0, "total_cost_usd":0.0, "error_types":{}
            }

        reqs = raw_metrics["requests"]
        
        latencies_sample = sorted(raw_metrics.get("recent_latencies_ms_samples", []))
        p50 = statistics.median(latencies_sample) if latencies_sample else 0.0
        # statistics.quantiles needs at least 2 data points for n > 1
        p90 = statistics.quantiles(latencies_sample, n=10)[8] if len(latencies_sample) >= 2 else (latencies_sample[0] if len(latencies_sample) == 1 else 0.0)
        p99 = statistics.quantiles(latencies_sample, n=100)[98] if len(latencies_sample) >= 2 else (latencies_sample[0] if len(latencies_sample) == 1 else 0.0)

        return {
            "key_str": self._serialize_metric_key(key),
            "requests": reqs, "successes": raw_metrics["successes"], "failures": raw_metrics["failures"],
            "success_rate": (raw_metrics["successes"] / reqs) if reqs > 0 else 0.0,
            "avg_latency_ms": (raw_metrics["total_latency_ms"] / reqs) if reqs > 0 else 0.0,
            "p50_latency_ms": p50, "p90_latency_ms": p90, "p99_latency_ms": p99,
            "avg_input_tokens": (raw_metrics["total_input_tokens"] / reqs) if reqs > 0 else 0.0,
            "avg_output_tokens": (raw_metrics["total_output_tokens"] / reqs) if reqs > 0 else 0.0,
            "avg_total_tokens": ((raw_metrics.get("total_input_tokens",0) + raw_metrics.get("total_output_tokens",0)) / reqs) if reqs > 0 else 0.0,
            "avg_cost_usd": (raw_metrics.get("total_cost_usd",0.0) / reqs) if reqs > 0 else 0.0,
            "total_cost_usd": raw_metrics.get("total_cost_usd",0.0),
            "error_types": dict(raw_metrics.get("error_types", {})) # Convert defaultdict for output
        }

    def get_all_metrics(self) -> Dict[METRIC_KEY_SERIALIZED_TYPE, Dict[str, Any]]: # pragma: no cover
        """Returns derived metrics for all tracked keys, with keys serialized for output."""
        # Create a snapshot of keys to iterate over, in case metrics dict changes during iteration by another task
        # (though get_metrics_for_key is read-only on shared data)
        keys_snapshot = list(self.metrics.keys())
        return {self._serialize_metric_key(key): self.get_metrics_for_key(key) for key in keys_snapshot}
    
    def get_performance_report(self) -> str: # pragma: no cover
        """Generates a human-readable performance report string."""
        report_parts = ["# LLM Performance Report (MindX - Augmentic)\n"]
        all_m = self.get_all_metrics()
        if not all_m:
            report_parts.append("No performance data recorded yet.\n")
            return "".join(report_parts)
        
        report_parts.append(f"Tracking {len(all_m)} metric keys (model or model/context combinations).\n")
        
        total_requests_all = sum(m.get("requests",0) for m in all_m.values())
        total_successes_all = sum(m.get("successes",0) for m in all_m.values())
        total_cost_all = sum(m.get("total_cost_usd",0.0) for m in all_m.values())

        if total_requests_all > 0:
            overall_sr_all = (total_successes_all / total_requests_all) * 100
            report_parts.append(f"\n## Overall Summary across all keys\n- Total Requests: {total_requests_all:,}\n- Overall Success Rate: {overall_sr_all:.2f}%\n- Total Estimated Cost: ${total_cost_all:.4f}\n")

        report_parts.append("\n## Detailed Metrics per Key\n")
        # Sort by number of requests (desc) then by key_str (asc) for consistent reporting
        sorted_metric_items = sorted(all_m.items(), key=lambda item: (-item[1].get('requests',0), item[0]))

        for key_str, metrics in sorted_metric_items:
            report_parts.append(f"### Metrics for: {key_str}\n")
            report_parts.append(f"- Requests: {metrics['requests']:,}\n")
            report_parts.append(f"- Success Rate: {metrics['success_rate']:.2%}\n")
            avg_lat_s = metrics['avg_latency_ms']/1000.0
            p50_s = metrics['p50_latency_ms']/1000.0; p90_s = metrics['p90_latency_ms']/1000.0; p99_s = metrics['p99_latency_ms']/1000.0
            report_parts.append(f"- Avg Latency: {avg_lat_s:.3f}s (P50: {p50_s:.3f}s, P90: {p90_s:.3f}s, P99: {p99_s:.3f}s)\n")
            report_parts.append(f"- Avg Tokens (In/Out/Total): {metrics['avg_input_tokens']:.0f} / {metrics['avg_output_tokens']:.0f} / {metrics['avg_total_tokens']:.0f}\n")
            report_parts.append(f"- Avg Cost: ${metrics['avg_cost_usd']:.6f} (Total Cost for this key: ${metrics['total_cost_usd']:.4f})\n")
            if metrics.get("error_types"):
                error_types_str = ", ".join([f"{k}: {v}" for k,v in sorted(metrics['error_types'].items())])
                report_parts.append(f"- Error Types: {error_types_str}\n")
            report_parts.append("\n")
        return "".join(report_parts)

    async def reset_metrics(self, key_prefix_serialized: Optional[str] = None): # pragma: no cover
        """Resets metrics. If key_prefix_serialized is given, resets keys starting with it. Async due to save."""
        async with self._lock: # Ensure consistency during reset
            if key_prefix_serialized:
                keys_to_reset = [k_internal for k_internal in list(self.metrics.keys()) 
                                 if self._serialize_metric_key(k_internal).startswith(key_prefix_serialized)]
                for k_to_reset in keys_to_reset:
                    del self.metrics[k_to_reset]
                logger.info(f"Reset metrics for keys starting with '{key_prefix_serialized}'. {len(keys_to_reset)} keys removed.")
            else:
                self.metrics.clear(); logger.info("Reset all performance metrics.")
        
        if self.persistence_enabled: await self._save_metrics_async() # Save after reset

    async def shutdown(self): # pragma: no cover
        """Shuts down the performance monitor, ensuring final metrics save and task cancellation."""
        logger.info("PerformanceMonitor shutting down...")
        if self.periodic_save_task and not self.periodic_save_task.done():
            self.periodic_save_task.cancel()
            try:
                await asyncio.wait_for(self.periodic_save_task, timeout=2.0) # Give it time to cancel
            except asyncio.CancelledError:
                logger.info("Periodic save task successfully cancelled during shutdown.")
            except asyncio.TimeoutError: # pragma: no cover
                logger.warning("Timeout waiting for periodic save task to cancel during shutdown.")
            except Exception as e: # pragma: no cover
                 logger.error(f"Error awaiting periodic save task during shutdown: {e}", exc_info=True)
        
        if self.persistence_enabled:
            await self._save_metrics_async() # Final save attempt
        logger.info("PerformanceMonitor shutdown complete.")
    
    @classmethod
    async def reset_instance_async(cls): # For testing # pragma: no cover
        """Asynchronously resets the singleton instance. Ensures monitoring task is stopped."""
        async with cls._lock:
            if cls._instance:
                await cls._instance.shutdown() # Call proper shutdown which handles tasks and saving
                cls._instance._initialized = False # Allow re-init
                cls._instance = None
        logger.debug("PerformanceMonitor instance reset asynchronously.")

# Asynchronous factory/getter for the singleton
async def get_performance_monitor_async(config_override: Optional[Config] = None, test_mode: bool = False) -> PerformanceMonitor: # pragma: no cover
    """Asynchronously gets or creates the PerformanceMonitor singleton instance."""
    if not PerformanceMonitor._instance or test_mode:
        async with PerformanceMonitor._lock: # Class lock for safe singleton creation
            if PerformanceMonitor._instance is None or test_mode:
                if test_mode and PerformanceMonitor._instance is not None:
                    await PerformanceMonitor._instance.shutdown() # Ensure old test instance is shut down
                    PerformanceMonitor._instance = None # Force re-creation
                PerformanceMonitor._instance = PerformanceMonitor(config_override=config_override, test_mode=test_mode)
    return PerformanceMonitor._instance

# Synchronous getter
def get_performance_monitor(config_override: Optional[Config] = None, test_mode: bool = False) -> PerformanceMonitor: # pragma: no cover
    """Synchronously gets or creates the PerformanceMonitor singleton instance."""
    if PerformanceMonitor._instance is None or test_mode:
        if test_mode and PerformanceMonitor._instance is not None:
            # For sync reset, a full async shutdown of old test instance is hard without a loop.
            # Rely on test_mode in __init__ to not start async tasks for the new instance if that's desired.
            # The async reset_instance_async is better if a loop is available.
            PerformanceMonitor._instance = None
        PerformanceMonitor._instance = PerformanceMonitor(config_override=config_override, test_mode=test_mode)
    return PerformanceMonitor._instance
