# monitoring/resource_monitor.py
import os
import time
# import logging # Use get_logger
import asyncio
import psutil # Requires pip install psutil
from typing import Dict, Any, Optional, List, Callable, Union, Coroutine, Tuple # Added Tuple
from pathlib import Path
from enum import Enum
import stat # For file permissions

# Import from sibling top-level package
from utils.config import Config, PROJECT_ROOT
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ResourceType(Enum): # pragma: no cover
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"

class ResourceMonitor:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs): # pragma: no cover
        if not cls._instance:
            cls._instance = super(ResourceMonitor, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_override: Optional[Config] = None, test_mode: bool = False): # pragma: no cover
        if hasattr(self, '_initialized') and self._initialized and not test_mode:
            return

        self.config = config_override or Config()

        self.max_cpu_percent: float = self.config.get("monitoring.resource.max_cpu_percent", 85.0)
        self.max_memory_percent: float = self.config.get("monitoring.resource.max_memory_percent", 85.0)

        self.disk_threshold_map: Dict[str, float] = {}
        self._configure_disk_thresholds()

        self.cpu_usage: float = 0.0
        self.memory_usage: float = 0.0
        self.disk_usage_map: Dict[str, float] = {path: 0.0 for path in self.disk_threshold_map.keys()}

        # Use forward reference 'ResourceMonitor' for type hints within the class
        self.alert_callbacks: List[Callable[['ResourceMonitor', ResourceType, float, Optional[str]], Coroutine[Any, Any, None]]] = []
        self.resolve_callbacks: List[Callable[['ResourceMonitor', ResourceType, float, Optional[str]], Coroutine[Any, Any, None]]] = []

        self.monitoring: bool = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.monitoring_interval: float = self.config.get("monitoring.resource.interval", 10.0)

        self._alert_active_flags: Dict[Union[ResourceType, Tuple[ResourceType, str]], bool] = {}
        self._last_alert_timestamp: Dict[Union[ResourceType, Tuple[ResourceType, str]], float] = {}
        self.re_alert_interval_seconds: float = self.config.get("monitoring.resource.re_alert_interval_seconds", 300.0)

        logger.info(
            f"ResourceMonitor initialized. CPU Limit: {self.max_cpu_percent}%, Memory Limit: {self.max_memory_percent}%, "
            f"Disk Thresholds: {self.disk_threshold_map}, Re-alert Interval: {self.re_alert_interval_seconds}s"
        )
        self._initialized = True

    def _configure_disk_thresholds(self): # pragma: no cover
        disk_configs_raw = self.config.get("monitoring.resource.disk_paths", [{"path": "/", "threshold": 90.0}])
        default_disk_threshold = self.config.get("monitoring.resource.max_disk_percent", 90.0)

        disk_configs_list: List[Any]
        if isinstance(disk_configs_raw, list):
            disk_configs_list = disk_configs_raw
        elif isinstance(disk_configs_raw, str):
            disk_configs_list = [{"path": disk_configs_raw, "threshold": default_disk_threshold}] # Ensure dict format
        else:
            logger.warning(f"Invalid 'monitoring.resource.disk_paths' config format: {type(disk_configs_raw)}. Defaulting to root path '/' only.")
            disk_configs_list = [{"path": "/"}]

        for item in disk_configs_list:
            path_str: Optional[str] = None
            threshold: Optional[float] = None
            if isinstance(item, str): # Handle simple string list if accidentally configured that way
                path_str = item
                threshold = default_disk_threshold
            elif isinstance(item, dict) and "path" in item:
                path_str = item["path"]
                try:
                    threshold = float(item.get("threshold", default_disk_threshold))
                    if not (0.0 < threshold <= 100.0): # pragma: no cover
                        logger.warning(f"Disk threshold for '{path_str}' ({threshold}%) out of (0, 100] range. Using default {default_disk_threshold}%.")
                        threshold = default_disk_threshold
                except ValueError: # pragma: no cover
                    logger.warning(f"Invalid threshold value for disk path '{path_str}': {item.get('threshold')}. Using default.")
                    threshold = default_disk_threshold
            
            if path_str and threshold is not None:
                try:
                    resolved_path = Path(path_str).resolve(strict=False)
                    norm_path_str = str(resolved_path)
                    if not resolved_path.exists() and os.path.ismount(path_str): # pragma: no cover
                         logger.info(f"Configured disk path '{path_str}' (mount point) does not strictly exist yet, but will be monitored.")
                    elif not resolved_path.exists() and not os.path.ismount(path_str): # Also check if it's not a mount
                         logger.warning(f"Configured disk path '{path_str}' resolved to '{norm_path_str}' which does not exist and is not a mount point. Monitoring may fail for this path.")
                    
                    self.disk_threshold_map[norm_path_str] = threshold
                except Exception as e: # pragma: no cover
                    logger.error(f"Invalid or inaccessible disk path configured '{path_str}': {e}. Skipping.")
            elif item:
                logger.warning(f"Malformed disk config item: {item}. Skipping.")
        
        if not self.disk_threshold_map:
             self.disk_threshold_map[str(Path("/").resolve())] = default_disk_threshold
             logger.warning("No valid disk paths configured or all failed. Defaulting to monitor root path '/' only.")


    def start_monitoring(self, interval: Optional[float] = None): # pragma: no cover
        if self.monitoring:
            logger.warning("Resource monitoring is already running.")
            return
        
        eff_interval = interval if interval is not None else self.monitoring_interval
        if eff_interval <= 0: # pragma: no cover
            logger.error(f"Invalid monitoring interval: {eff_interval}. Must be positive. Not starting.")
            return

        self.monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitor_resources_loop(eff_interval))
        logger.info(f"Started resource monitoring. Interval: {eff_interval}s")
    
    def stop_monitoring(self): # pragma: no cover
        if not self.monitoring:
            logger.info("Resource monitoring is not running or already stopped.")
            return
        
        self.monitoring = False
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            logger.info("Resource monitoring cancellation requested.")
        self.monitoring_task = None
    
    async def _monitor_resources_loop(self, interval: float): # pragma: no cover
        logger.info("Resource monitoring loop started.")
        try:
            psutil.cpu_percent(interval=0.1)
        except Exception as e: # pragma: no cover
            logger.error(f"Failed initial psutil.cpu_percent call during monitor startup: {e}")

        while self.monitoring:
            start_check_time = time.monotonic()
            try:
                await self.update_current_usage()
                
                disk_usage_str_parts = []
                for p, u in self.disk_usage_map.items():
                    try: p_display = Path(p).name # Fails if p is not a valid path string, e.g. empty
                    except: p_display = p[:15] # Fallback for display
                    disk_usage_str_parts.append(f"{p_display}:{u if u >= 0 else 'ERR'}%")
                disk_usage_str = ", ".join(disk_usage_str_parts)
                logger.debug(f"Res Usage: CPU {self.cpu_usage:.1f}%, Mem {self.memory_usage:.1f}%, Disks [{disk_usage_str}]")
                
                await self._check_and_trigger_alerts()
                
                elapsed_check_time = time.monotonic() - start_check_time
                sleep_duration = max(0.1, interval - elapsed_check_time)
                await asyncio.sleep(sleep_duration)

            except asyncio.CancelledError: # pragma: no cover
                logger.info("Resource monitoring loop gracefully cancelled.")
                break
            except Exception as e: # pragma: no cover
                logger.error(f"Error in resource_monitor loop: {e}", exc_info=True)
                await asyncio.sleep(interval * 2)
        logger.info("Resource monitoring loop finished.")

    async def update_current_usage(self): # pragma: no cover
        try:
            loop = asyncio.get_running_loop()
            
            self.cpu_usage = await loop.run_in_executor(None, psutil.cpu_percent, None)
            
            mem_info = await loop.run_in_executor(None, psutil.virtual_memory)
            self.memory_usage = mem_info.percent
            
            disk_paths_to_check = list(self.disk_threshold_map.keys())
            for path_str in disk_paths_to_check:
                try:
                    disk_info = await loop.run_in_executor(None, psutil.disk_usage, path_str)
                    self.disk_usage_map[path_str] = disk_info.percent
                except FileNotFoundError: # pragma: no cover
                    logger.warning(f"Disk path '{path_str}' no longer found. Monitoring for it disabled.")
                    self.disk_usage_map.pop(path_str, None)
                    self.disk_threshold_map.pop(path_str, None)
                    self._alert_active_flags.pop((ResourceType.DISK, path_str), None)
                except Exception as e_disk: # pragma: no cover
                    logger.error(f"Error getting disk usage for '{path_str}': {e_disk}")
                    self.disk_usage_map[path_str] = -1.0
        except Exception as e_update: # pragma: no cover
             logger.error(f"Failed to update one or more resource usage metrics: {e_update}", exc_info=True)


    async def _check_and_trigger_alerts(self): # pragma: no cover
        now = time.time()
        
        cpu_key = ResourceType.CPU
        if self.cpu_usage > self.max_cpu_percent:
            if not self._alert_active_flags.get(cpu_key) or \
               (now - self._last_alert_timestamp.get(cpu_key, 0) > self.re_alert_interval_seconds):
                logger.warning(f"ALERT: CPU usage ({self.cpu_usage:.1f}%) > limit ({self.max_cpu_percent}%)")
                await self._execute_callbacks(self.alert_callbacks, self, cpu_key, self.cpu_usage, None)
                self._alert_active_flags[cpu_key] = True
                self._last_alert_timestamp[cpu_key] = now
        elif self._alert_active_flags.get(cpu_key):
             logger.info(f"RESOLVED: CPU usage ({self.cpu_usage:.1f}%) back below limit ({self.max_cpu_percent}%).")
             await self._execute_callbacks(self.resolve_callbacks, self, cpu_key, self.cpu_usage, None)
             self._alert_active_flags[cpu_key] = False

        mem_key = ResourceType.MEMORY
        if self.memory_usage > self.max_memory_percent:
            if not self._alert_active_flags.get(mem_key) or \
               (now - self._last_alert_timestamp.get(mem_key, 0) > self.re_alert_interval_seconds):
                logger.warning(f"ALERT: Memory usage ({self.memory_usage:.1f}%) > limit ({self.max_memory_percent}%)")
                await self._execute_callbacks(self.alert_callbacks, self, mem_key, self.memory_usage, None)
                self._alert_active_flags[mem_key] = True
                self._last_alert_timestamp[mem_key] = now
        elif self._alert_active_flags.get(mem_key):
             logger.info(f"RESOLVED: Memory usage ({self.memory_usage:.1f}%) back below limit ({self.max_memory_percent}%).")
             await self._execute_callbacks(self.resolve_callbacks, self, mem_key, self.memory_usage, None)
             self._alert_active_flags[mem_key] = False

        for path_str, usage in self.disk_usage_map.items():
            if usage < 0: continue
            threshold = self.disk_threshold_map.get(path_str)
            if threshold is None: continue

            disk_key_tuple = (ResourceType.DISK, path_str)
            if usage > threshold:
                if not self._alert_active_flags.get(disk_key_tuple) or \
                   (now - self._last_alert_timestamp.get(disk_key_tuple, 0) > self.re_alert_interval_seconds):
                    logger.warning(f"ALERT: Disk usage for '{path_str}' ({usage:.1f}%) > limit ({threshold}%)")
                    await self._execute_callbacks(self.alert_callbacks, self, ResourceType.DISK, usage, path_str)
                    self._alert_active_flags[disk_key_tuple] = True
                    self._last_alert_timestamp[disk_key_tuple] = now
            elif self._alert_active_flags.get(disk_key_tuple):
                 logger.info(f"RESOLVED: Disk usage for '{path_str}' ({usage:.1f}%) back below limit ({threshold}%).")
                 await self._execute_callbacks(self.resolve_callbacks, self, ResourceType.DISK, usage, path_str)
                 self._alert_active_flags[disk_key_tuple] = False

    async def _execute_callbacks(self,
                                 callbacks: List[Callable[['ResourceMonitor', ResourceType, float, Optional[str]], Coroutine[Any, Any, None]]],
                                 monitor_instance: 'ResourceMonitor', # Use string forward reference
                                 rtype: ResourceType,
                                 value: float,
                                 path: Optional[str] = None): # pragma: no cover
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(monitor_instance, rtype, value, path)
                else: # pragma: no cover
                    logger.warning(f"Callback {getattr(callback, '__name__', 'unknown')} is not an async function, but was registered. Calling it synchronously.")
                    callback(monitor_instance, rtype, value, path)
            except Exception as e_cb:
                logger.error(f"Error in resource monitor callback '{getattr(callback, '__name__', 'unknown_callback')}': {e_cb}", exc_info=True)

    def register_alert_callback(self, callback: Callable[['ResourceMonitor', ResourceType, float, Optional[str]], Coroutine[Any, Any, None]]): # pragma: no cover
        if callback not in self.alert_callbacks: self.alert_callbacks.append(callback)

    def register_resolve_callback(self, callback: Callable[['ResourceMonitor', ResourceType, float, Optional[str]], Coroutine[Any, Any, None]]): # pragma: no cover
        if callback not in self.resolve_callbacks: self.resolve_callbacks.append(callback)

    def get_resource_usage(self) -> Dict[str, Union[float, Dict[str, float]]]: # pragma: no cover
        return {
            "cpu_percent": self.cpu_usage,
            "memory_percent": self.memory_usage,
            "disk_usage_map": self.disk_usage_map.copy()
        }

    def get_resource_limits(self) -> Dict[str, Union[float, Dict[str, float]]]: # pragma: no cover
        return {
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_percent": self.max_memory_percent,
            "disk_threshold_map": self.disk_threshold_map.copy()
        }

    def set_resource_limits(self, max_cpu_percent: Optional[float] = None,
                           max_memory_percent: Optional[float] = None,
                           disk_threshold_map_update: Optional[Dict[str, float]] = None): # pragma: no cover
        if max_cpu_percent is not None:
            if 0 < max_cpu_percent <= 100: self.max_cpu_percent = max_cpu_percent
            else: logger.warning(f"Invalid max_cpu_percent: {max_cpu_percent}. Must be (0, 100].")
        if max_memory_percent is not None:
            if 0 < max_memory_percent <= 100: self.max_memory_percent = max_memory_percent
            else: logger.warning(f"Invalid max_memory_percent: {max_memory_percent}. Must be (0, 100].")

        if disk_threshold_map_update:
            for path_str, threshold in disk_threshold_map_update.items():
                if not (0 < threshold <= 100): # pragma: no cover
                     logger.warning(f"Invalid disk threshold for '{path_str}': {threshold}. Must be (0, 100]. Skipping update for this path.")
                     continue
                try:
                    norm_path = str(Path(path_str).resolve(strict=False))
                    self.disk_threshold_map[norm_path] = threshold
                    if norm_path not in self.disk_usage_map: self.disk_usage_map[norm_path] = 0.0
                    disk_key_tuple = (ResourceType.DISK, norm_path)
                    self._alert_active_flags.pop(disk_key_tuple, None)
                    self._last_alert_timestamp.pop(disk_key_tuple, None)
                except Exception as e: # pragma: no cover
                    logger.error(f"Error processing disk threshold update for '{path_str}': {e}")

        logger.info(f"Resource limits updated. CPU: {self.max_cpu_percent}%, Mem: {self.max_memory_percent}%, Disk: {self.disk_threshold_map}")

    async def _shutdown_monitoring_task(self): # pragma: no cover
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring = False
            self.monitoring_task.cancel()
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=2.0)
            except asyncio.CancelledError:
                logger.info("Resource monitoring task was cancelled as expected during shutdown.")
            except asyncio.TimeoutError: # pragma: no cover
                logger.warning("Timeout waiting for resource monitoring task to fully cancel during shutdown.")
            except Exception as e: # pragma: no cover
                logger.error(f"Error during monitoring task shutdown: {e}", exc_info=True)
        self.monitoring_task = None


    @classmethod
    async def reset_instance_async(cls): # pragma: no cover
        async with cls._lock:
            if cls._instance:
                await cls._instance._shutdown_monitoring_task()
                cls._instance._initialized = False
                cls._instance = None
        logger.debug("ResourceMonitor instance reset asynchronously.")

async def get_resource_monitor_async(config_override: Optional[Config] = None, test_mode: bool = False) -> ResourceMonitor: # pragma: no cover
    if not ResourceMonitor._instance or test_mode:
        async with ResourceMonitor._lock:
            if ResourceMonitor._instance is None or test_mode:
                if test_mode and ResourceMonitor._instance is not None:
                    await ResourceMonitor._instance._shutdown_monitoring_task()
                    ResourceMonitor._instance = None
                ResourceMonitor._instance = ResourceMonitor(config_override=config_override, test_mode=test_mode)
    return ResourceMonitor._instance

def get_resource_monitor(config_override: Optional[Config] = None, test_mode: bool = False) -> ResourceMonitor: # pragma: no cover
    if ResourceMonitor._instance is None or test_mode:
        # This sync getter can't fully handle async shutdown of a previous test instance gracefully
        # Best to use async_get_resource_monitor in test setups if possible
        if test_mode and ResourceMonitor._instance is not None:
             logger.warning("Test mode reset for ResourceMonitor in sync getter: async shutdown of previous task not guaranteed.")
             # Try a best effort sync shutdown if no loop is running
             if ResourceMonitor._instance.monitoring_task and not ResourceMonitor._instance.monitoring_task.done():
                 ResourceMonitor._instance.monitoring_task.cancel() # No await here
             ResourceMonitor._instance = None

        ResourceMonitor._instance = ResourceMonitor(config_override=config_override, test_mode=test_mode)
    return ResourceMonitor._instance
